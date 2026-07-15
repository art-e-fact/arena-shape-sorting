"""RL pick-and-place task for shape sorting.

Subclasses Arena's :class:`PickAndPlaceTask` and adds the MDP pieces needed for
RSL-RL training: dense rewards, task observations, and success termination that
can be suppressed during training (fixed episode horizon).
"""

from __future__ import annotations

from dataclasses import MISSING

import torch
import warp as wp

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import ManagerTermBase, ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.assets.register import register_task
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.tasks.observations import observations
from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask, TerminationsCfg
from isaaclab_arena.tasks.terminations import (
    SuccessMode,
    check_success,
    object_on_destination,
    objects_in_proximity,
)


class _SpawnHeightLatch(ManagerTermBase):
    """Tracks each object's spawn height so lift rewards work with relation-based placement."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._spawn_z = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids=None):
        object_asset = self._env.scene[self.cfg.params["object_cfg"].name]
        spawn_z = wp.to_torch(object_asset.data.root_pos_w)[:, 2]
        if env_ids is None:
            self._spawn_z = spawn_z.clone()
        else:
            self._spawn_z[env_ids] = spawn_z[env_ids]

    def height_gain(self, env: ManagerBasedEnv, object_cfg: SceneEntityCfg) -> torch.Tensor:
        """Positive height delta since episode reset [m]."""
        object_asset = env.scene[object_cfg.name]
        current_z = wp.to_torch(object_asset.data.root_pos_w)[:, 2]
        return torch.clamp(current_z - self._spawn_z, min=0.0)


class ObjectHeightGainReward(_SpawnHeightLatch):
    """Continuous reward for lifting the object above its reset height."""

    def __call__(self, env, object_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
        return torch.tanh(self.height_gain(env, object_cfg) / std)


def gripper_object_distance(
    env: ManagerBasedEnv,
    std: float,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    ee_body_name: str,
) -> torch.Tensor:
    """Reach reward using the gripper body, matching Droid policy observations.

    Arena's ``object_ee_distance`` reads ``ee_frame`` target index 0, which on the
    Droid embodiment is anchored to ``panda_link0`` (shoulder) rather than the
    gripper. Arm motion barely changes that signal, so RL gets no reach gradient.
    """
    object_asset = env.scene[object_cfg.name]
    robot = env.scene[robot_cfg.name]
    body_idx = robot.data.body_names.index(ee_body_name)
    object_pos_w = wp.to_torch(object_asset.data.root_pos_w)
    ee_pos_w = wp.to_torch(robot.data.body_pos_w)[:, body_idx, :]
    distance = torch.norm(object_pos_w - ee_pos_w, dim=1)
    return 1.0 - torch.tanh(distance / std)


def object_to_destination_distance(
    env: ManagerBasedEnv,
    std: float,
    object_cfg: SceneEntityCfg,
    destination_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Dense reward for moving the object toward the destination (no lift gate)."""
    object_asset = env.scene[object_cfg.name]
    destination = env.scene[destination_cfg.name]
    object_pos = wp.to_torch(object_asset.data.root_pos_w)
    destination_pos = wp.to_torch(destination.data.root_pos_w)
    distance = torch.norm(object_pos - destination_pos, dim=1)
    return 1.0 - torch.tanh(distance / std)


def pick_and_place_rl_success(
    env,
    rl_training: bool,
    object_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    destination_cfg: SceneEntityCfg,
    force_threshold: float,
    velocity_threshold: float,
    max_x_separation: float | None,
    max_y_separation: float | None,
    max_z_separation: float | None,
) -> torch.Tensor:
    """Success termination for pick-and-place, suppressed while training."""
    if rl_training:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    predicates = [
        TerminationTermCfg(
            func=object_on_destination,
            params={
                "object_cfg": object_cfg,
                "contact_sensor_cfg": contact_sensor_cfg,
                "force_threshold": force_threshold,
                "velocity_threshold": velocity_threshold,
            },
        ),
    ]
    if max_x_separation is not None:
        predicates.append(
            TerminationTermCfg(
                func=objects_in_proximity,
                params={
                    "object_cfg": object_cfg,
                    "target_object_cfg": destination_cfg,
                    "max_x_separation": max_x_separation,
                    "max_y_separation": max_y_separation,
                    "max_z_separation": max_z_separation,
                },
            )
        )
    return check_success(env, predicates=predicates, mode=SuccessMode.ALL)


@register_task
class PickAndPlaceTaskRL(PickAndPlaceTask):
    """Pick-and-place with dense rewards and goal-conditioned observations."""

    def __init__(
        self,
        pick_up_object: Asset,
        destination_location: Asset,
        background_scene: Asset,
        embodiment: EmbodimentBase,
        episode_length_s: float | None = 20.0,
        rl_training_mode: bool = True,
        lift_height_std: float = 0.02,
        ee_body_name: str = "base_link",
        force_threshold: float = 0.1,
        velocity_threshold: float = 0.1,
        max_separation: tuple[float, float, float] | None = (0.10, 0.10, 0.20),
        task_description: str | None = None,
    ):
        """Initialize the pick-and-place RL task.

        Args:
            pick_up_object: Object to pick up.
            destination_location: Target container or surface.
            background_scene: Table / scene used for drop detection.
            embodiment: Robot embodiment (needed for EE frame and observations).
            episode_length_s: Episode length in seconds.
            rl_training_mode: If True, success does not end the episode early.
            lift_height_std: Tanh scale for the continuous lift reward [m].
            ee_body_name: Robot body used for the reach reward (Droid gripper: ``base_link``).
            force_threshold: Contact force threshold for placement success.
            velocity_threshold: Velocity threshold for placement success.
            max_separation: Optional (x, y, z) proximity thresholds for bowl placement.
            task_description: Optional override for the task description.
        """
        self.rl_training_mode = rl_training_mode
        self.embodiment = embodiment
        self.lift_height_std = lift_height_std
        self.ee_body_name = ee_body_name

        super().__init__(
            pick_up_object=pick_up_object,
            destination_location=destination_location,
            background_scene=background_scene,
            episode_length_s=episode_length_s,
            force_threshold=force_threshold,
            velocity_threshold=velocity_threshold,
            max_separation=max_separation,
            task_description=task_description,
        )

        robot_name = embodiment.get_embodiment_name_in_scene()

        self.observation_cfg = PickAndPlaceObservationsCfg(
            pick_up_object=pick_up_object,
            destination_location=destination_location,
            robot_name=robot_name,
        )
        self.rewards_cfg = PickAndPlaceRewardCfg(
            pick_up_object=pick_up_object,
            destination_location=destination_location,
            lift_height_std=self.lift_height_std,
            robot_name=robot_name,
            ee_body_name=self.ee_body_name,
        )
        self.termination_cfg = self._make_rl_termination_cfg()

    def _make_rl_termination_cfg(self):
        """Keep drop detection; gate success on placement during evaluation only."""
        max_sep = self.max_separation
        success = TerminationTermCfg(
            func=pick_and_place_rl_success,
            params={
                "rl_training": self.rl_training_mode,
                "object_cfg": SceneEntityCfg(self.pick_up_object.name),
                "contact_sensor_cfg": SceneEntityCfg("pick_up_object_contact_sensor"),
                "destination_cfg": SceneEntityCfg(self.destination_location.name),
                "force_threshold": self.force_threshold,
                "velocity_threshold": self.velocity_threshold,
                "max_x_separation": max_sep[0] if max_sep is not None else None,
                "max_y_separation": max_sep[1] if max_sep is not None else None,
                "max_z_separation": max_sep[2] if max_sep is not None else None,
            },
        )
        object_dropped = TerminationTermCfg(
            func=mdp_isaac_lab.root_height_below_minimum,
            params={
                "minimum_height": self.background_scene.object_min_z,
                "asset_cfg": SceneEntityCfg(self.pick_up_object.name),
            },
        )
        return TerminationsCfg(success=success, object_dropped=object_dropped)

    def get_observation_cfg(self):
        return self.observation_cfg

    def get_rewards_cfg(self):
        return self.rewards_cfg

    def get_termination_cfg(self):
        return self.termination_cfg

    def get_viewer_cfg(self) -> ViewerCfg:
        # Relation-placed objects may expose PosePerEnv, which breaks look-at helpers.
        return ViewerCfg(eye=(1.5, 0.0, 1.0), lookat=(0.2, 0.0, 0.0))


@configclass
class PickAndPlaceObservationsCfg:
    """Task observations: object and destination poses in the robot root frame."""

    task_obs: ObsGroup = MISSING

    def __init__(self, pick_up_object: Asset, destination_location: Asset, robot_name: str):
        @configclass
        class TaskObsCfg(ObsGroup):
            object_position = ObsTerm(
                func=observations.object_position_in_frame,
                params={
                    "root_frame_cfg": SceneEntityCfg(robot_name),
                    "object_cfg": SceneEntityCfg(pick_up_object.name),
                },
            )
            destination_position = ObsTerm(
                func=observations.object_position_in_frame,
                params={
                    "root_frame_cfg": SceneEntityCfg(robot_name),
                    "object_cfg": SceneEntityCfg(destination_location.name),
                },
            )

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True

        self.task_obs = TaskObsCfg()


@configclass
class PickAndPlaceRewardCfg:
    """Dense shaping: reach, lift, then guide the object into the destination."""

    reaching_object: RewardTermCfg = MISSING
    lifting_object: RewardTermCfg = MISSING
    object_to_destination: RewardTermCfg = MISSING
    object_to_destination_fine: RewardTermCfg = MISSING

    def __init__(
        self,
        pick_up_object: Asset,
        destination_location: Asset,
        lift_height_std: float,
        robot_name: str,
        ee_body_name: str,
    ):
        self.reaching_object = RewardTermCfg(
            func=gripper_object_distance,
            params={
                "std": 0.1,
                "object_cfg": SceneEntityCfg(pick_up_object.name),
                "robot_cfg": SceneEntityCfg(robot_name),
                "ee_body_name": ee_body_name,
            },
            weight=5.0,
        )
        # Continuous lift shaping: small height gains produce non-zero reward immediately.
        self.lifting_object = RewardTermCfg(
            func=ObjectHeightGainReward,
            params={
                "object_cfg": SceneEntityCfg(pick_up_object.name),
                "std": lift_height_std,
            },
            weight=15.0,
        )
        # Ungated distance-to-destination: sliding/pushing toward the bowl also earns reward.
        self.object_to_destination = RewardTermCfg(
            func=object_to_destination_distance,
            params={
                "std": 0.2,
                "object_cfg": SceneEntityCfg(pick_up_object.name),
                "destination_cfg": SceneEntityCfg(destination_location.name),
            },
            weight=16.0,
        )
        self.object_to_destination_fine = RewardTermCfg(
            func=object_to_destination_distance,
            params={
                "std": 0.05,
                "object_cfg": SceneEntityCfg(pick_up_object.name),
                "destination_cfg": SceneEntityCfg(destination_location.name),
            },
            weight=5.0,
        )
