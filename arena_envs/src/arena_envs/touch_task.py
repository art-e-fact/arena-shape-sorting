"""Touch tasks for arena_envs.

``TouchTask`` (imitation flavour) and ``TouchTaskRL`` (adds observations, rewards and
RL-mode success suppression). Modelled on ``isaaclab_arena.tasks.press_button_task``.
Success = the touch target is touched (see :class:`arena_envs.touchable.Touchable`).
"""

from __future__ import annotations

import math
import random
import numpy as np
from dataclasses import MISSING

import torch

import isaaclab.envs.mdp as mdp_isaac_lab
import isaaclab.utils.math as math_utils
from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import EventTermCfg, ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.register import register_task
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.observations import observations
from isaaclab_arena.tasks.rewards import rewards
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object
from isaaclab_arena.utils.configclass import make_configclass

from isaaclab_arena.utils.pose import Pose

from arena_envs.touchable import Touchable, touched_reward, touched_termination


# ---------------------------------------------------------------------------
# Per-env uniform-in-ball spawn randomisation.
#
# Must be a module-level function (not a method) so Isaac Lab can serialise
# the EventTermCfg func reference as "arena_envs.touch_task:randomize_sphere_in_ball".
# ---------------------------------------------------------------------------


def randomize_sphere_in_ball(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    center: tuple[float, float, float],
    radius: float,
) -> None:
    """Reset event: teleport the sphere to a uniform random point inside a ball.

    Sampling uses the cube-root radius trick so the density is uniform in volume
    (not concentrated at the centre).

    Args:
        env: The manager-based environment.
        env_ids: Indices of the environments being reset.
        asset_cfg: Scene entity config identifying the sphere rigid-body.
        center: (x, y, z) mean spawn position in env-local coordinates.
        radius: Ball radius in metres. 0 ≤ r, positions clipped to the surface when
            the sampled radius exceeds this value.
    """
    if env_ids is None or len(env_ids) == 0:
        return

    asset = env.scene[asset_cfg.name]
    cx, cy, cz = center

    for env_id in env_ids.tolist():
        # Sample a unit direction uniformly on the sphere.
        # Draw from a 3-D normal distribution and normalise.
        dx, dy, dz = random.gauss(0.0, 1.0), random.gauss(0.0, 1.0), random.gauss(0.0, 1.0)
        length = math.sqrt(dx * dx + dy * dy + dz * dz)
        if length < 1e-8:
            dx, dy, dz = 1.0, 0.0, 0.0
            length = 1.0
        dx, dy, dz = dx / length, dy / length, dz / length

        # Scale by a uniform-in-volume radius (cube-root trick).
        r = radius * (random.random() ** (1.0 / 3.0))

        # Env-local position + world origin of this env sub-scene.
        origin = env.scene.env_origins[env_id]  # [3]
        pos_x = cx + dx * r + origin[0].item()
        pos_y = cy + dy * r + origin[1].item()
        pos_z = cz + dz * r + origin[2].item()

        position = torch.tensor([[pos_x, pos_y, pos_z]], device=env.device)
        # Identity quaternion (xyzw); kinematic sphere has no meaningful orientation.
        orientation = math_utils.quat_from_euler_xyz(
            torch.zeros(1, device=env.device),
            torch.zeros(1, device=env.device),
            torch.zeros(1, device=env.device),
        )
        root_pose = torch.cat([position, orientation], dim=-1)
        asset.write_root_pose_to_sim_index(
            root_pose=root_pose,
            env_ids=torch.tensor([env_id], device=env.device),
        )
        asset.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(1, 6, device=env.device),
            env_ids=torch.tensor([env_id], device=env.device),
        )


@configclass
class TouchTerminationsCfg:
    """Termination terms for the Touch task."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    # Provided by the task at construction time (depends on the touch object).
    success: TerminationTermCfg = MISSING


@configclass
class TouchEventsCfg:
    """Reset-event terms for the Touch task (sphere spawn randomisation)."""

    randomize_sphere: EventTermCfg = MISSING


@register_task
class TouchTask(TaskBase):
    """Touch a floating target. Success = the object is touched."""

    def __init__(
        self,
        touch_object: Touchable,
        episode_length_s: float = 5.0,
        task_description: str | None = None,
    ):
        super().__init__(episode_length_s=episode_length_s)
        assert isinstance(touch_object, Touchable), "touch_object must be an instance of Touchable"
        self.touch_object = touch_object
        self.task_description = (
            f"Touch the {touch_object.name}" if task_description is None else task_description
        )
        self._events_cfg = None

    def get_scene_cfg(self):
        # Contribute the object's touch contact sensor to the scene.
        sensor_name = self.touch_object.get_touch_sensor_name()
        sensor_cfg = self.touch_object.get_touch_sensor_cfg()
        SceneCfg = make_configclass("TouchSceneCfg", [(sensor_name, ContactSensorCfg, sensor_cfg)])
        return SceneCfg()

    def get_termination_cfg(self):
        success = TerminationTermCfg(
            func=touched_termination,
            params={
                "sensor_name": self.touch_object.get_touch_sensor_name(),
                "force_threshold": self.touch_object.touch_force_threshold,
            },
        )
        return TouchTerminationsCfg(success=success)

    def get_events_cfg(self):
        return self._events_cfg

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError("Function not implemented yet.")

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.touch_object,
            offset=np.array([-1.5, -1.5, 1.5]),
        )


@register_task
class TouchTaskRL(TouchTask):
    """RL version of :class:`TouchTask` with reach + touch rewards."""

    def __init__(
        self,
        touch_object: Touchable,
        embodiment: EmbodimentBase,
        episode_length_s: float = 5.0,
        rl_training_mode: bool = True,
        task_description: str | None = None,
        sphere_spawn_radius: float = 0.0,
    ):
        """Initialize the Touch RL task.

        Args:
            touch_object: The touchable target.
            embodiment: The robot embodiment.
            episode_length_s: Episode length in seconds.
            rl_training_mode: If True, disables success termination during training.
            task_description: Optional override for the task description.
        """
        self.rl_training_mode = rl_training_mode
        self.embodiment = embodiment

        super().__init__(
            touch_object=touch_object,
            episode_length_s=episode_length_s,
            task_description=task_description,
        )

        if sphere_spawn_radius > 0.0:
            initial_pose = touch_object.get_initial_pose()
            center = initial_pose.position_xyz if isinstance(initial_pose, Pose) else (0.5, 0.0, 0.3)
            self._events_cfg = TouchEventsCfg(
                randomize_sphere=EventTermCfg(
                    func=randomize_sphere_in_ball,
                    mode="reset",
                    params={
                        "asset_cfg": SceneEntityCfg(touch_object.name),
                        "center": center,
                        "radius": sphere_spawn_radius,
                    },
                )
            )

        robot_name = self.embodiment.get_embodiment_name_in_scene()
        self.observation_cfg = TouchObservationsCfg(touch_object=self.touch_object, robot_name=robot_name)
        self.rewards_cfg = TouchRewardCfg(
            touch_object=self.touch_object,
            ee_frame_name=self.embodiment.get_ee_frame_name(self.embodiment.get_arm_mode()),
        )
        self.termination_cfg = self.make_rl_termination_cfg()

    def make_rl_termination_cfg(self):
        """Success termination that is suppressed while ``rl_training_mode`` is on."""
        success = TerminationTermCfg(
            func=touched_termination,
            params={
                "sensor_name": self.touch_object.get_touch_sensor_name(),
                "force_threshold": self.touch_object.touch_force_threshold,
                "rl_training": self.rl_training_mode,
            },
        )
        return TouchTerminationsCfg(success=success)

    def get_observation_cfg(self):
        return self.observation_cfg

    def get_rewards_cfg(self):
        return self.rewards_cfg

    def get_termination_cfg(self):
        return self.termination_cfg


@configclass
class TouchObservationsCfg:
    """Observation specifications for the Touch task."""

    task_obs: ObsGroup = MISSING

    def __init__(self, touch_object: Touchable, robot_name: str):
        @configclass
        class TaskObsCfg(ObsGroup):
            object_position = ObsTerm(
                func=observations.object_position_in_frame,
                params={
                    "root_frame_cfg": SceneEntityCfg(robot_name),
                    "object_cfg": SceneEntityCfg(touch_object.name),
                },
            )

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True

        self.task_obs = TaskObsCfg()


@configclass
class TouchRewardCfg:
    """Reward terms for the Touch task: reach the target, then touch it."""

    reaching_object: RewardTermCfg = MISSING
    touching_object: RewardTermCfg = MISSING

    def __init__(self, touch_object: Touchable, ee_frame_name: str):
        self.reaching_object = RewardTermCfg(
            func=rewards.object_ee_distance,
            params={
                "std": 0.1,
                "object_cfg": SceneEntityCfg(touch_object.name),
                "ee_frame_cfg": SceneEntityCfg(ee_frame_name),
            },
            weight=1.0,
        )
        self.touching_object = RewardTermCfg(
            func=touched_reward,
            params={
                "sensor_name": touch_object.get_touch_sensor_name(),
                "force_threshold": touch_object.touch_force_threshold,
            },
            weight=5.0,
        )
