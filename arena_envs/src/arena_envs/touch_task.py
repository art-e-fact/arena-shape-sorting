"""Touch tasks for arena_envs.

``TouchTask`` (imitation flavour) and ``TouchTaskRL`` (adds observations, rewards and
RL-mode success suppression). Modelled on ``isaaclab_arena.tasks.press_button_task``.
Success = **every** touch target has been touched at least once during the episode.
"""

from __future__ import annotations

import numpy as np
from dataclasses import MISSING

import torch
import warp as wp

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, ManagerTermBase, ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack.mdp.franka_stack_events import randomize_object_pose

from isaaclab_arena.assets.register import register_task
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.observations import observations
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object
from isaaclab_arena.utils.configclass import make_configclass

from arena_envs.touchable import Touchable, compute_is_touched


# ---------------------------------------------------------------------------
# Multi-sphere "touch each" state.
#
# "Touch each sphere at least once" needs per-episode latched state, which a
# stateless MDP term function cannot hold. We use class-based ``ManagerTermBase``
# terms: they keep a per-env, per-sphere boolean buffer, OR-in fresh contacts every
# step, and clear it in ``reset()`` on episode reset (the reward / termination /
# observation managers all call ``reset`` on class terms).
#
# The term classes are referenced by config as ``arena_envs.touch_task:ClassName``,
# so -- like the module-level functions in ``touchable`` -- they must live at module
# scope to survive Isaac Lab's config serialization round-trip.
# ---------------------------------------------------------------------------


class _SphereTouchLatch(ManagerTermBase):
    """Base for terms that track which spheres have been touched this episode.

    Maintains ``self._touched`` of shape ``(num_envs, num_spheres)``; each entry
    latches ``True`` once its sphere's contact force exceeds the threshold, and is
    cleared per-env on reset.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._sensor_names: list[str] = list(cfg.params["sensor_names"])
        self._force_threshold: float = cfg.params["force_threshold"]
        self._touched = torch.zeros(
            self.num_envs, len(self._sensor_names), dtype=torch.bool, device=self.device
        )

    def reset(self, env_ids=None):
        if env_ids is None:
            self._touched[:] = False
        else:
            self._touched[env_ids] = False

    def _update_latch(self, env) -> torch.Tensor:
        for i, sensor_name in enumerate(self._sensor_names):
            self._touched[:, i] |= compute_is_touched(env, sensor_name, self._force_threshold)
        return self._touched


class AllSpheresTouched(_SphereTouchLatch):
    """Termination: success once every sphere has been touched at least once.

    ``rl_training=True`` always returns False so success never ends the episode
    early during RL training (keeps a fixed horizon), mirroring ``touched_termination``.
    """

    def __call__(self, env, sensor_names, force_threshold, rl_training: bool = False) -> torch.Tensor:
        touched = self._update_latch(env)
        if rl_training:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return touched.all(dim=1)


class SpheresTouchedFlags(_SphereTouchLatch):
    """Observation: per-sphere "already touched this episode" flags (num_envs, num_spheres)."""

    def __call__(self, env, sensor_names, force_threshold) -> torch.Tensor:
        return self._update_latch(env).float()


class TouchProgressReward(_SphereTouchLatch):
    """Reward: reach the nearest not-yet-touched sphere, plus a bonus per newly touched one."""

    def __call__(
        self, env, sensor_names, object_names, ee_frame_name, force_threshold, std, touch_bonus
    ) -> torch.Tensor:
        prev = self._touched.clone()
        touched = self._update_latch(env)
        newly = touched & ~prev
        bonus = newly.float().sum(dim=1) * touch_bonus

        # Reach reward toward the closest sphere that has not been touched yet.
        ee_frame = env.scene[ee_frame_name]
        ee_w = wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :]  # (num_envs, 3)
        distances = []
        for object_name in object_names:
            obj = env.scene[object_name]
            obj_w = wp.to_torch(obj.data.root_pos_w)[:, :3]
            distances.append(torch.norm(obj_w - ee_w, dim=1))
        distance = torch.stack(distances, dim=1)  # (num_envs, num_spheres)
        # Mask out already-touched spheres so the agent is guided to the remaining ones.
        distance = distance + touched.float() * 1e6
        nearest = distance.min(dim=1).values
        reaching = 1.0 - torch.tanh(nearest / std)
        return reaching + bonus


@configclass
class TouchTerminationsCfg:
    """Termination terms for the Touch task."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    # Provided by the task at construction time (depends on the touch objects).
    success: TerminationTermCfg = MISSING


@configclass
class TouchEventsCfg:
    """Reset-event terms for the Touch task (multi-sphere scatter placement)."""

    scatter_spheres: EventTermCfg = MISSING


@register_task
class TouchTask(TaskBase):
    """Touch floating targets. Success = every target has been touched at least once."""

    def __init__(
        self,
        touch_objects: list[Touchable],
        episode_length_s: float = 5.0,
        task_description: str | None = None,
        scatter_region_center: tuple[float, float, float] | None = None,
        scatter_region_half_extent: tuple[float, float, float] | None = None,
        min_separation: float = 0.0,
    ):
        super().__init__(episode_length_s=episode_length_s)
        assert len(touch_objects) > 0, "touch_objects must contain at least one target"
        assert all(isinstance(o, Touchable) for o in touch_objects), (
            "every touch_object must be an instance of Touchable"
        )
        self.touch_objects = list(touch_objects)
        self.task_description = (
            f"Touch all {len(self.touch_objects)} spheres" if task_description is None else task_description
        )
        self._events_cfg = self._make_events_cfg(
            scatter_region_center, scatter_region_half_extent, min_separation
        )

    def _sensor_names(self) -> list[str]:
        return [o.get_touch_sensor_name() for o in self.touch_objects]

    def _force_threshold(self) -> float:
        return self.touch_objects[0].touch_force_threshold

    def _make_events_cfg(self, center, half_extent, min_separation):
        """Reset event that scatters all spheres in a shared box region with min-separation."""
        if center is None or half_extent is None:
            return None
        cx, cy, cz = center
        hx, hy, hz = half_extent
        pose_range = {
            "x": (cx - hx, cx + hx),
            "y": (cy - hy, cy + hy),
            "z": (cz - hz, cz + hz),
        }
        return TouchEventsCfg(
            scatter_spheres=EventTermCfg(
                func=randomize_object_pose,
                mode="reset",
                params={
                    "asset_cfgs": [SceneEntityCfg(o.name) for o in self.touch_objects],
                    "min_separation": min_separation,
                    "pose_range": pose_range,
                },
            )
        )

    def get_scene_cfg(self):
        # Contribute one touch contact sensor per target to the scene.
        fields = [
            (o.get_touch_sensor_name(), ContactSensorCfg, o.get_touch_sensor_cfg())
            for o in self.touch_objects
        ]
        SceneCfg = make_configclass("TouchSceneCfg", fields)
        return SceneCfg()

    def get_termination_cfg(self):
        success = TerminationTermCfg(
            func=AllSpheresTouched,
            params={
                "sensor_names": self._sensor_names(),
                "force_threshold": self._force_threshold(),
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
            lookat_object=self.touch_objects[0],
            offset=np.array([-1.5, -1.5, 1.5]),
        )


@register_task
class TouchTaskRL(TouchTask):
    """RL version of :class:`TouchTask` with reach + touch rewards."""

    def __init__(
        self,
        touch_objects: list[Touchable],
        embodiment: EmbodimentBase,
        episode_length_s: float = 5.0,
        rl_training_mode: bool = True,
        task_description: str | None = None,
        scatter_region_center: tuple[float, float, float] | None = None,
        scatter_region_half_extent: tuple[float, float, float] | None = None,
        min_separation: float = 0.0,
    ):
        """Initialize the Touch RL task.

        Args:
            touch_objects: The touchable targets. Success requires touching each once.
            embodiment: The robot embodiment.
            episode_length_s: Episode length in seconds.
            rl_training_mode: If True, disables success termination during training.
            task_description: Optional override for the task description.
            scatter_region_center: Center (x, y, z) of the shared spawn box.
            scatter_region_half_extent: Half-extent (x, y, z) of the shared spawn box.
            min_separation: Minimum distance between spheres when scattering.
        """
        self.rl_training_mode = rl_training_mode
        self.embodiment = embodiment

        super().__init__(
            touch_objects=touch_objects,
            episode_length_s=episode_length_s,
            task_description=task_description,
            scatter_region_center=scatter_region_center,
            scatter_region_half_extent=scatter_region_half_extent,
            min_separation=min_separation,
        )

        robot_name = self.embodiment.get_embodiment_name_in_scene()
        self.observation_cfg = TouchObservationsCfg(touch_objects=self.touch_objects, robot_name=robot_name)
        self.rewards_cfg = TouchRewardCfg(
            touch_objects=self.touch_objects,
            ee_frame_name=self.embodiment.get_ee_frame_name(self.embodiment.get_arm_mode()),
        )
        self.termination_cfg = self.make_rl_termination_cfg()

    def make_rl_termination_cfg(self):
        """Success termination that is suppressed while ``rl_training_mode`` is on."""
        success = TerminationTermCfg(
            func=AllSpheresTouched,
            params={
                "sensor_names": self._sensor_names(),
                "force_threshold": self._force_threshold(),
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

    def __init__(self, touch_objects: list[Touchable], robot_name: str):
        sensor_names = [o.get_touch_sensor_name() for o in touch_objects]
        force_threshold = touch_objects[0].touch_force_threshold

        # One position term per sphere (in the robot root frame) + latched touched flags,
        # so the policy can tell which targets still need to be touched.
        fields = [
            (
                f"object_position_{i}",
                ObsTerm,
                ObsTerm(
                    func=observations.object_position_in_frame,
                    params={
                        "root_frame_cfg": SceneEntityCfg(robot_name),
                        "object_cfg": SceneEntityCfg(o.name),
                    },
                ),
            )
            for i, o in enumerate(touch_objects)
        ]
        fields.append(
            (
                "touched_flags",
                ObsTerm,
                ObsTerm(
                    func=SpheresTouchedFlags,
                    params={"sensor_names": sensor_names, "force_threshold": force_threshold},
                ),
            )
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

        TaskObsCfg = make_configclass(
            "TouchTaskObsCfg", fields, bases=(ObsGroup,), namespace={"__post_init__": __post_init__}
        )
        self.task_obs = TaskObsCfg()


@configclass
class TouchRewardCfg:
    """Reward terms for the Touch task: reach the nearest untouched sphere, then touch each."""

    touch_progress: RewardTermCfg = MISSING

    def __init__(self, touch_objects: list[Touchable], ee_frame_name: str):
        self.touch_progress = RewardTermCfg(
            func=TouchProgressReward,
            params={
                "sensor_names": [o.get_touch_sensor_name() for o in touch_objects],
                "object_names": [o.name for o in touch_objects],
                "ee_frame_name": ee_frame_name,
                "force_threshold": touch_objects[0].touch_force_threshold,
                "std": 0.1,
                "touch_bonus": 5.0,
            },
            weight=1.0,
        )
