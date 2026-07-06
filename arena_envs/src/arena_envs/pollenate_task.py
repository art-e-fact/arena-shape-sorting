"""Pollenate tasks for arena_envs.

``PollenateTask`` (imitation flavour). Success = the EE has reached within threshold
of every ``approach_frame_*`` on the plant at least once during the episode.
"""

from __future__ import annotations

from dataclasses import MISSING

import numpy as np

import torch
import warp as wp

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import ManagerTermBase, ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_arena.assets.register import register_task
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object

from arena_envs.pollenateable import Pollenateable, get_frame_pos_w, get_frame_world_poses


# ---------------------------------------------------------------------------
# Module-level observation helpers (must stay at module scope for config reload).
# ---------------------------------------------------------------------------


def approach_frame_poses_in_frame(
    env: ManagerBasedEnv,
    root_frame_cfg: SceneEntityCfg,
    plant_name: str,
    frame_names: list[str],
    frame_suffixes: dict[str, str],
) -> torch.Tensor:
    """All approach-frame poses in the robot root frame, shape ``(num_envs, N, 7)``.

    Layout per pose: ``(x, y, z, qw, qx, qy, qz)`` (Isaac Lab quaternion layout).
    """
    root = env.scene[root_frame_cfg.name]
    root_pos_w = wp.to_torch(root.data.root_pos_w)
    root_quat_w = wp.to_torch(root.data.root_quat_w)
    poses_b = []
    for name in frame_names:
        pos_w, quat_w = get_frame_world_poses(env, plant_name, frame_suffixes[name])
        pos_b, quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, pos_w, quat_w)
        poses_b.append(torch.cat([pos_b, quat_b], dim=-1))
    return torch.stack(poses_b, dim=1).reshape(env.num_envs, -1)


# ---------------------------------------------------------------------------
# Per-frame reach latch (mirrors ``_SphereTouchLatch`` in ``touch_task``).
# ---------------------------------------------------------------------------


class _ApproachFrameReachLatch(ManagerTermBase):
    """Tracks which approach frames the EE has reached this episode."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._frame_names: list[str] = list(cfg.params["frame_names"])
        self._plant_name: str = cfg.params["plant_name"]
        self._frame_suffixes: dict[str, str] = dict(cfg.params["frame_suffixes"])
        self._ee_frame_name: str = cfg.params["ee_frame_name"]
        self._reach_threshold: float = cfg.params["reach_threshold"]
        self._reached = torch.zeros(
            self.num_envs, len(self._frame_names), dtype=torch.bool, device=self.device
        )

    def reset(self, env_ids=None):
        if env_ids is None:
            self._reached[:] = False
        else:
            self._reached[env_ids] = False

    def _ee_pos_w(self, env) -> torch.Tensor:
        ee_frame = env.scene[self._ee_frame_name]
        return wp.to_torch(ee_frame.data.target_pos_w)[..., 0, :]  # (num_envs, 3)

    def _update_latch(self, env) -> torch.Tensor:
        ee_w = self._ee_pos_w(env)
        for i, frame_name in enumerate(self._frame_names):
            frame_w = get_frame_pos_w(env, self._plant_name, self._frame_suffixes[frame_name])
            dist = torch.norm(frame_w - ee_w, dim=1)
            self._reached[:, i] |= dist < self._reach_threshold
        return self._reached


class AllApproachFramesReached(_ApproachFrameReachLatch):
    """Termination: success once every approach frame has been reached at least once."""

    def __call__(self, env, frame_names, frame_suffixes, plant_name, ee_frame_name, reach_threshold) -> torch.Tensor:
        return self._update_latch(env).all(dim=1)


class ApproachFramesReachedFlags(_ApproachFrameReachLatch):
    """Observation: per-frame "already reached this episode" flags ``(num_envs, N)``."""

    def __call__(self, env, frame_names, frame_suffixes, plant_name, ee_frame_name, reach_threshold) -> torch.Tensor:
        return self._update_latch(env).float()


@configclass
class PollenateTerminationsCfg:
    """Termination terms for the Pollenate task."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    success: TerminationTermCfg = MISSING


@configclass
class PollenateObservationsCfg:
    """Observation specifications for the Pollenate task."""

    task_obs: ObsGroup = MISSING

    def __init__(
        self,
        frame_names: list[str],
        frame_suffixes: dict[str, str],
        plant_name: str,
        robot_name: str,
        ee_frame_name: str,
        reach_threshold: float,
    ):
        @configclass
        class TaskObsCfg(ObsGroup):
            # (num_envs, N * 7): flattened xyz + quat wxyz per frame in the robot root frame.
            approach_frame_poses = ObsTerm(
                func=approach_frame_poses_in_frame,
                params={
                    "root_frame_cfg": SceneEntityCfg(robot_name),
                    "plant_name": plant_name,
                    "frame_names": frame_names,
                    "frame_suffixes": frame_suffixes,
                },
            )
            reached_flags = ObsTerm(
                func=ApproachFramesReachedFlags,
                params={
                    "frame_names": frame_names,
                    "frame_suffixes": frame_suffixes,
                    "plant_name": plant_name,
                    "ee_frame_name": ee_frame_name,
                    "reach_threshold": reach_threshold,
                },
            )

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True

        self.task_obs = TaskObsCfg()


@register_task
class PollenateTask(TaskBase):
    """Reach every approach frame on a pollinateable plant."""

    def __init__(
        self,
        pollinate_object: Pollenateable,
        robot_name: str = "robot",
        ee_frame_name: str = "ee_frame",
        episode_length_s: float = 30.0,
        task_description: str | None = None,
    ):
        super().__init__(episode_length_s=episode_length_s)
        assert isinstance(pollinate_object, Pollenateable), "pollinate_object must be a Pollenateable"
        self.pollinate_object = pollinate_object
        self.robot_name = robot_name
        self.ee_frame_name = ee_frame_name
        self.approach_frames = pollinate_object.get_approach_frames()
        assert len(self.approach_frames) > 0, "pollinate_object must expose at least one approach frame"
        n = len(self.approach_frames)
        self.task_description = (
            f"Reach all {n} approach frames on {pollinate_object.name}"
            if task_description is None
            else task_description
        )
        self.observation_cfg = PollenateObservationsCfg(
            frame_names=[f.name for f in self.approach_frames],
            frame_suffixes={f.name: f.suffix for f in self.approach_frames},
            plant_name=pollinate_object.name,
            robot_name=robot_name,
            ee_frame_name=ee_frame_name,
            reach_threshold=pollinate_object.reach_distance_threshold,
        )

    def _frame_names(self) -> list[str]:
        return [f.name for f in self.approach_frames]

    def get_scene_cfg(self):
        pass

    def get_termination_cfg(self):
        success = TerminationTermCfg(
            func=AllApproachFramesReached,
            params={
                "frame_names": self._frame_names(),
                "frame_suffixes": {f.name: f.suffix for f in self.approach_frames},
                "plant_name": self.pollinate_object.name,
                "ee_frame_name": self.ee_frame_name,
                "reach_threshold": self.pollinate_object.reach_distance_threshold,
            },
        )
        return PollenateTerminationsCfg(success=success)

    def get_events_cfg(self):
        return None

    def get_observation_cfg(self):
        return self.observation_cfg

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError("Function not implemented yet.")

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.pollinate_object,
            offset=np.array([-1.5, -1.5, 1.5]),
        )
