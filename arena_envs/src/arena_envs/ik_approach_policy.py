"""Scripted cuRobo approach policy for pollinate tasks.

Drives the Franka end-effector to each approach-frame waypoint in turn, using
``approach_frame_poses`` from ``PollenateObservationsCfg`` as the sole source of
targets. Orientation is ignored for now: only the xyz position of each frame is
used, and the current EE orientation is held.

For each target, cuRobo plans a collision-free EE path once; the policy follows
every waypoint in that plan via the planner's iterator, then holds at the goal
with differential IK until the frame is latched as reached.

Emits the differential-IK arm action layout ``[dpos(3), drot(3), gripper(1)]``.
Assumes a single environment (env 0) with the Franka base at the world origin.

Run with:
    --policy_type arena_envs.ik_approach_policy.IkApproachPolicy
    [--ik_approach_step_m 0.05]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import warp as wp

from isaaclab.utils.math import combine_frame_transforms

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase


@dataclass
class IkApproachPolicyArgs:
    step_m: float = 0.05

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "IkApproachPolicyArgs":
        return cls(step_m=args.ik_approach_step_m)


@register_policy
class IkApproachPolicy(PolicyBase):
    """Drives the Franka EE to pollinate approach frames using cuRobo and task observations."""

    name = "ik_approach"
    config_class = IkApproachPolicyArgs

    def __init__(self, config: IkApproachPolicyArgs):
        super().__init__(config)
        self._planner = None
        self._active_target_idx: int | None = None

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._planner is not None:
            self._planner.reset_plan()
        self._active_target_idx = None

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _ee_pose_w(base) -> torch.Tensor:
        """Current end-effector pose in the world frame as a 4x4 matrix (env 0)."""
        import isaaclab.utils.math as PoseUtils

        ee = base.scene["ee_frame"]
        pos = wp.to_torch(ee.data.target_pos_w)[0, 0]
        quat = wp.to_torch(ee.data.target_quat_w)[0, 0]
        return PoseUtils.make_pose(pos, PoseUtils.matrix_from_quat(quat.unsqueeze(0))[0])

    def _delta_action(self, base, target_pose: torch.Tensor, gripper: float = 1.0) -> torch.Tensor:
        """Turn a 4x4 world target pose into a clamped relative-pose action (1, 7)."""
        import isaaclab.utils.math as PoseUtils

        curr = self._ee_pose_w(base)
        dpos = torch.clamp(target_pose[:3, 3] - curr[:3, 3], -self.config.step_m, self.config.step_m)
        delta_rot = target_pose[:3, :3] @ curr[:3, :3].transpose(-1, -2)
        drot = PoseUtils.axis_angle_from_quat(PoseUtils.quat_from_matrix(delta_rot))
        grip = torch.tensor([gripper], device=dpos.device, dtype=dpos.dtype)
        return torch.cat([dpos, drot, grip]).unsqueeze(0)

    @staticmethod
    def _parse_task_obs(task_obs) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(approach_frame_poses, reached_flags)`` for env 0.

        ``approach_frame_poses`` has shape ``(N, 7)`` as ``(x, y, z, qw, qx, qy, qz)``
        in the robot root frame. ``reached_flags`` has shape ``(N,)``.
        """
        if isinstance(task_obs, dict):
            poses = task_obs["approach_frame_poses"][0].reshape(-1, 7)
            reached = task_obs["reached_flags"][0]
            return poses, reached

        flat = task_obs[0]
        n_frames = flat.shape[-1] // 8
        poses = flat[: n_frames * 7].reshape(n_frames, 7)
        reached = flat[n_frames * 7 :]
        return poses, reached

    def _waypoints_w(self, base, poses_b: torch.Tensor) -> list[torch.Tensor]:
        """Convert root-frame approach poses to world-frame 4x4 targets (position only)."""
        import isaaclab.utils.math as PoseUtils

        robot = base.scene["robot"]
        root_pos_w = wp.to_torch(robot.data.root_pos_w)[0:1]
        root_quat_w = wp.to_torch(robot.data.root_quat_w)[0:1]
        ee_rot = self._ee_pose_w(base)[:3, :3]

        targets: list[torch.Tensor] = []
        for pos_b in poses_b[:, :3]:
            pos_w, _ = combine_frame_transforms(root_pos_w, root_quat_w, pos_b.unsqueeze(0))
            targets.append(PoseUtils.make_pose(pos_w[0], ee_rot))
        return targets

    @staticmethod
    def _curobo_pose_to_matrix(pose) -> torch.Tensor:
        """Convert a cuRobo ``Pose`` to a 4x4 matrix on the env device."""
        import isaaclab.utils.math as PoseUtils

        position = pose.position if isinstance(pose.position, torch.Tensor) else torch.tensor(pose.position)
        rotation = (
            pose.get_rotation()
            if isinstance(pose.get_rotation(), torch.Tensor)
            else torch.tensor(pose.get_rotation())
        )
        return PoseUtils.make_pose(position, rotation)[0]

    def _has_active_plan(self) -> bool:
        return self._planner is not None and self._planner.has_next_waypoint()

    # -- main ------------------------------------------------------------------

    def get_action(self, env, observation) -> torch.Tensor:
        base = env.unwrapped
        poses_b, reached_flags = self._parse_task_obs(observation["task_obs"])
        frame_targets_w = self._waypoints_w(base, poses_b)
        ee_pose = self._ee_pose_w(base)
        ee_pos = ee_pose[:3, 3]

        reached = {i for i, flag in enumerate(reached_flags) if flag > 0.5}
        remaining = [i for i in range(len(frame_targets_w)) if i not in reached]
        if not remaining:
            return self._delta_action(base, ee_pose)

        if self._has_active_plan():
            target_pose = self._curobo_pose_to_matrix(self._planner.get_next_waypoint_ee_pose())
            return self._delta_action(base, target_pose)

        # Plan exhausted: hold at the current goal until it is latched as reached.
        if self._active_target_idx is not None and self._active_target_idx in remaining:
            return self._delta_action(base, frame_targets_w[self._active_target_idx])

        self._active_target_idx = min(
            remaining,
            key=lambda i: torch.norm(frame_targets_w[i][:3, 3] - ee_pos),
        )
        goal = frame_targets_w[self._active_target_idx]
        if not self._plan_to(base, goal):
            return self._delta_action(base, goal)

        target_pose = self._curobo_pose_to_matrix(self._planner.get_next_waypoint_ee_pose())
        return self._delta_action(base, target_pose)

    def _plan_to(self, base, goal_pose: torch.Tensor) -> bool:
        """Plan a collision-free path to ``goal_pose``; returns whether planning succeeded."""
        import isaaclab.utils.math as PoseUtils
        from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
        from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

        with torch.inference_mode(False), torch.enable_grad():
            if self._planner is None:
                cfg = CuroboPlannerCfg.franka_config()
                self._planner = CuroboPlanner(env=base, robot=base.scene["robot"], config=cfg, env_id=0)

            goal = PoseUtils.make_pose(
                goal_pose[:3, 3].clone(),
                self._ee_pose_w(base)[:3, :3].clone(),
            )
            return self._planner.update_world_and_plan_motion(goal)

    # -- CLI plumbing ----------------------------------------------------------

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument(
            "--ik_approach_step_m",
            type=float,
            default=0.2,
            help="Max EE translation per step (metres).",
        )
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> "IkApproachPolicy":
        return IkApproachPolicy(IkApproachPolicyArgs.from_cli_args(args))
