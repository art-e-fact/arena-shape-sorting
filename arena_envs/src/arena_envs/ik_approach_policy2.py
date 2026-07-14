"""Simplified cuRobo approach policy for pollinate tasks.

A cleaner rewrite of ``ik_approach_policy``. It has one job: drive the Franka
end-effector through every approach frame in ``approach_frame_poses`` (in order),
one after another, and then stop.

Design choices that keep it simple:

- It ignores ``reached_flags`` entirely. Progress is decided purely by whether
  the planned motion has finished and whether the EE is on the goal position.
- For each frame it plans one collision-free cuRobo path, follows every waypoint,
  then settles onto the exact approach pose before advancing to the next frame.
- The commanded step shrinks as the EE nears the current approach frame
  (proportional control), so the arm decelerates into each target.
- Orientation is held by default. With ``--ik_approach2_align`` the EE is also
  rotated to match each approach frame's orientation.

Emits the differential-IK arm action layout ``[dpos(3), drot(3), gripper(1)]``.
Assumes a single environment (env 0) with the Franka base at the world origin.

Run with:
    --policy_type arena_envs.ik_approach_policy2.IkApproachPolicy2
    [--ik_approach2_step_m 0.2] [--ik_approach2_gain 0.5] [--ik_approach2_align]
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
class IkApproachPolicy2Args:
    step_m: float = 0.2
    """Maximum EE translation per step [m]."""
    gain: float = 0.5
    """Proportional gain on the remaining distance; <1 gives a smooth slowdown."""
    align: bool = False
    """If True, also rotate the EE to match each approach frame's orientation."""

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "IkApproachPolicy2Args":
        return cls(
            step_m=args.ik_approach2_step_m,
            gain=args.ik_approach2_gain,
            align=args.ik_approach2_align,
        )


@register_policy
class IkApproachPolicy2(PolicyBase):
    """Drives the Franka EE through every pollinate approach frame, in order."""

    name = "ik_approach2"
    config_class = IkApproachPolicy2Args

    # Slowest commanded step [m]; keeps converging once very close to the goal.
    _MIN_STEP_M = 0.01
    # Position tolerance [m] for considering an approach frame reached.
    _REACH_TOL_M = 0.01
    # Largest orientation change commanded per step [rad] when aligning.
    _MAX_ROT_STEP = 0.2

    def __init__(self, config: IkApproachPolicy2Args):
        super().__init__(config)
        self._planner = None
        self._target_idx = 0
        # Whether we have already issued a plan for the current target frame.
        self._planned = False

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._planner is not None:
            self._planner.reset_plan()
        self._target_idx = 0
        self._planned = False

    # -- main ------------------------------------------------------------------

    def get_action(self, env, observation) -> torch.Tensor:
        base = env.unwrapped
        goals = self._world_goals(base, self._parse_poses(observation["task_obs"]))

        # All approach frames done: hold still.
        if self._target_idx >= len(goals):
            return self._stop(base)

        goal_pos, goal_rot = goals[self._target_idx]

        # 1) Follow the active plan, one waypoint per step.
        if self._planner is not None and self._planner.has_next_waypoint():
            waypoint = self._curobo_pose_to_matrix(self._planner.get_next_waypoint_ee_pose())
            return self._action(base, waypoint[:3, 3], goal_rot, goal_pos)

        # 2) No plan yet for this target: plan once, then start following it.
        if not self._planned:
            self._planned = True
            if self._plan_to(base, goal_pos, goal_rot) and self._planner.has_next_waypoint():
                waypoint = self._curobo_pose_to_matrix(self._planner.get_next_waypoint_ee_pose())
                return self._action(base, waypoint[:3, 3], goal_rot, goal_pos)

        # 3) Plan finished (or planning failed): drive straight onto the pose.
        ee_pos = self._ee_pose_w(base)[:3, 3]
        if torch.norm(goal_pos - ee_pos) < self._REACH_TOL_M:
            self._advance()
            return self._stop(base)
        return self._action(base, goal_pos, goal_rot, goal_pos)

    def _advance(self) -> None:
        """Move on to the next approach frame."""
        self._target_idx += 1
        self._planned = False
        if self._planner is not None:
            self._planner.reset_plan()

    # -- action construction ---------------------------------------------------

    def _action(
        self,
        base,
        target_pos: torch.Tensor,
        target_rot: torch.Tensor,
        goal_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Relative-pose action (1, 7) moving the EE toward ``target_pos``.

        The step magnitude is scaled by the distance to ``goal_pos`` (the current
        approach frame) so the arm slows down as it arrives. ``target_rot`` is only
        used when alignment is enabled; otherwise the current orientation is held.
        """
        import isaaclab.utils.math as PoseUtils

        ee_pose = self._ee_pose_w(base)
        ee_pos = ee_pose[:3, 3]
        ee_rot = ee_pose[:3, :3]

        # Slow down near the goal: cap this step at gain * remaining distance.
        max_step = torch.clamp(
            self.config.gain * torch.norm(goal_pos - ee_pos),
            min=self._MIN_STEP_M,
            max=self.config.step_m,
        )
        dpos = target_pos - ee_pos
        dist = torch.norm(dpos)
        if dist > max_step:
            dpos = dpos * (max_step / dist)

        if self.config.align:
            delta_rot = target_rot @ ee_rot.transpose(-1, -2)
            drot = PoseUtils.axis_angle_from_quat(PoseUtils.quat_from_matrix(delta_rot))
            angle = torch.norm(drot)
            if angle > self._MAX_ROT_STEP:
                drot = drot * (self._MAX_ROT_STEP / angle)
        else:
            drot = torch.zeros(3, device=dpos.device, dtype=dpos.dtype)

        grip = torch.tensor([-1.0], device=dpos.device, dtype=dpos.dtype)
        return torch.cat([dpos, drot, grip]).unsqueeze(0)

    def _stop(self, base) -> torch.Tensor:
        """Zero-motion action (1, 7) that holds the current pose."""
        device = self._ee_pose_w(base).device
        action = torch.zeros(7, device=device)
        action[6] = -1.0
        return action.unsqueeze(0)

    # -- geometry helpers ------------------------------------------------------

    @staticmethod
    def _ee_pose_w(base) -> torch.Tensor:
        """Current end-effector pose in the world frame as a 4x4 matrix (env 0)."""
        import isaaclab.utils.math as PoseUtils

        ee = base.scene["ee_frame"]
        pos = wp.to_torch(ee.data.target_pos_w)[0, 0]
        quat = wp.to_torch(ee.data.target_quat_w)[0, 0]
        return PoseUtils.make_pose(pos, PoseUtils.matrix_from_quat(quat.unsqueeze(0))[0])

    @staticmethod
    def _parse_poses(task_obs) -> torch.Tensor:
        """Return the ``(N, 7)`` approach-frame poses for env 0, as ``(x,y,z,qw,qx,qy,qz)``.

        Poses are expressed in the robot root frame. ``reached_flags`` are ignored.
        """
        if isinstance(task_obs, dict):
            return task_obs["approach_frame_poses"][0].reshape(-1, 7)

        flat = task_obs[0]
        n_frames = flat.shape[-1] // 8
        return flat[: n_frames * 7].reshape(n_frames, 7)

    def _world_goals(self, base, poses_b: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Convert root-frame approach poses to world-frame ``(position, rotation)`` targets."""
        import isaaclab.utils.math as PoseUtils

        robot = base.scene["robot"]
        root_pos_w = wp.to_torch(robot.data.root_pos_w)[0:1]
        root_quat_w = wp.to_torch(robot.data.root_quat_w)[0:1]

        goals: list[tuple[torch.Tensor, torch.Tensor]] = []
        for pose_b in poses_b:
            pos_w, quat_w = combine_frame_transforms(
                root_pos_w,
                root_quat_w,
                pose_b[:3].unsqueeze(0),
                pose_b[3:7].unsqueeze(0),
            )
            goals.append((pos_w[0], PoseUtils.matrix_from_quat(quat_w)[0]))
        return goals

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

    # -- planning --------------------------------------------------------------

    def _plan_to(self, base, goal_pos: torch.Tensor, goal_rot: torch.Tensor) -> bool:
        """Plan a collision-free path to the goal; returns whether planning succeeded."""
        import isaaclab.utils.math as PoseUtils
        from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
        from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

        with torch.inference_mode(False), torch.enable_grad():
            if self._planner is None:
                cfg = CuroboPlannerCfg.franka_config()
                self._planner = CuroboPlanner(env=base, robot=base.scene["robot"], config=cfg, env_id=0)

            rot = goal_rot if self.config.align else self._ee_pose_w(base)[:3, :3]
            goal = PoseUtils.make_pose(goal_pos.clone(), rot.clone())
            return self._planner.update_world_and_plan_motion(goal)

    # -- CLI plumbing ----------------------------------------------------------

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument(
            "--ik_approach2_step_m",
            type=float,
            default=0.2,
            help="Max EE translation per step (metres).",
        )
        parser.add_argument(
            "--ik_approach2_gain",
            type=float,
            default=0.5,
            help="Proportional gain on remaining distance; <1 slows the EE near each frame.",
        )
        parser.add_argument(
            "--ik_approach2_align",
            action="store_true",
            help="Also rotate the EE to match each approach frame's orientation.",
        )
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> "IkApproachPolicy2":
        return IkApproachPolicy2(IkApproachPolicy2Args.from_cli_args(args))
