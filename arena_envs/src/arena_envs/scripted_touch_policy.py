"""Scripted demo policy for the ``touch_sphere`` task.

A minimal, non-learned policy that drives the Franka's end-effector to the touch
target so you can *see* the task being solved. It plugs into the standard policy
runner via ``@register_policy`` (like ``ZeroActionPolicy``), so no change to the
environment is needed.

Two modes:
  * straight line (default): closed-loop, step the EE straight at the sphere each frame.
  * cuRobo: plan a collision-free EE path once, then follow its waypoints.

Both emit the differential-IK arm action layout ``[dpos(3), drot(3), gripper(1)]``
(relative pose delta + binary gripper). Assumes a single environment (env 0) with
the Franka base at the world origin, matching the isaaclab_mimic cuRobo examples.

Run with:
    --policy_type arena_envs.scripted_touch_policy.ScriptedTouchPolicy
    [--scripted_use_curobo] [--scripted_step_m 0.05]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase


@dataclass
class ScriptedTouchPolicyArgs:
    step_m: float = 0.05  # max EE translation commanded per step (metres)
    use_curobo: bool = False  # plan a collision-free path with cuRobo instead of a straight line

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "ScriptedTouchPolicyArgs":
        return cls(step_m=args.scripted_step_m, use_curobo=args.scripted_use_curobo)


@register_policy
class ScriptedTouchPolicy(PolicyBase):
    """Drives the Franka EE to the ``touch_sphere`` target with a scripted trajectory."""

    name = "scripted_touch"
    config_class = ScriptedTouchPolicyArgs

    def __init__(self, config: ScriptedTouchPolicyArgs):
        super().__init__(config)
        self._planner = None  # lazily created CuroboPlanner (cuRobo mode only)
        self._waypoints: list[torch.Tensor] | None = None  # cached 4x4 EE target poses
        self._wp_index = 0

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        # Drop the cached plan so the next episode re-plans from the new state.
        self._waypoints = None
        self._wp_index = 0

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def _ee_pose_w(base) -> torch.Tensor:
        """Current end-effector pose in the world frame as a 4x4 matrix (env 0)."""
        import warp as wp

        import isaaclab.utils.math as PoseUtils

        ee = base.scene["ee_frame"]
        pos = wp.to_torch(ee.data.target_pos_w)[0, 0]  # (3,)
        quat = wp.to_torch(ee.data.target_quat_w)[0, 0]  # (4,) wxyz
        return PoseUtils.make_pose(pos, PoseUtils.matrix_from_quat(quat.unsqueeze(0))[0])

    def _delta_action(self, base, target_pose: torch.Tensor, gripper: float = 1.0) -> torch.Tensor:
        """Turn a 4x4 world target pose into a clamped relative-pose action (1, 7).

        This mirrors the mimic ``target_eef_pose_to_action``: a delta position plus an
        axis-angle delta rotation from the current EE pose to ``target_pose``.
        """
        import isaaclab.utils.math as PoseUtils

        curr = self._ee_pose_w(base)
        dpos = torch.clamp(target_pose[:3, 3] - curr[:3, 3], -self.config.step_m, self.config.step_m)
        delta_rot = target_pose[:3, :3] @ curr[:3, :3].transpose(-1, -2)
        drot = PoseUtils.axis_angle_from_quat(PoseUtils.quat_from_matrix(delta_rot))
        grip = torch.tensor([gripper], device=dpos.device, dtype=dpos.dtype)
        return torch.cat([dpos, drot, grip]).unsqueeze(0)  # (1, 7)

    # -- main ------------------------------------------------------------------

    def get_action(self, env, observation) -> torch.Tensor:
        import warp as wp

        import isaaclab.utils.math as PoseUtils

        base = env.unwrapped
        sphere_pos = wp.to_torch(base.scene["touch_sphere"].data.root_pos_w)[0]  # (3,) world

        if not self.config.use_curobo:
            # Straight-line closed-loop reach: aim the EE at the sphere centre, keep orientation.
            target = PoseUtils.make_pose(sphere_pos, self._ee_pose_w(base)[:3, :3])
            return self._delta_action(base, target)

        # cuRobo mode: plan a collision-free path to the sphere once, then follow it.
        if self._waypoints is None:
            self._waypoints = self._plan_to(base, sphere_pos)

        # One waypoint per step; hold the final pose once exhausted so the EE settles into contact.
        target = self._waypoints[min(self._wp_index, len(self._waypoints) - 1)]
        self._wp_index += 1
        return self._delta_action(base, target)

    def _plan_to(self, base, sphere_pos: torch.Tensor) -> list[torch.Tensor]:
        """Build the planner on first use and return a list of 4x4 EE target poses."""
        import isaaclab.utils.math as PoseUtils
        from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
        from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

        # cuRobo's optimizer runs autograd (cost.backward()), which is forbidden under the
        # torch.inference_mode() the policy runner wraps get_action in. Re-enable grad and
        # clone the goal into a normal (non-inference) tensor so cuRobo can use it.
        with torch.inference_mode(False), torch.enable_grad():
            if self._planner is None:
                cfg = CuroboPlannerCfg.franka_config()
                # Don't treat the touch target itself as an obstacle, or the planner would
                # avoid the very thing we want to touch.
                cfg.world_ignore_substrings = list(cfg.world_ignore_substrings) + ["TouchSphere"]
                self._planner = CuroboPlanner(env=base, robot=base.scene["robot"], config=cfg, env_id=0)

            # Reach the sphere while keeping the current EE orientation (orientation is irrelevant here).
            goal = PoseUtils.make_pose(sphere_pos.clone(), self._ee_pose_w(base)[:3, :3].clone())
            if not self._planner.update_world_and_plan_motion(goal):
                # Planning failed: fall back to a single straight-line target.
                return [goal.detach()]
            return [p.detach() for p in self._planner.get_planned_poses()]

    # -- CLI plumbing ----------------------------------------------------------

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument(
            "--scripted_step_m", type=float, default=0.05, help="Max EE translation per step (metres)."
        )
        parser.add_argument(
            "--scripted_use_curobo",
            action="store_true",
            help="Plan a collision-free path with cuRobo instead of a straight line.",
        )
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> "ScriptedTouchPolicy":
        return ScriptedTouchPolicy(ScriptedTouchPolicyArgs.from_cli_args(args))
