"""Scripted demo policy for the ``touch_sphere`` task.

A minimal, non-learned policy that drives the Franka's end-effector to each touch
target in turn so you can *see* the task being solved. It always heads for the
nearest not-yet-touched sphere (order doesn't matter). It plugs into the standard
policy runner via ``@register_policy`` (like ``ZeroActionPolicy``), so no change to
the environment is needed.

Two modes:
  * straight line (default): closed-loop, step the EE straight at the nearest sphere each frame.
  * cuRobo: plan a collision-free EE path to one sphere at a time, then follow its waypoints.

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
    """Drives the Franka EE to each ``touch_sphere_<i>`` target with a scripted trajectory."""

    name = "scripted_touch"
    config_class = ScriptedTouchPolicyArgs
    reach_tol_m = 0.06  # EE-to-centre distance under which a sphere counts as touched

    def __init__(self, config: ScriptedTouchPolicyArgs):
        super().__init__(config)
        self._planner = None  # lazily created CuroboPlanner (cuRobo mode only)
        self._names: list[str] | None = None  # cached touch-target entity names
        self._touched: set[int] = set()  # indices of spheres already reached this episode
        self._target_idx: int | None = None  # sphere currently being planned to (cuRobo mode)
        self._waypoints: list[torch.Tensor] | None = None  # cached 4x4 EE target poses
        self._wp_index = 0

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        # Drop episode progress and any cached plan so the next episode starts fresh.
        # ``_names`` is kept: the scene entities are stable across episodes.
        self._touched = set()
        self._target_idx = None
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

    def _sphere_names(self, base) -> list[str]:
        """Discover and cache the touch-target entity names (``touch_sphere_<i>``)."""
        import re

        if self._names is None:
            names = [k for k in base.scene.rigid_objects.keys() if re.fullmatch(r"touch_sphere_\d+", k)]
            self._names = sorted(names, key=lambda n: int(n.rsplit("_", 1)[1]))
        return self._names

    def _sphere_pos(self, base, name: str) -> torch.Tensor:
        """World-frame position (3,) of sphere ``name`` in env 0."""
        import warp as wp

        return wp.to_torch(base.scene[name].data.root_pos_w)[0]

    # -- main ------------------------------------------------------------------

    def get_action(self, env, observation) -> torch.Tensor:
        import isaaclab.utils.math as PoseUtils

        base = env.unwrapped
        names = self._sphere_names(base)
        ee_pose = self._ee_pose_w(base)
        ee_rot, ee_pos = ee_pose[:3, :3], ee_pose[:3, 3]

        # Latch every sphere the EE is currently close enough to have touched.
        for i, name in enumerate(names):
            if torch.norm(self._sphere_pos(base, name) - ee_pos) < self.reach_tol_m:
                self._touched.add(i)

        remaining = [i for i in range(len(names)) if i not in self._touched]
        if not remaining:
            # All spheres touched: hold the current pose (near-zero delta).
            return self._delta_action(base, ee_pose)

        if not self.config.use_curobo:
            # Straight-line closed-loop reach toward the ne arest not-yet-touched sphere.
            target_idx = min(remaining, key=lambda i: torch.norm(self._sphere_pos(base, names[i]) - ee_pos))
            target = PoseUtils.make_pose(self._sphere_pos(base, names[target_idx]), ee_rot)
            return self._delta_action(base, target)

        # cuRobo mode: plan to one sphere at a time, replanning when the target changes.
        if self._target_idx is None or self._target_idx in self._touched:
            self._target_idx = min(
                remaining, key=lambda i: torch.norm(self._sphere_pos(base, names[i]) - ee_pos)
            )
            self._waypoints = self._plan_to(base, names[self._target_idx], self._sphere_pos(base, names[self._target_idx]))
            self._wp_index = 0

        # One waypoint per step; hold the final pose once exhausted so the EE settles into contact.
        target = self._waypoints[min(self._wp_index, len(self._waypoints) - 1)]
        self._wp_index += 1
        return self._delta_action(base, target)

    def _plan_to(self, base, target_name: str, sphere_pos: torch.Tensor) -> list[torch.Tensor]:
        """Build the planner on first use and return a list of 4x4 EE target poses.

        The sphere we're currently reaching for is temporarily removed from the collision
        world before planning: its own geometry would otherwise make the goal pose (the
        sphere centre) collide, and cuRobo's collision-aware IK returns ``IK_FAIL``. The
        other spheres stay active, so the planner still avoids them.
        """
        import isaaclab.utils.math as PoseUtils
        from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner
        from isaaclab_mimic.motion_planners.curobo.curobo_planner_cfg import CuroboPlannerCfg

        # cuRobo's optimizer runs autograd (cost.backward()), which is forbidden under the
        # torch.inference_mode() the policy runner wraps get_action in. Re-enable grad and
        # clone the goal into a normal (non-inference) tensor so cuRobo can use it.
        with torch.inference_mode(False), torch.enable_grad():
            if self._planner is None:
                cfg = CuroboPlannerCfg.franka_config()
                cfg.visualize_spheres=True
                cfg.visualize_plan=True
                cfg.debug_planner=True
                self._planner = CuroboPlanner(env=base, robot=base.scene["robot"], config=cfg, env_id=0)

            # Disable only the target sphere's obstacle so the goal is reachable; the rest
            # remain active for collision avoidance. Always restore it afterwards.
            target_path = self._planner._get_object_mappings().get(target_name)
            coll_checker = self._planner.motion_gen.world_coll_checker
            if target_path is not None:
                coll_checker.enable_obstacle(target_path, enable=False)
            try:
                # Reach the sphere while keeping the current EE orientation (orientation is irrelevant here).
                goal = PoseUtils.make_pose(sphere_pos.clone(), self._ee_pose_w(base)[:3, :3].clone())
                if not self._planner.update_world_and_plan_motion(goal):
                    # Planning failed: fall back to a single straight-line target.
                    return [goal.detach()]
                return [p.detach() for p in self._planner.get_planned_poses()]
            finally:
                if target_path is not None:
                    coll_checker.enable_obstacle(target_path, enable=True)

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
