"""cuRobo motion facade: planner, collision world, attach, joint remapping.

``MotionClient`` hides MotionPlanner / AttachmentManager quirks so the policy
only deals with plan / attach / joint-action APIs. Optional ``CollisionDebugViz``
receives world / joint / attach updates (Viser today; Rerun later).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from arena_so101.mapping import SIM_JOINT_NAMES
from shape_sorting.curobo_viz import CollisionDebugViz, NullCollisionDebugViz
from shape_sorting.curobo_world import (
    DEFAULT_COLLISION_CACHE,
    entity_position_in_robot_base,
    load_world_from_env,
    obstacle_counts,
    obstacle_names_for_entity,
    sync_obstacle_poses_from_env,
)

if TYPE_CHECKING:
    import gymnasium as gym
    from curobo.motion_planner import MotionPlanner
    from curobo.types import JointState, Pose

try:
    import arena_so101 as _arena_so101

    _DEFAULT_ROBOT_YML = (
        Path(_arena_so101.__file__).resolve().parent
        / "embodiments"
        / "data"
        / "curobo"
        / "so101.yml"
    )
except Exception:
    _DEFAULT_ROBOT_YML = Path()

_JAW_INDEX = SIM_JOINT_NAMES.index("Jaw")

# so101.yml extra_collision_spheres.attached_object
_ATTACH_NUM_SPHERES = 16
_ATTACH_SURFACE_RADIUS = 0.006


@dataclass
class MotionClientCfg:
    """Planner + attach settings used by ``MotionClient``."""

    robot_yml: str = ""
    position_tolerance: float = 0.015
    orientation_tolerance: float = 0.1
    use_cuda_graph: bool = False
    self_collision_check: bool = False
    waypoint_stride: int = 2
    attach_num_spheres: int = _ATTACH_NUM_SPHERES
    attach_surface_radius: float = _ATTACH_SURFACE_RADIUS


@dataclass
class PlanResult:
    """Output of ``MotionClient.plan_to_pose``."""

    waypoints: torch.Tensor  # (T, n_planner_joints) in planner joint order
    success: bool
    hold_action: torch.Tensor  # (n_sim_joints,) SIM_JOINT_NAMES order fallback


@dataclass
class AttachResult:
    """Output of ``MotionClient.attach``."""

    grasp_offset_ee: torch.Tensor  # (3,) object origin in tool frame
    spheres_ee: torch.Tensor  # (N, 4) attached spheres in tool frame


def _format_plan_failure(result) -> str:
    if result is None:
        return "IK found no seeds (result=None); holding current joint pose."

    parts = [f"success={result.success.tolist()}"]
    if result.feasible is not None:
        parts.append(f"feasible={result.feasible.tolist()}")
    if result.position_error is not None:
        parts.append(f"pos_err={result.position_error.detach().flatten().tolist()}")
    if result.rotation_error is not None:
        parts.append(f"rot_err={result.rotation_error.detach().flatten().tolist()}")
    if result.metrics is not None and getattr(result.metrics, "constraint", None) is not None:
        parts.append(f"constraint={result.metrics.constraint.detach().flatten().tolist()}")
    parts.append("holding current joint pose.")
    return "; ".join(parts)


def resolve_robot_yml(robot_yml: str = "") -> Path:
    path = (
        Path(robot_yml).expanduser().resolve()
        if robot_yml
        else Path(_DEFAULT_ROBOT_YML).resolve()
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"cuRobo robot YAML not found: {path}. "
            "Generate it with: python -m arena_so101.generate_curobo_config --skip-usd-convert"
        )
    return path


class MotionClient:
    """Facade over cuRobo MotionPlanner + collision world + object attach."""

    def __init__(
        self,
        cfg: MotionClientCfg,
        *,
        debug: CollisionDebugViz | None = None,
    ):
        self.cfg = cfg
        self.debug: CollisionDebugViz = debug or NullCollisionDebugViz()
        self._planner: MotionPlanner | None = None
        self._world_loaded = False
        self._sim_joint_indices: dict[str, int] | None = None
        # Last planned EE goal in robot base (xyzw quat). Used by Kit markers.
        self.last_goal_xyz: torch.Tensor | None = None
        self.last_goal_quat_xyzw: torch.Tensor | None = None
        # Attach state (cleared on detach).
        self.grasp_offset_ee: torch.Tensor | None = None
        self._spheres_ee: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def ensure_ready(self, device: torch.device) -> None:
        if self._planner is not None:
            return
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo.types import DeviceCfg

        if device.type != "cuda":
            raise RuntimeError(f"cuRobo requires a CUDA device, got {device}")

        yml = resolve_robot_yml(self.cfg.robot_yml)
        print(f"[MotionClient] Loading robot config: {yml}")
        with torch.inference_mode(False):
            cfg = MotionPlannerCfg.create(
                robot=str(yml),
                scene_model=None,
                collision_cache=dict(DEFAULT_COLLISION_CACHE),
                self_collision_check=self.cfg.self_collision_check,
                use_cuda_graph=self.cfg.use_cuda_graph,
                position_tolerance=self.cfg.position_tolerance,
                orientation_tolerance=self.cfg.orientation_tolerance,
                device_cfg=DeviceCfg(device=device, dtype=torch.float32),
            )
            self._planner = MotionPlanner(cfg)
        self._world_loaded = False
        print(
            f"[MotionClient] Planner ready. joints={self._planner.joint_names} "
            f"tool_frames={self._planner.tool_frames}"
        )

    def close(self) -> None:
        self.detach()
        self._planner = None
        self._world_loaded = False
        self._sim_joint_indices = None
        self.last_goal_xyz = None
        self.last_goal_quat_xyzw = None
        self.debug.close()

    @property
    def planner(self) -> MotionPlanner:
        assert self._planner is not None, "Call ensure_ready() first"
        return self._planner

    @property
    def joint_names(self) -> list[str]:
        return list(self.planner.joint_names)

    @property
    def tool_frame(self) -> str:
        frames = self.planner.tool_frames
        return str(frames[0]) if frames else "gripper"

    # ------------------------------------------------------------------
    # Collision world
    # ------------------------------------------------------------------

    def sync_world(self, env: gym.Env) -> None:
        """Load USD geometry once, then refresh dynamic obstacle poses."""
        planner = self.planner
        am = planner.attachment_manager
        # update_world / pose sync can clear enable flags — remember disabled set.
        disabled = list(am._disabled_obstacle_names)

        if not self._world_loaded:
            scene = load_world_from_env(planner, env, env_id=0)
            self._world_loaded = True
            print(
                f"[MotionClient] Collision world loaded: {obstacle_counts(scene)} "
                f"({len(scene)} obstacles)"
            )

        scene = sync_obstacle_poses_from_env(planner, env, env_id=0)

        for name in disabled:
            if planner.scene_collision_checker.check_obstacle_exists(name, env_idx=0):
                planner.scene_collision_checker.enable_obstacle(name, enable=False, env_idx=0)
        if disabled:
            am._disabled_obstacle_names = disabled
            am._disabled_num_envs = 1
            print(f"[MotionClient] Pose sync; re-disabled={disabled}")

        self.debug.set_world(scene)

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def plan_to_pose(
        self,
        env: gym.Env,
        device: torch.device,
        goal_xyz: torch.Tensor,
        goal_quat_xyzw: torch.Tensor,
        *,
        label: str = "plan",
    ) -> PlanResult:
        """Plan EE motion to ``goal`` (robot base). Syncs collision world first."""
        from curobo.types import GoalToolPose
        from isaaclab.utils.math import convert_quat

        planner = self.planner
        self.last_goal_xyz = goal_xyz.detach()
        self.last_goal_quat_xyzw = goal_quat_xyzw.detach()

        # policy_runner wraps get_action in inference_mode(); cuRobo stores goal
        # tensors by reference then copy_'s into them — clone out of that mode.
        with torch.inference_mode(False):
            self.sync_world(env)
            goal_xyz = goal_xyz.detach().clone()
            goal_quat_xyzw = goal_quat_xyzw.detach().clone()
            goal_quat_wxyz = convert_quat(goal_quat_xyzw, to="wxyz").clone()
            q_start = self.planner_joint_state(env, device)

            self.debug.set_goal(
                goal_xyz.detach().cpu().tolist(),
                goal_quat_wxyz.detach().cpu().tolist(),
            )
            self._push_debug_joints(q_start)

            goal = GoalToolPose(
                tool_frames=list(planner.tool_frames),
                position=goal_xyz.view(1, 1, 1, 1, 3).clone(),
                quaternion=goal_quat_wxyz.view(1, 1, 1, 1, 4).clone(),
            )

            print(
                f"[MotionClient] {label} planning EE "
                f"pos=({goal_xyz[0]:.3f}, {goal_xyz[1]:.3f}, {goal_xyz[2]:.3f})…"
            )
            result = planner.plan_pose(goal, q_start, max_attempts=10)

            hold = self.joint_action(env, device)
            if result is None or not bool(result.success.any()):
                print(f"[MotionClient] {label} failed — {_format_plan_failure(result)}")
                # Single-waypoint "traj" that holds current pose.
                name_to_idx = self._sim_name_to_index(env)
                row = torch.stack([hold[name_to_idx[n]] for n in planner.joint_names])
                return PlanResult(
                    waypoints=row.unsqueeze(0).detach(),
                    success=False,
                    hold_action=hold.detach(),
                )

            positions = result.get_interpolated_plan().position
            while positions.ndim > 2:
                positions = positions[0]
            stride = max(1, int(self.cfg.waypoint_stride))
            waypoints = positions[::stride].detach().contiguous()
            print(f"[MotionClient] {label} plan OK — {waypoints.shape[0]} waypoints.")
            return PlanResult(waypoints=waypoints, success=True, hold_action=hold.detach())

    def clear_last_goal(self) -> None:
        self.last_goal_xyz = None
        self.last_goal_quat_xyzw = None

    # ------------------------------------------------------------------
    # Attach / detach
    # ------------------------------------------------------------------

    def attach(self, env: gym.Env, device: torch.device, entity_name: str) -> AttachResult:
        """Fit collision spheres on ``entity_name`` and attach them to the tool."""
        from curobo.sphere_fit import SphereFitType

        planner = self.planner
        names = self._obstacle_names_matching(entity_name)
        if not names:
            all_names = planner.scene_collision_checker.get_obstacle_names(0)
            raise RuntimeError(
                f"No cuRobo obstacles match '{entity_name}'. Have: {all_names}"
            )

        scene_model = planner.scene_collision_checker.scene_model
        if isinstance(scene_model, list):
            scene_model = scene_model[0]
        obstacles = []
        for name in names:
            obs = scene_model.get_obstacle(name)
            if obs is None:
                raise RuntimeError(f"Obstacle '{name}' missing from scene_model.")
            obstacles.append(obs)

        # Work around AttachmentManager.update bug: transform_points(...).squeeze(0)
        # collapses to 1D when only one sphere is fitted. Fit in world (robot base),
        # map spheres into the EE frame ourselves, then update with no pose offset.
        print(f"[MotionClient] ATTACH obstacles={names}")
        with torch.inference_mode(False):
            self.sync_world(env)
            am = planner.attachment_manager
            q = self.planner_joint_state(env, device)
            spheres_w = am.fit_spheres(
                obstacles,
                num_spheres=int(self.cfg.attach_num_spheres),
                surface_radius=float(self.cfg.attach_surface_radius),
                sphere_fit_type=SphereFitType.VOXEL,
            )

            ee = self.ee_pose(q)
            centers_ee = ee.inverse().transform_points(spheres_w[:, :3].contiguous())
            centers_ee = centers_ee.reshape(-1, 3)
            spheres_ee = torch.cat([centers_ee, spheres_w[:, 3:4]], dim=-1)

            am.update(spheres_ee, q, link_name="attached_object", world_objects_pose_offset=None)
            for name in names:
                planner.scene_collision_checker.enable_obstacle(name, enable=False, env_idx=0)
            am._disabled_obstacle_names = list(names)
            am._disabled_num_envs = 1
            self._spheres_ee = spheres_ee.detach().clone()
            print(f"[MotionClient] ATTACH fitted {spheres_ee.shape[0]} spheres.")

        grasp_offset = self._grasp_offset_from_ee(env, device, entity_name, ee_pose=ee)
        self.grasp_offset_ee = grasp_offset
        self._push_debug_joints(q)
        return AttachResult(grasp_offset_ee=grasp_offset, spheres_ee=self._spheres_ee)

    def detach(self) -> None:
        """Clear attached spheres and re-enable any disabled world obstacles."""
        self.grasp_offset_ee = None
        self._spheres_ee = None
        self.debug.clear_attached_spheres()
        if self._planner is None:
            return
        am = self._planner.attachment_manager
        if am._attached_link_name is None and not am._disabled_obstacle_names:
            return
        with torch.inference_mode(False):
            print(
                f"[MotionClient] DETACH link={am._attached_link_name} "
                f"re-enable={list(am._disabled_obstacle_names)}"
            )
            am.detach()

    # ------------------------------------------------------------------
    # Joints / actions (sim ↔ planner)
    # ------------------------------------------------------------------

    def joint_action(self, env: gym.Env, device: torch.device) -> torch.Tensor:
        """Current sim joints in ``SIM_JOINT_NAMES`` order. Shape ``(6,)``."""
        robot = env.unwrapped.scene["robot"]
        q = robot.data.joint_pos[0].detach().to(device=device, dtype=torch.float32)
        name_to_idx = self._sim_name_to_index(env)
        action = torch.zeros(len(SIM_JOINT_NAMES), device=device, dtype=torch.float32)
        for i, name in enumerate(SIM_JOINT_NAMES):
            action[i] = q[name_to_idx[name]]
        return action

    def measured_jaw(self, env: gym.Env, device: torch.device) -> torch.Tensor:
        return self.joint_action(env, device)[_JAW_INDEX]

    def planner_joint_state(self, env: gym.Env, device: torch.device) -> JointState:
        from curobo.types import JointState

        robot = env.unwrapped.scene["robot"]
        q = robot.data.joint_pos[0].detach().to(device=device, dtype=torch.float32)
        name_to_idx = self._sim_name_to_index(env)
        active = [q[name_to_idx[name]] for name in self.planner.joint_names]
        # clone: JointState may be stored in cuRobo goal buffers and mutated inplace.
        q_active = torch.stack(active).unsqueeze(0).contiguous().clone()
        return JointState.from_position(q_active, joint_names=list(self.planner.joint_names))

    def q_to_action(
        self, q_planner: torch.Tensor, device: torch.device, *, jaw: float
    ) -> torch.Tensor:
        """Map a planner waypoint (+ jaw command) to a sim action vector."""
        action = torch.zeros(len(SIM_JOINT_NAMES), device=device, dtype=torch.float32)
        sim_index = {n: i for i, n in enumerate(SIM_JOINT_NAMES)}
        q = q_planner.detach()
        # Interpolated plans may carry locked joints (e.g. Jaw); keep active DoF only.
        n = len(self.planner.joint_names)
        if q.shape[-1] > n:
            q = q[..., :n]
        for i, name in enumerate(self.planner.joint_names):
            action[sim_index[name]] = q[i]
        action[_JAW_INDEX] = float(jaw)
        return action

    def notify_joint(self, q_planner: torch.Tensor | JointState) -> None:
        """Push joint state (and attached spheres) to the debug sink during playback."""
        self._push_debug_joints(q_planner)

    def ee_pose(self, joint_state: JointState) -> Pose:
        return self.planner.compute_kinematics(joint_state).tool_poses.get_link_pose(
            self.planner.tool_frames[0]
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _sim_name_to_index(self, env: gym.Env) -> dict[str, int]:
        if self._sim_joint_indices is not None:
            return self._sim_joint_indices
        robot = env.unwrapped.scene["robot"]
        names = list(robot.data.joint_names)
        self._sim_joint_indices = {n: i for i, n in enumerate(names)}
        missing = [n for n in SIM_JOINT_NAMES if n not in self._sim_joint_indices]
        if missing:
            raise RuntimeError(
                f"Sim robot is missing expected SO-101 joints: {missing}. Have: {names}"
            )
        return self._sim_joint_indices

    def _obstacle_names_matching(self, entity_name: str) -> list[str]:
        names = self.planner.scene_collision_checker.get_obstacle_names(0)
        return obstacle_names_for_entity(names, entity_name)

    def _grasp_offset_from_ee(
        self,
        env: gym.Env,
        device: torch.device,
        entity_name: str,
        *,
        ee_pose: Pose,
    ) -> torch.Tensor:
        obj_b = entity_position_in_robot_base(env, entity_name, device=device)
        offset = ee_pose.inverse().transform_points(obj_b.view(1, 3).contiguous())
        offset = offset.reshape(3).detach().clone()
        print(
            f"[MotionClient] grasp offset (EE)=("
            f"{float(offset[0]):.4f}, {float(offset[1]):.4f}, {float(offset[2]):.4f})"
        )
        return offset

    def _as_planner_joint_state(self, joint_state_or_q) -> JointState:
        from curobo.types import JointState

        n_active = len(self.planner.joint_names)
        if isinstance(joint_state_or_q, JointState):
            name_to_i = {n: i for i, n in enumerate(joint_state_or_q.joint_names)}
            q = torch.stack(
                [joint_state_or_q.position[..., name_to_i[n]] for n in self.planner.joint_names],
                dim=-1,
            )
        else:
            q = joint_state_or_q.detach()
            if q.shape[-1] > n_active:
                q = q[..., :n_active]
            elif q.shape[-1] != n_active:
                raise RuntimeError(
                    f"Expected {n_active} planner DoF, got {q.shape[-1]}"
                )
        if q.ndim == 1:
            q = q.unsqueeze(0)
        return JointState.from_position(
            q.contiguous().clone(),
            joint_names=list(self.planner.joint_names),
        )

    def _spheres_world(self, joint_state: JointState) -> torch.Tensor | None:
        if self._spheres_ee is None:
            return None
        ee = self.ee_pose(joint_state)
        centers_w = ee.transform_points(self._spheres_ee[:, :3].contiguous()).reshape(-1, 3)
        return torch.cat([centers_w, self._spheres_ee[:, 3:4]], dim=-1)

    def _push_debug_joints(self, joint_state_or_q) -> None:
        js = self._as_planner_joint_state(joint_state_or_q)
        spheres_w = self._spheres_world(js)
        if spheres_w is not None:
            self.debug.set_attached_spheres(spheres_w)
        self.debug.set_robot_state(js)
