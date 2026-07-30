"""cuRobo pick-and-place smoke-test policy for SO-101.

Sequence: APPROACH → CLOSE → ATTACH → PLACE → OPEN → DONE.
Plans with ``plan_pose`` (straight to pose; no ``plan_grasp``). Collision world
comes from the live USD stage (see ``curobo_world``). Requires
``--embodiment so101_abs_joint``.

Example::

    python submodules/IsaacLab-Arena/isaaclab_arena/evaluation/policy_runner.py \\
      --viz kit \\
      --policy_type shape_sorting.curobo_policy.CuroboPolicy \\
      --num_steps 400 \\
      --external_environment_class_path shape_sorting.shape_sorting_env:ShapeSortingEnvironment \\
      shape_sorting_test \\
      --embodiment so101_abs_joint \\
      --goal_object shape_piece_cube
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import gymnasium as gym
import torch
from gymnasium.spaces.dict import Dict as GymSpacesDict

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

from arena_so101.mapping import SIM_JOINT_NAMES
from shape_sorting.curobo_world import DEFAULT_COLLISION_CACHE, load_world_from_env, obstacle_counts

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

# Grasp pose relative to goal_object (robot base frame).
_GOAL_XY_STANDOFF_M = 0.03
_GOAL_Z_M = 0.14
_GOAL_TILT_RAD = 0.0
_GOAL_ROLL_RAD = 0.0

# Workshop USD Jaw limits (degrees → radians), same as arena_so101.
_JAW_OPEN_RAD = math.radians(100.0)
_JAW_CLOSE_RAD = math.radians(-10.0)
_JAW_INDEX = SIM_JOINT_NAMES.index("Jaw")

# so101.yml extra_collision_spheres.attached_object
_ATTACH_NUM_SPHERES = 4


class Phase(Enum):
    APPROACH = auto()
    CLOSE = auto()
    ATTACH = auto()
    PLACE = auto()
    OPEN = auto()
    DONE = auto()


def _format_plan_failure(result) -> str:
    """Human-readable plan failure summary from a TrajOpt / MotionPlanner result."""
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


def so101_ee_pose_xyzw(
    x: float,
    y: float,
    z: float,
    tilt: float = 0.0,
    roll: float = 0.0,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a reachable SO-101 EE pose from 5-DoF task parameters.

    Position ``(x, y, z)`` is in the robot base frame. At ``tilt = roll = 0`` the
    tool frame is FLU in the arm plane: X forward, Y left, Z up.
    """
    from isaaclab.utils.math import quat_from_matrix

    pos = torch.tensor([x, y, z], device=device, dtype=dtype)
    rho = math.hypot(x, y)
    psi = 0.0 if rho < 1e-8 else math.atan2(y, x)

    x0 = torch.tensor([math.cos(psi), math.sin(psi), 0.0], device=device, dtype=dtype)
    z0 = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    y0 = torch.linalg.cross(z0, x0)
    y0 = y0 / y0.norm().clamp_min(1e-8)
    rot0 = torch.stack([x0, y0, z0], dim=1)

    ct, st = math.cos(tilt), math.sin(tilt)
    cr, sr = math.cos(roll), math.sin(roll)
    rot_y = torch.tensor(
        [[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]],
        device=device,
        dtype=dtype,
    )
    rot_z = torch.tensor(
        [[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    rot = rot0 @ rot_y @ rot_z
    return pos, quat_from_matrix(rot.unsqueeze(0))[0].to(dtype=dtype)


@dataclass
class CuroboPolicyCfg(PolicyCfg):
    """Configure a cuRobo pick-and-place sequence for SO-101 abs-joint control."""

    robot_yml: str = ""
    """Path to cuRobo ``so101.yml``. Empty uses the generated package default."""

    goal_object: str = "shape_piece_cube"
    """Scene entity to grasp (required)."""

    place_object: str = "sorting_box"
    """Scene entity used as the place reference (XY from its origin)."""

    place_z_offset_m: float = 0.14
    """EE height above ``place_object`` origin in the robot base frame [m]."""

    position_tolerance: float = 0.015
    """cuRobo position convergence tolerance [m]."""

    orientation_tolerance: float = 0.1
    """Orientation tolerance [rad]."""

    jaw_open: float = _JAW_OPEN_RAD
    """Jaw command during APPROACH / OPEN [rad]."""

    jaw_closed: float = _JAW_CLOSE_RAD
    """Jaw command during CLOSE / ATTACH / PLACE [rad]."""

    close_steps: int = 20
    """Sim steps to hold the closed jaw before ATTACH."""

    open_steps: int = 20
    """Sim steps to hold the open jaw before DONE."""

    use_cuda_graph: bool = False
    """Enable cuRobo CUDA graphs."""

    self_collision_check: bool = False
    """Enable self-collision costs."""

    waypoint_stride: int = 2
    """Play every N-th interpolated waypoint (1 = all)."""

    debug_viz: bool = True
    """Draw goal + EE frame markers in the Kit viewport."""

    debug_viser: bool = False
    """Open a Viser page with collision meshes + robot spheres."""

    debug_viser_port: int = 8080
    """Viser HTTP port when ``debug_viser`` is enabled."""

    marker_frame_scale: float = 0.08
    """World-scale of the frame marker axes [m]."""


@register_policy
class CuroboPolicy(PolicyBase[CuroboPolicyCfg]):
    """APPROACH → CLOSE → ATTACH → PLACE → OPEN pick-and-place smoke test."""

    name = "curobo_reach"

    def __init__(self, config: CuroboPolicyCfg):
        super().__init__(config)
        self._planner = None
        self._phase = Phase.APPROACH
        self._traj: torch.Tensor | None = None
        self._step_idx = 0
        self._phase_steps = 0
        self._hold_action: torch.Tensor | None = None
        self._active_goal_b: torch.Tensor | None = None  # (3,) robot base
        self._active_goal_quat_xyzw: torch.Tensor | None = None
        self._sim_joint_indices: dict[str, int] | None = None
        self._world_loaded = False
        self._collision_scene = None
        self._viser = None
        self._goal_marker = None
        self._ee_marker = None
        self._ee_body_id: int | None = None

    # ------------------------------------------------------------------
    # PolicyBase
    # ------------------------------------------------------------------

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self._phase = Phase.APPROACH
        self._traj = None
        self._step_idx = 0
        self._phase_steps = 0
        self._hold_action = None
        self._active_goal_b = None
        self._active_goal_quat_xyzw = None

    def close(self) -> None:
        self._planner = None
        self._traj = None
        self._world_loaded = False
        self._collision_scene = None
        self._viser = None
        self._goal_marker = None
        self._ee_marker = None
        self._ee_body_id = None

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        device = torch.device(env.unwrapped.device)
        num_envs = env.unwrapped.num_envs
        action_dim = env.action_space.shape[-1]
        if action_dim != len(SIM_JOINT_NAMES):
            raise RuntimeError(
                f"CuroboPolicy expects so101_abs_joint (action dim {len(SIM_JOINT_NAMES)}), "
                f"got action dim {action_dim}. Pass `--embodiment so101_abs_joint`."
            )

        self._ensure_planner(device)

        if self._phase is Phase.APPROACH:
            action = self._step_approach(env, device)
        elif self._phase is Phase.CLOSE:
            action = self._step_close(device)
        elif self._phase is Phase.ATTACH:
            action = self._step_attach(env, device)
        elif self._phase is Phase.PLACE:
            action = self._step_place(env, device)
        elif self._phase is Phase.OPEN:
            action = self._step_open(device)
        else:  # DONE
            action = self._hold_action
            if action is None:
                action = self._current_joint_action(env, device)

        self._update_debug_viz(env, device)
        return action.unsqueeze(0).expand(num_envs, -1).contiguous()

    # ------------------------------------------------------------------
    # APPROACH — plan to grasp pose, jaw open
    # ------------------------------------------------------------------

    def _step_approach(self, env: gym.Env, device: torch.device) -> torch.Tensor:
        if self._traj is None:
            goal_xyz, goal_quat = self._grasp_pose_in_robot_base(env, device)
            self._plan_to_pose(env, device, goal_xyz, goal_quat, label="APPROACH")

        action = self._playback_traj(device, jaw=self.config.jaw_open)
        if action is not None:
            return action

        print(f"[CuroboPolicy] APPROACH done → CLOSE ({self.config.close_steps} steps).")
        self._phase = Phase.CLOSE
        self._phase_steps = 0
        return self._step_close(device)

    # ------------------------------------------------------------------
    # CLOSE — hold arm, close jaw
    # ------------------------------------------------------------------

    def _step_close(self, device: torch.device) -> torch.Tensor:
        action = self._hold_with_jaw(device, self.config.jaw_closed)
        self._phase_steps += 1
        if self._phase_steps >= max(1, int(self.config.close_steps)):
            print("[CuroboPolicy] CLOSE done → ATTACH.")
            self._phase = Phase.ATTACH
        return action

    # ------------------------------------------------------------------
    # ATTACH — object spheres on robot, disable world copy (instant)
    # ------------------------------------------------------------------

    def _step_attach(self, env: gym.Env, device: torch.device) -> torch.Tensor:
        self._attach_grasped_object(env, device)
        print("[CuroboPolicy] ATTACH done → PLACE.")
        self._phase = Phase.PLACE
        self._traj = None
        self._step_idx = 0
        return self._step_place(env, device)

    def _attach_grasped_object(self, env: gym.Env, device: torch.device) -> None:
        assert self._planner is not None
        names = self._obstacle_names_matching(self.config.goal_object)
        if not names:
            all_names = self._planner.scene_collision_checker.get_obstacle_names(0)
            raise RuntimeError(
                f"No cuRobo obstacles match goal_object '{self.config.goal_object}'. "
                f"Have: {all_names}"
            )

        scene_model = self._planner.scene_collision_checker.scene_model
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
        print(f"[CuroboPolicy] ATTACH obstacles={names}")
        with torch.inference_mode(False):
            am = self._planner.attachment_manager
            q = self._current_planner_joint_state(env, device)
            spheres_w = am.fit_spheres(obstacles, num_spheres=_ATTACH_NUM_SPHERES)

            ee = self._planner.compute_kinematics(q).tool_poses.get_link_pose(
                self._planner.tool_frames[0]
            )
            centers_ee = ee.inverse().transform_points(spheres_w[:, :3].contiguous())
            centers_ee = centers_ee.reshape(-1, 3)
            spheres_ee = torch.cat([centers_ee, spheres_w[:, 3:4]], dim=-1)

            # am.update(spheres_ee, q, link_name="attached_object", world_objects_pose_offset=None)
            for name in names:
                self._planner.scene_collision_checker.enable_obstacle(name, enable=False, env_idx=0)
            am._disabled_obstacle_names = list(names)
            am._disabled_num_envs = 1
            print(f"[CuroboPolicy] ATTACH fitted {spheres_ee.shape[0]} spheres on attached_object.")

    def _obstacle_names_matching(self, entity_name: str) -> list[str]:
        """USD obstacle names whose prim path contains ``/{entity_name}``."""
        assert self._planner is not None
        names = self._planner.scene_collision_checker.get_obstacle_names(0)
        needle = f"/{entity_name}"
        return [n for n in names if needle in n or n.endswith(entity_name)]

    # ------------------------------------------------------------------
    # PLACE — plan above sorting_box, jaw closed
    # ------------------------------------------------------------------

    def _step_place(self, env: gym.Env, device: torch.device) -> torch.Tensor:
        if self._traj is None:
            goal_xyz, goal_quat = self._place_pose_in_robot_base(env, device)
            self._plan_to_pose(env, device, goal_xyz, goal_quat, label="PLACE")

        action = self._playback_traj(device, jaw=self.config.jaw_closed)
        if action is not None:
            return action

        print(f"[CuroboPolicy] PLACE done → OPEN ({self.config.open_steps} steps).")
        self._phase = Phase.OPEN
        self._phase_steps = 0
        return self._step_open(device)

    # ------------------------------------------------------------------
    # OPEN — hold arm, open jaw
    # ------------------------------------------------------------------

    def _step_open(self, device: torch.device) -> torch.Tensor:
        action = self._hold_with_jaw(device, self.config.jaw_open)
        self._phase_steps += 1
        if self._phase_steps >= max(1, int(self.config.open_steps)):
            print("[CuroboPolicy] OPEN done → DONE.")
            self._phase = Phase.DONE
        return action

    # ------------------------------------------------------------------
    # Shared motion helpers
    # ------------------------------------------------------------------

    def _playback_traj(self, device: torch.device, *, jaw: float) -> torch.Tensor | None:
        """Advance one waypoint, or None when the trajectory is finished."""
        assert self._traj is not None
        if self._step_idx >= self._traj.shape[0]:
            return None
        q = self._traj[self._step_idx]
        self._step_idx += 1
        action = self._planner_q_to_action(q, device, jaw=jaw)
        self._hold_action = action
        return action

    def _hold_with_jaw(self, device: torch.device, jaw: float) -> torch.Tensor:
        assert self._hold_action is not None
        action = self._hold_action.clone()
        action[_JAW_INDEX] = float(jaw)
        self._hold_action = action
        return action

    def _plan_to_pose(
        self,
        env: gym.Env,
        device: torch.device,
        goal_xyz: torch.Tensor,
        goal_quat_xyzw: torch.Tensor,
        *,
        label: str,
    ) -> None:
        from curobo.types import GoalToolPose
        from isaaclab.utils.math import convert_quat

        assert self._planner is not None
        self._active_goal_b = goal_xyz.detach()
        self._active_goal_quat_xyzw = goal_quat_xyzw.detach()

        # policy_runner wraps get_action in inference_mode(); cuRobo stores goal
        # tensors by reference then copy_s into them — clone to normal tensors.
        with torch.inference_mode(False):
            self._ensure_collision_world(env)
            goal_xyz = goal_xyz.detach().clone()
            goal_quat_xyzw = goal_quat_xyzw.detach().clone()
            goal_quat_wxyz = convert_quat(goal_quat_xyzw, to="wxyz").clone()
            q_start = self._current_planner_joint_state(env, device)
            self._maybe_open_viser(q_start, goal_xyz, goal_quat_wxyz)
            goal = GoalToolPose(
                tool_frames=list(self._planner.tool_frames),
                position=goal_xyz.view(1, 1, 1, 1, 3).clone(),
                quaternion=goal_quat_wxyz.view(1, 1, 1, 1, 4).clone(),
            )

            print(
                f"[CuroboPolicy] {label} planning EE "
                f"pos=({goal_xyz[0]:.3f}, {goal_xyz[1]:.3f}, {goal_xyz[2]:.3f})…"
            )
            result = self._planner.plan_pose(goal, q_start, max_attempts=10)
            if result is None or not bool(result.success.any()):
                print(f"[CuroboPolicy] {label} failed — {_format_plan_failure(result)}")
                hold = self._current_joint_action(env, device)
                name_to_idx = self._sim_name_to_index(env)
                row = torch.stack([hold[name_to_idx[n]] for n in self._planner.joint_names])
                self._traj = row.unsqueeze(0).detach()
                self._hold_action = hold.detach()
                self._step_idx = 0
                return

            positions = result.get_interpolated_plan().position
            while positions.ndim > 2:
                positions = positions[0]
            stride = max(1, int(self.config.waypoint_stride))
            self._traj = positions[::stride].detach().contiguous()
            self._step_idx = 0
            print(f"[CuroboPolicy] {label} plan OK — {self._traj.shape[0]} waypoints.")

    # ------------------------------------------------------------------
    # Planner / world setup
    # ------------------------------------------------------------------

    def _robot_yml_path(self) -> Path:
        if self.config.robot_yml:
            path = Path(self.config.robot_yml).expanduser().resolve()
        else:
            path = Path(_DEFAULT_ROBOT_YML).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"cuRobo robot YAML not found: {path}. "
                "Generate it with: python -m arena_so101.generate_curobo_config --skip-usd-convert"
            )
        return path

    def _ensure_planner(self, device: torch.device) -> None:
        if self._planner is not None:
            return
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo.types import DeviceCfg

        if device.type != "cuda":
            raise RuntimeError(f"cuRobo requires a CUDA device, got {device}")

        yml = self._robot_yml_path()
        print(f"[CuroboPolicy] Loading robot config: {yml}")
        with torch.inference_mode(False):
            cfg = MotionPlannerCfg.create(
                robot=str(yml),
                scene_model=None,
                collision_cache=dict(DEFAULT_COLLISION_CACHE),
                self_collision_check=self.config.self_collision_check,
                use_cuda_graph=self.config.use_cuda_graph,
                position_tolerance=self.config.position_tolerance,
                orientation_tolerance=self.config.orientation_tolerance,
                device_cfg=DeviceCfg(device=device, dtype=torch.float32),
            )
            self._planner = MotionPlanner(cfg)
        self._world_loaded = False
        print(
            f"[CuroboPolicy] Planner ready. active joints={self._planner.joint_names} "
            f"tool_frames={self._planner.tool_frames}"
        )

    def _ensure_collision_world(self, env: gym.Env) -> None:
        if self._world_loaded:
            return
        assert self._planner is not None
        scene = load_world_from_env(self._planner, env, env_id=0)
        self._collision_scene = scene if self.config.debug_viser else None
        self._world_loaded = True
        print(f"[CuroboPolicy] Collision world loaded: {obstacle_counts(scene)} ({len(scene)} obstacles)")

    def _maybe_open_viser(self, q_start, goal_xyz, goal_quat_wxyz) -> None:
        if not self.config.debug_viser or self._viser is not None or self._collision_scene is None:
            return
        from shape_sorting.curobo_viz import open_collision_world_viser

        self._viser = open_collision_world_viser(
            robot_yml=self._robot_yml_path(),
            scene=self._collision_scene,
            joint_state=q_start,
            goal_position_xyz=goal_xyz.detach().cpu().tolist(),
            goal_quaternion_wxyz=goal_quat_wxyz.detach().cpu().tolist(),
            port=int(self.config.debug_viser_port),
        )

    # ------------------------------------------------------------------
    # Goal poses
    # ------------------------------------------------------------------

    def _scene_entity(self, env: gym.Env, name: str):
        scene = env.unwrapped.scene
        try:
            return scene[name]
        except KeyError as exc:
            available = sorted(scene.keys()) if hasattr(scene, "keys") else []
            raise KeyError(f"CuroboPolicy entity '{name}' not in scene. Available: {available}") from exc

    def _entity_position_in_robot_base(
        self, env: gym.Env, device: torch.device, entity_name: str
    ) -> torch.Tensor:
        """Entity origin in the robot base / URDF frame. Shape (3,)."""
        import warp as wp
        from isaaclab.utils.math import subtract_frame_transforms

        robot = env.unwrapped.scene["robot"]
        obj = self._scene_entity(env, entity_name)
        robot_pose_w = wp.to_torch(robot.data.root_pose_w)[0].to(device=device, dtype=torch.float32)
        obj_pose_w = wp.to_torch(obj.data.root_pose_w)[0].to(device=device, dtype=torch.float32)

        obj_b, _ = subtract_frame_transforms(
            robot_pose_w[0:3].unsqueeze(0),
            robot_pose_w[3:7].unsqueeze(0),
            obj_pose_w[0:3].unsqueeze(0),
            obj_pose_w[3:7].unsqueeze(0),
        )
        return obj_b[0]

    def _grasp_pose_in_robot_base(
        self, env: gym.Env, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Grasp EE pose: XY standoff on base→object line, fixed Z."""
        if not self.config.goal_object:
            raise ValueError("CuroboPolicy requires --goal_object <scene_entity_name>.")
        obj_b = self._entity_position_in_robot_base(env, device, self.config.goal_object)
        xy = obj_b[0:2]
        rho = torch.linalg.norm(xy)
        if float(rho) < 1e-6:
            raise RuntimeError(
                f"CuroboPolicy: goal_object '{self.config.goal_object}' is directly above "
                "the robot base; cannot define a horizontal approach line."
            )
        xy_goal = xy * (1.0 - _GOAL_XY_STANDOFF_M / rho)
        return so101_ee_pose_xyzw(
            float(xy_goal[0]),
            float(xy_goal[1]),
            _GOAL_Z_M,
            tilt=_GOAL_TILT_RAD,
            roll=_GOAL_ROLL_RAD,
            device=device,
        )

    def _place_pose_in_robot_base(
        self, env: gym.Env, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Place EE pose: directly above ``place_object`` by ``place_z_offset_m``."""
        box_b = self._entity_position_in_robot_base(env, device, self.config.place_object)
        return so101_ee_pose_xyzw(
            float(box_b[0]),
            float(box_b[1]),
            float(box_b[2]) + float(self.config.place_z_offset_m),
            tilt=_GOAL_TILT_RAD,
            roll=_GOAL_ROLL_RAD,
            device=device,
        )

    def _goal_pose_w(
        self, env: gym.Env, device: torch.device, num_envs: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """World-frame active goal for debug markers."""
        import warp as wp
        from isaaclab.utils.math import combine_frame_transforms

        if self._active_goal_b is None or self._active_goal_quat_xyzw is None:
            goal_b, goal_quat_xyzw = self._grasp_pose_in_robot_base(env, device)
        else:
            goal_b = self._active_goal_b
            goal_quat_xyzw = self._active_goal_quat_xyzw

        robot = env.unwrapped.scene["robot"]
        robot_pose_w = wp.to_torch(robot.data.root_pose_w).to(device=device, dtype=torch.float32)
        t = goal_b.unsqueeze(0).expand(num_envs, -1)
        q = goal_quat_xyzw.unsqueeze(0).expand(num_envs, -1)
        return combine_frame_transforms(robot_pose_w[:, 0:3], robot_pose_w[:, 3:7], t, q)

    # ------------------------------------------------------------------
    # Joint / action helpers
    # ------------------------------------------------------------------

    def _sim_name_to_index(self, env: gym.Env) -> dict[str, int]:
        if self._sim_joint_indices is not None:
            return self._sim_joint_indices
        robot = env.unwrapped.scene["robot"]
        names = list(robot.data.joint_names)
        self._sim_joint_indices = {n: i for i, n in enumerate(names)}
        missing = [n for n in SIM_JOINT_NAMES if n not in self._sim_joint_indices]
        if missing:
            raise RuntimeError(f"Sim robot is missing expected SO-101 joints: {missing}. Have: {names}")
        return self._sim_joint_indices

    def _current_joint_action(self, env: gym.Env, device: torch.device) -> torch.Tensor:
        robot = env.unwrapped.scene["robot"]
        q = robot.data.joint_pos[0].detach().to(device=device, dtype=torch.float32)
        name_to_idx = self._sim_name_to_index(env)
        action = torch.zeros(len(SIM_JOINT_NAMES), device=device, dtype=torch.float32)
        for i, name in enumerate(SIM_JOINT_NAMES):
            action[i] = q[name_to_idx[name]]
        return action

    def _current_planner_joint_state(self, env: gym.Env, device: torch.device):
        from curobo.types import JointState

        assert self._planner is not None
        robot = env.unwrapped.scene["robot"]
        q = robot.data.joint_pos[0].detach().to(device=device, dtype=torch.float32)
        name_to_idx = self._sim_name_to_index(env)
        active = [q[name_to_idx[name]] for name in self._planner.joint_names]
        # clone: JointState may be stored in cuRobo goal buffers and mutated inplace.
        q_active = torch.stack(active).unsqueeze(0).contiguous().clone()
        return JointState.from_position(q_active, joint_names=list(self._planner.joint_names))

    def _planner_q_to_action(
        self, q_planner: torch.Tensor, device: torch.device, *, jaw: float
    ) -> torch.Tensor:
        assert self._planner is not None
        action = torch.zeros(len(SIM_JOINT_NAMES), device=device, dtype=torch.float32)
        sim_index = {n: i for i, n in enumerate(SIM_JOINT_NAMES)}
        for i, name in enumerate(self._planner.joint_names):
            action[sim_index[name]] = q_planner[i]
        action[_JAW_INDEX] = float(jaw)
        return action

    # ------------------------------------------------------------------
    # Debug viz
    # ------------------------------------------------------------------

    def _tool_frame_name(self) -> str:
        if self._planner is not None and self._planner.tool_frames:
            return str(self._planner.tool_frames[0])
        return "gripper"

    def _ensure_markers(self, env: gym.Env) -> None:
        if not self.config.debug_viz:
            return
        if self._goal_marker is not None and self._ee_marker is not None:
            return

        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import FRAME_MARKER_CFG

        scale = float(self.config.marker_frame_scale)

        goal_cfg = FRAME_MARKER_CFG.copy()
        goal_cfg.prim_path = "/Visuals/CuroboPolicy/goal_frame"
        goal_cfg.markers["frame"].scale = (scale, scale, scale)
        self._goal_marker = VisualizationMarkers(goal_cfg)

        ee_cfg = FRAME_MARKER_CFG.copy()
        ee_cfg.prim_path = "/Visuals/CuroboPolicy/ee_frame"
        ee_cfg.markers["frame"].scale = (scale, scale, scale)
        self._ee_marker = VisualizationMarkers(ee_cfg)

        robot = env.unwrapped.scene["robot"]
        tool = self._tool_frame_name()
        body_ids, body_names = robot.find_bodies(tool)
        if not body_ids:
            raise RuntimeError(
                f"CuroboPolicy debug viz: robot has no body matching tool frame '{tool}'. "
                f"Have: {list(robot.body_names)}"
            )
        self._ee_body_id = int(body_ids[0])
        print(f"[CuroboPolicy] Debug markers ready (goal frame + EE frame '{body_names[0]}').")

    def _update_debug_viz(self, env: gym.Env, device: torch.device) -> None:
        if not self.config.debug_viz:
            return

        import warp as wp

        self._ensure_markers(env)
        assert self._goal_marker is not None and self._ee_marker is not None
        assert self._ee_body_id is not None

        robot = env.unwrapped.scene["robot"]
        num_envs = env.unwrapped.num_envs

        goal_pos_w, goal_quat_xyzw = self._goal_pose_w(env, device, num_envs)
        self._goal_marker.visualize(translations=goal_pos_w, orientations=goal_quat_xyzw)

        ee_pose_w = wp.to_torch(robot.data.body_link_pose_w)[:, self._ee_body_id, :].to(
            device=device, dtype=torch.float32
        )
        self._ee_marker.visualize(translations=ee_pose_w[:, 0:3], orientations=ee_pose_w[:, 3:7])
