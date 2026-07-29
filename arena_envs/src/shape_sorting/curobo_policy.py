"""Minimal cuRobo reach policy for SO-101 smoke tests.

Plans once to a 5-DoF EE pose (position + in-plane tilt + wrist roll), then
plays back absolute joint waypoints. The goal XY lies on the line from the
robot base to ``goal_object``, standoff toward the robot; Z is fixed. Requires
``--embodiment so101_abs_joint``.

Example::

    python submodules/IsaacLab-Arena/isaaclab_arena/evaluation/policy_runner.py \\
      --viz kit \\
      --policy_type shape_sorting.curobo_policy.CuroboPolicy \\
      --num_steps 200 \\
      --external_environment_class_path shape_sorting.shape_sorting_env:ShapeSortingEnvironment \\
      shape_sorting_test \\
      --embodiment so101_abs_joint

Reach near a piece::

    ... --goal_object shape_piece_cube
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import torch
from gymnasium.spaces.dict import Dict as GymSpacesDict

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

from arena_so101.mapping import SIM_JOINT_NAMES

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

# Approach pose relative to goal_object (robot base frame).
_GOAL_XY_STANDOFF_M = 0.03
"""Horizontal distance from the object origin toward the robot base [m]."""

_GOAL_Z_M = 0.12
"""Goal EE height in the robot base frame [m]."""

_GOAL_TILT_RAD = 0.0
"""In-plane tilt about tool Y (left); 0 = upright (Z up)."""

_GOAL_ROLL_RAD = 0.0
"""Wrist roll about tool Z [rad]."""


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

    Position ``(x, y, z)`` is in the robot base frame. Orientation is constrained
    to the arm plane (vertical plane through the base and the EE). At
    ``tilt = roll = 0`` the tool frame is FLU in that plane: **X forward**
    (horizontal radial), **Y left** (plane normal), **Z up**. ``tilt`` pitches
    about tool Y (in-plane); ``roll`` rotates about tool Z.

    Args:
        x: EE x [m].
        y: EE y [m].
        z: EE z [m].
        tilt: In-plane pitch about tool Y [rad].
        roll: Roll about tool Z [rad].

    Returns:
        ``(position, quaternion_xyzw)`` with shapes ``(3,)`` and ``(4,)``.
    """
    from isaaclab.utils.math import quat_from_matrix

    pos = torch.tensor([x, y, z], device=device, dtype=dtype)

    # Arm-plane bearing from XY. Fallback when EE is above the base origin.
    rho = math.hypot(x, y)
    if rho < 1e-8:
        psi = 0.0
    else:
        psi = math.atan2(y, x)

    # Zero pose (tilt=roll=0): X forward, Y left, Z up.
    x0 = torch.tensor([math.cos(psi), math.sin(psi), 0.0], device=device, dtype=dtype)
    z0 = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    y0 = torch.linalg.cross(z0, x0)  # left = Z × X
    y0 = y0 / y0.norm().clamp_min(1e-8)
    rot0 = torch.stack([x0, y0, z0], dim=1)

    # Intrinsic tilt about Y, then roll about Z: R = R0 @ Ry(tilt) @ Rz(roll).
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
    quat_xyzw = quat_from_matrix(rot.unsqueeze(0))[0]
    return pos, quat_xyzw.to(dtype=dtype)


@dataclass
class CuroboPolicyCfg(PolicyCfg):
    """Configure a one-shot cuRobo EE reach for SO-101 absolute-joint control."""

    robot_yml: str = ""
    """Path to cuRobo ``so101.yml``. Empty uses the generated package default."""

    goal_object: str = "shape_piece_cube"
    """Scene entity used to place the approach goal (required)."""

    position_tolerance: float = 0.015
    """cuRobo position convergence tolerance [m]."""

    orientation_tolerance: float = 0.1
    """Orientation tolerance [rad]. Tight enough that the synthesized 5-DoF pose is enforced."""

    jaw_open: float = 1.7453292519943295
    """Jaw command while executing the arm plan [rad] (workshop open)."""

    use_cuda_graph: bool = False
    """Enable cuRobo CUDA graphs (faster steady-state; slower first compile)."""

    self_collision_check: bool = False
    """Enable self-collision costs. Off by default for a more reliable smoke test."""

    waypoint_stride: int = 2
    """Play every N-th interpolated waypoint (1 = all). Speeds up playback vs sim dt."""

    debug_viz: bool = True
    """Draw goal + EE frame markers in the Kit viewport."""

    marker_frame_scale: float = 0.08
    """World-scale of the frame marker axes [m]."""


@register_policy
class CuroboPolicy(PolicyBase[CuroboPolicyCfg]):
    """Plan to a fixed EE pose with cuRobo, then stream absolute joint targets."""

    name = "curobo_reach"

    def __init__(self, config: CuroboPolicyCfg):
        super().__init__(config)
        self._planner = None
        self._traj: torch.Tensor | None = None  # (T, n_active_joints)
        self._step_idx = 0
        self._hold_action: torch.Tensor | None = None
        self._sim_joint_indices: dict[str, int] | None = None
        self._goal_marker = None
        self._ee_marker = None
        self._ee_body_id: int | None = None

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        # Replan on next get_action (single-env smoke test; ignore env_ids).
        self._traj = None
        self._step_idx = 0
        self._hold_action = None

    def close(self) -> None:
        self._planner = None
        self._traj = None
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
        if self._traj is None:
            self._plan(env, device)

        assert self._traj is not None
        if self._step_idx < self._traj.shape[0]:
            q = self._traj[self._step_idx]
            self._step_idx += 1
            action_1 = self._planner_q_to_action(q, device)
            self._hold_action = action_1
        else:
            action_1 = self._hold_action
            if action_1 is None:
                action_1 = self._current_joint_action(env, device)

        self._update_debug_viz(env, device)
        return action_1.unsqueeze(0).expand(num_envs, -1).contiguous()

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
        cfg = MotionPlannerCfg.create(
            robot=str(yml),
            scene_model=None,
            self_collision_check=self.config.self_collision_check,
            use_cuda_graph=self.config.use_cuda_graph,
            position_tolerance=self.config.position_tolerance,
            orientation_tolerance=self.config.orientation_tolerance,
            device_cfg=DeviceCfg(device=device, dtype=torch.float32),
        )
        self._planner = MotionPlanner(cfg)
        print(
            f"[CuroboPolicy] Planner ready. active joints={self._planner.joint_names} "
            f"tool_frames={self._planner.tool_frames}"
        )

    def _plan(self, env: gym.Env, device: torch.device) -> None:
        from curobo.types import GoalToolPose
        from isaaclab.utils.math import convert_quat

        assert self._planner is not None
        # policy_runner wraps get_action in torch.inference_mode(); cuRobo IK needs autograd.
        with torch.inference_mode(False):
            q_start = self._current_planner_joint_state(env, device)
            goal_xyz, goal_quat_xyzw = self._goal_pose_in_robot_base(env, device)
            # cuRobo Pose / GoalToolPose use wxyz; Isaac Lab 3 uses xyzw.
            goal_quat_wxyz = convert_quat(goal_quat_xyzw, to="wxyz")
            goal = GoalToolPose(
                tool_frames=list(self._planner.tool_frames),
                position=goal_xyz.view(1, 1, 1, 1, 3),
                quaternion=goal_quat_wxyz.view(1, 1, 1, 1, 4),
            )

            print(
                f"[CuroboPolicy] Planning to EE "
                f"pos=({goal_xyz[0]:.3f}, {goal_xyz[1]:.3f}, {goal_xyz[2]:.3f}) "
                f"quat_xyzw=({goal_quat_xyzw[0]:.3f}, {goal_quat_xyzw[1]:.3f}, "
                f"{goal_quat_xyzw[2]:.3f}, {goal_quat_xyzw[3]:.3f}) "
                f"in robot base frame (object='{self.config.goal_object}', "
                f"tilt={_GOAL_TILT_RAD:.3f}, roll={_GOAL_ROLL_RAD:.3f})…"
            )
            result = self._planner.plan_pose(goal, q_start)
            if result is None or not bool(result.success.any()):
                print("[CuroboPolicy] Planning failed — holding current joint pose.")
                hold = self._current_joint_action(env, device)
                name_to_idx = self._sim_name_to_index(env)
                row = torch.stack([hold[name_to_idx[n]] for n in self._planner.joint_names])
                self._traj = row.unsqueeze(0).detach()
                self._hold_action = hold.detach()
                self._step_idx = 0
                return

            interpolated = result.get_interpolated_plan()
            positions = interpolated.position
            while positions.ndim > 2:
                positions = positions[0]
            stride = max(1, int(self.config.waypoint_stride))
            self._traj = positions[::stride].detach().contiguous()
            self._step_idx = 0
            self._hold_action = self._planner_q_to_action(self._traj[-1], device).detach()
            print(f"[CuroboPolicy] Plan OK — {self._traj.shape[0]} waypoints (stride={stride}).")

    def _scene_entity(self, env: gym.Env, name: str):
        scene = env.unwrapped.scene
        try:
            return scene[name]
        except KeyError as exc:
            available = sorted(scene.keys()) if hasattr(scene, "keys") else []
            raise KeyError(
                f"CuroboPolicy goal_object '{name}' not in scene. Available: {available}"
            ) from exc

    def _object_position_in_robot_base(self, env: gym.Env, device: torch.device) -> torch.Tensor:
        """Object origin in the robot base / URDF frame. Shape (3,)."""
        import warp as wp
        from isaaclab.utils.math import subtract_frame_transforms

        if not self.config.goal_object:
            raise ValueError("CuroboPolicy requires --goal_object <scene_entity_name>.")

        robot = env.unwrapped.scene["robot"]
        obj = self._scene_entity(env, self.config.goal_object)
        robot_pose_w = wp.to_torch(robot.data.root_pose_w)[0].to(device=device, dtype=torch.float32)
        obj_pose_w = wp.to_torch(obj.data.root_pose_w)[0].to(device=device, dtype=torch.float32)

        obj_b, _ = subtract_frame_transforms(
            robot_pose_w[0:3].unsqueeze(0),
            robot_pose_w[3:7].unsqueeze(0),
            obj_pose_w[0:3].unsqueeze(0),
            obj_pose_w[3:7].unsqueeze(0),
        )
        return obj_b[0]

    def _goal_pose_in_robot_base(
        self, env: gym.Env, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Approach pose in robot base frame: XY standoff on base→object line, fixed Z."""
        obj_b = self._object_position_in_robot_base(env, device)
        xy = obj_b[0:2]
        rho = torch.linalg.norm(xy)
        if float(rho) < 1e-6:
            raise RuntimeError(
                f"CuroboPolicy: goal_object '{self.config.goal_object}' is directly above "
                "the robot base; cannot define a horizontal approach line."
            )
        # Point on the base→object ray, standoff toward the robot from the object.
        xy_goal = xy * (1.0 - _GOAL_XY_STANDOFF_M / rho)
        return so101_ee_pose_xyzw(
            float(xy_goal[0]),
            float(xy_goal[1]),
            _GOAL_Z_M,
            tilt=_GOAL_TILT_RAD,
            roll=_GOAL_ROLL_RAD,
            device=device,
        )

    def _goal_pose_w(
        self, env: gym.Env, device: torch.device, num_envs: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """World-frame goal pose for debug markers. Shapes (num_envs, 3) and (num_envs, 4) xyzw."""
        import warp as wp
        from isaaclab.utils.math import combine_frame_transforms

        goal_b, goal_quat_xyzw = self._goal_pose_in_robot_base(env, device)
        robot = env.unwrapped.scene["robot"]
        robot_pose_w = wp.to_torch(robot.data.root_pose_w).to(device=device, dtype=torch.float32)
        t = goal_b.unsqueeze(0).expand(num_envs, -1)
        q = goal_quat_xyzw.unsqueeze(0).expand(num_envs, -1)
        return combine_frame_transforms(robot_pose_w[:, 0:3], robot_pose_w[:, 3:7], t, q)

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
        q_active = torch.stack(active).unsqueeze(0).contiguous()
        return JointState.from_position(q_active, joint_names=list(self._planner.joint_names))

    def _planner_q_to_action(self, q_planner: torch.Tensor, device: torch.device) -> torch.Tensor:
        assert self._planner is not None
        action = torch.zeros(len(SIM_JOINT_NAMES), device=device, dtype=torch.float32)
        sim_index = {n: i for i, n in enumerate(SIM_JOINT_NAMES)}
        for i, name in enumerate(self._planner.joint_names):
            action[sim_index[name]] = q_planner[i]
        action[sim_index["Jaw"]] = float(self.config.jaw_open)
        return action

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
