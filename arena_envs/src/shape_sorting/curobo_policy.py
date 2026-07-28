"""Minimal cuRobo reach policy for SO-101 smoke tests.

Plans once to a fixed end-effector position, then plays back absolute joint
waypoints. Goal ``(x, y, z)`` is in the robot base / URDF frame by default, or
in a scene object's frame when ``--goal_object`` is set. Requires
``--embodiment so101_abs_joint``.

Example::

    python submodules/IsaacLab-Arena/isaaclab_arena/evaluation/policy_runner.py \\
      --viz kit \\
      --policy_type shape_sorting.curobo_policy.CuroboPolicy \\
      --num_steps 200 \\
      --external_environment_class_path shape_sorting.shape_sorting_env:ShapeSortingEnvironment \\
      shape_sorting_test \\
      --embodiment so101_abs_joint

Reach 5 cm above a piece (object frame)::

    ... --goal_object shape_piece_cube --goal_x 0 --goal_y 0 --goal_z 0.05
"""

from __future__ import annotations

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


@dataclass
class CuroboPolicyCfg(PolicyCfg):
    """Configure a one-shot cuRobo EE reach for SO-101 absolute-joint control."""

    robot_yml: str = ""
    """Path to cuRobo ``so101.yml``. Empty uses the generated package default."""

    goal_x: float = 0.18
    """Goal EE x [m]. Robot base frame, or ``goal_object`` frame if set."""

    goal_y: float = 0.0
    """Goal EE y [m]. Robot base frame, or ``goal_object`` frame if set."""

    goal_z: float = 0.12
    """Goal EE z [m]. Robot base frame, or ``goal_object`` frame if set."""

    goal_object: str = ""
    """Scene entity name whose frame interprets the goal. Empty = robot base."""

    position_tolerance: float = 0.015
    """cuRobo position convergence tolerance [m]."""

    orientation_tolerance: float = 3.14
    """Orientation tolerance [rad]. Default ~ignore (SO-101 is 5-DOF)."""

    jaw_open: float = 1.7453292519943295
    """Jaw command while executing the arm plan [rad] (workshop open)."""

    use_cuda_graph: bool = False
    """Enable cuRobo CUDA graphs (faster steady-state; slower first compile)."""

    self_collision_check: bool = False
    """Enable self-collision costs. Off by default for a more reliable smoke test."""

    waypoint_stride: int = 2
    """Play every N-th interpolated waypoint (1 = all). Speeds up playback vs sim dt."""

    debug_viz: bool = True
    """Draw goal sphere + EE frame markers in the Kit viewport."""

    marker_frame_scale: float = 0.08
    """World-scale of the EE frame marker axes [m]."""

    marker_sphere_radius: float = 0.015
    """Radius of the goal sphere marker [m]."""


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

        assert self._planner is not None
        # policy_runner wraps get_action in torch.inference_mode(); cuRobo IK needs autograd.
        with torch.inference_mode(False):
            q_start = self._current_planner_joint_state(env, device)
            goal_xyz = self._goal_position_in_robot_base(env, device)
            goal_pos = goal_xyz.view(1, 1, 1, 1, 3)
            # wxyz identity — orientation is effectively free via large tolerance.
            goal_quat = torch.tensor(
                [[[[[1.0, 0.0, 0.0, 0.0]]]]], device=device, dtype=torch.float32
            )
            goal = GoalToolPose(
                tool_frames=list(self._planner.tool_frames),
                position=goal_pos,
                quaternion=goal_quat,
            )

            print(
                f"[CuroboPolicy] Planning to EE "
                f"({goal_xyz[0]:.3f}, {goal_xyz[1]:.3f}, {goal_xyz[2]:.3f}) "
                f"in robot base frame"
                + (
                    f" (from '{self.config.goal_object}' local "
                    f"({self.config.goal_x:.3f}, {self.config.goal_y:.3f}, {self.config.goal_z:.3f}))"
                    if self.config.goal_object
                    else ""
                )
                + "…"
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

    def _goal_offset(self, device: torch.device) -> torch.Tensor:
        """Configured goal offset as a (3,) tensor."""
        return torch.tensor(
            [self.config.goal_x, self.config.goal_y, self.config.goal_z],
            device=device,
            dtype=torch.float32,
        )

    def _scene_entity(self, env: gym.Env, name: str):
        scene = env.unwrapped.scene
        try:
            return scene[name]
        except KeyError as exc:
            available = sorted(scene.keys()) if hasattr(scene, "keys") else []
            raise KeyError(
                f"CuroboPolicy goal_object '{name}' not in scene. Available: {available}"
            ) from exc

    def _goal_position_in_robot_base(self, env: gym.Env, device: torch.device) -> torch.Tensor:
        """Resolve configured goal into the robot base / URDF frame. Shape (3,)."""
        import warp as wp
        from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms

        offset = self._goal_offset(device)
        if not self.config.goal_object:
            return offset

        robot = env.unwrapped.scene["robot"]
        obj = self._scene_entity(env, self.config.goal_object)
        robot_pose_w = wp.to_torch(robot.data.root_pose_w)[0].to(device=device, dtype=torch.float32)
        obj_pose_w = wp.to_torch(obj.data.root_pose_w)[0].to(device=device, dtype=torch.float32)

        goal_w, _ = combine_frame_transforms(
            obj_pose_w[0:3].unsqueeze(0),
            obj_pose_w[3:7].unsqueeze(0),
            offset.unsqueeze(0),
        )
        goal_b, _ = subtract_frame_transforms(
            robot_pose_w[0:3].unsqueeze(0),
            robot_pose_w[3:7].unsqueeze(0),
            goal_w,
        )
        return goal_b[0]

    def _goal_position_w(self, env: gym.Env, device: torch.device, num_envs: int) -> torch.Tensor:
        """World-frame goal positions for debug markers. Shape (num_envs, 3)."""
        import warp as wp
        from isaaclab.utils.math import combine_frame_transforms

        offset = self._goal_offset(device).unsqueeze(0).expand(num_envs, -1)
        if self.config.goal_object:
            obj = self._scene_entity(env, self.config.goal_object)
            pose_w = wp.to_torch(obj.data.root_pose_w).to(device=device, dtype=torch.float32)
        else:
            robot = env.unwrapped.scene["robot"]
            pose_w = wp.to_torch(robot.data.root_pose_w).to(device=device, dtype=torch.float32)
        goal_w, _ = combine_frame_transforms(pose_w[:, 0:3], pose_w[:, 3:7], offset)
        return goal_w

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
        from isaaclab.markers.config import FRAME_MARKER_CFG, SPHERE_MARKER_CFG

        sphere_cfg = SPHERE_MARKER_CFG.copy()
        sphere_cfg.prim_path = "/Visuals/CuroboPolicy/goal"
        sphere_cfg.markers["sphere"].radius = float(self.config.marker_sphere_radius)
        self._goal_marker = VisualizationMarkers(sphere_cfg)

        frame_cfg = FRAME_MARKER_CFG.copy()
        frame_cfg.prim_path = "/Visuals/CuroboPolicy/ee_frame"
        scale = float(self.config.marker_frame_scale)
        frame_cfg.markers["frame"].scale = (scale, scale, scale)
        self._ee_marker = VisualizationMarkers(frame_cfg)

        robot = env.unwrapped.scene["robot"]
        tool = self._tool_frame_name()
        body_ids, body_names = robot.find_bodies(tool)
        if not body_ids:
            raise RuntimeError(
                f"CuroboPolicy debug viz: robot has no body matching tool frame '{tool}'. "
                f"Have: {list(robot.body_names)}"
            )
        self._ee_body_id = int(body_ids[0])
        print(f"[CuroboPolicy] Debug markers ready (goal sphere + EE frame '{body_names[0]}').")

    def _update_debug_viz(self, env: gym.Env, device: torch.device) -> None:
        if not self.config.debug_viz:
            return

        import warp as wp

        self._ensure_markers(env)
        assert self._goal_marker is not None and self._ee_marker is not None
        assert self._ee_body_id is not None

        robot = env.unwrapped.scene["robot"]
        num_envs = env.unwrapped.num_envs

        goal_pos_w = self._goal_position_w(env, device, num_envs)
        self._goal_marker.visualize(translations=goal_pos_w)

        ee_pose_w = wp.to_torch(robot.data.body_link_pose_w)[:, self._ee_body_id, :].to(
            device=device, dtype=torch.float32
        )
        self._ee_marker.visualize(translations=ee_pose_w[:, 0:3], orientations=ee_pose_w[:, 3:7])
