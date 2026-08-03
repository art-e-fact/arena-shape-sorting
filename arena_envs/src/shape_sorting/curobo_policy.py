"""cuRobo pick-and-place smoke-test policy for SO-101.

Sequence per shape: APPROACH → CLOSE → ATTACH → PLACE → OPEN, then the next
shape from ``env.unwrapped.cfg.shapes``. After the last shape: HOME → DONE.
Plans via ``MotionClient`` (see ``curobo_motion``); collision world sync lives
there too. Requires ``--embodiment so101_abs_joint``.

Example::

    python submodules/IsaacLab-Arena/isaaclab_arena/evaluation/policy_runner.py \\
      --viz kit \\
      --policy_type shape_sorting.curobo_policy.CuroboPolicy \\
      --num_steps 1200 \\
      --external_environment_class_path shape_sorting.shape_sorting_env:ShapeSortingEnvironment \\
      shape_sorting_test \\
      --embodiment so101_abs_joint
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto

import gymnasium as gym
import torch
from gymnasium.spaces.dict import Dict as GymSpacesDict

from arena_so101.mapping import SIM_JOINT_NAMES
from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg
from shape_sorting.curobo_motion import MotionClient, MotionClientCfg, resolve_robot_yml
from shape_sorting.curobo_viz import KitFrameMarkers, NullCollisionDebugViz, ViserCollisionDebugViz
from shape_sorting.curobo_world import entity_position_in_robot_base
from shape_sorting.shape_sorting_env import ShapeInfo

# Grasp pose relative to goal_object (robot base frame).
_GOAL_XY_STANDOFF_M = 0.03
_GOAL_Z_M = 0.145
_GOAL_TILT_RAD = 0.0
_GOAL_ROLL_RAD = 0.0

# Workshop USD Jaw limits (degrees → radians), same as arena_so101.
_JAW_OPEN_RAD = math.radians(100.0)
_JAW_CLOSE_RAD = math.radians(-10.0)
_JAW_INDEX = SIM_JOINT_NAMES.index("Jaw")

_HOME_JOINT_RAD = tuple(
    math.radians(v)
    for v in (
        -1.6e-05,  # shoulder_pan / Rotation
        4.5e-03,   # shoulder_lift / Pitch
        5.5e-03,   # elbow_flex / Elbow
        2.1e-03,   # wrist_flex / Wrist_Pitch
        -4.6e-06,  # wrist_roll / Wrist_Roll
        0.200,     # gripper / Jaw
    )
)


class Phase(Enum):
    APPROACH = auto()
    CLOSE = auto()
    ATTACH = auto()
    PLACE = auto()
    OPEN = auto()
    HOME = auto()
    DONE = auto()


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

    goal_object: str = ""
    """Optional single scene entity to grasp when ``env.cfg.shapes`` is absent."""

    place_object: str = "sorting_box"
    """Unused; place XY/Z come from ``hole_frames`` matching the current shape."""

    place_z_offset_m: float = 0.02
    """Object-origin height above the matched lid hole in the robot base frame [m]."""

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

    jaw_empty_tol: float = math.radians(5.0)
    """If measured jaw is within this of ``jaw_closed`` after CLOSE, treat grasp as empty."""

    open_steps: int = 20
    """Sim steps to hold the open jaw before HOME."""

    home_steps: int = 30
    """Sim steps to cosine-interpolate to the home joint pose."""

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
    """Pick each ``env.cfg.shapes`` piece, then HOME."""

    name = "curobo_reach"

    def __init__(self, config: CuroboPolicyCfg):
        super().__init__(config)
        self._motion: MotionClient | None = None
        self._kit_markers: KitFrameMarkers | None = None
        self._phase = Phase.APPROACH
        self._traj: torch.Tensor | None = None
        self._step_idx = 0
        self._phase_steps = 0
        self._hold_action: torch.Tensor | None = None
        self._shape_idx = 0
        self._shapes: list[ShapeInfo] | None = None

    # ------------------------------------------------------------------
    # PolicyBase
    # ------------------------------------------------------------------

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._motion is not None:
            self._motion.detach()
            self._motion.clear_last_goal()
        self._phase = Phase.APPROACH
        self._traj = None
        self._step_idx = 0
        self._phase_steps = 0
        self._hold_action = None
        self._shape_idx = 0
        self._shapes = None

    def close(self) -> None:
        if self._motion is not None:
            self._motion.close()
            self._motion = None
        if self._kit_markers is not None:
            self._kit_markers.close()
            self._kit_markers = None
        self._traj = None
        self._shapes = None

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        device = torch.device(env.unwrapped.device)
        num_envs = env.unwrapped.num_envs
        action_dim = env.action_space.shape[-1]
        if action_dim != len(SIM_JOINT_NAMES):
            raise RuntimeError(
                f"CuroboPolicy expects so101_abs_joint (action dim {len(SIM_JOINT_NAMES)}), "
                f"got action dim {action_dim}. Pass `--embodiment so101_abs_joint`."
            )

        motion = self._ensure_motion(device)

        if self._phase is Phase.APPROACH:
            action = self._step_approach(env, device, motion)
        elif self._phase is Phase.CLOSE:
            action = self._step_close(env, device, motion)
        elif self._phase is Phase.ATTACH:
            action = self._step_attach(env, device, motion)
        elif self._phase is Phase.PLACE:
            action = self._step_place(env, device, motion)
        elif self._phase is Phase.OPEN:
            action = self._step_open(env, device, motion)
        elif self._phase is Phase.HOME:
            action = self._step_home(env, device, motion)
        else:  # DONE
            action = self._hold_action
            if action is None:
                action = motion.joint_action(env, device)

        self._update_kit_markers(env, device, motion)
        return action.unsqueeze(0).expand(num_envs, -1).contiguous()

    # ------------------------------------------------------------------
    # Phase steps
    # ------------------------------------------------------------------

    def _step_approach(
        self, env: gym.Env, device: torch.device, motion: MotionClient
    ) -> torch.Tensor:
        if self._traj is None:
            motion.detach()  # free-space approach must not carry a previous attach
            shape = self._current_shape(env)
            print(
                f"[CuroboPolicy] APPROACH shape {self._shape_idx + 1}/"
                f"{len(self._resolve_shapes(env))} ({shape.name})."
            )
            goal_xyz, goal_quat = self._grasp_pose_in_robot_base(env, device)
            plan = motion.plan_to_pose(env, device, goal_xyz, goal_quat, label="APPROACH")
            self._traj = plan.waypoints
            self._hold_action = plan.hold_action
            self._step_idx = 0

        action = self._playback_traj(device, motion, jaw=self.config.jaw_open)
        if action is not None:
            return action

        print(f"[CuroboPolicy] APPROACH done → CLOSE ({self.config.close_steps} steps).")
        self._phase = Phase.CLOSE
        self._phase_steps = 0
        return self._step_close(env, device, motion)

    def _step_close(
        self, env: gym.Env, device: torch.device, motion: MotionClient
    ) -> torch.Tensor:
        action = self._hold_with_jaw(device, self.config.jaw_closed)
        self._phase_steps += 1
        if self._phase_steps < max(1, int(self.config.close_steps)):
            return action

        jaw = motion.measured_jaw(env, device)
        if self._jaw_is_fully_closed(jaw):
            print(
                f"[CuroboPolicy] CLOSE: grasp failed, jaw fully closed "
                f"({float(jaw):.3f} ≈ {self.config.jaw_closed:.3f}) — empty grasp → APPROACH."
            )
            self._phase = Phase.APPROACH
            self._traj = None
            self._step_idx = 0
            self._phase_steps = 0
            motion.clear_last_goal()
            return self._step_approach(env, device, motion)

        print(f"[CuroboPolicy] CLOSE done (jaw={float(jaw):.3f}) → ATTACH.")
        self._phase = Phase.ATTACH
        return action

    def _step_attach(
        self, env: gym.Env, device: torch.device, motion: MotionClient
    ) -> torch.Tensor:
        motion.attach(env, device, self._current_shape(env).name)
        print("[CuroboPolicy] ATTACH done → PLACE.")
        self._phase = Phase.PLACE
        self._traj = None
        self._step_idx = 0
        return self._step_place(env, device, motion)

    def _step_place(
        self, env: gym.Env, device: torch.device, motion: MotionClient
    ) -> torch.Tensor:
        if self._traj is None:
            goal_xyz, goal_quat = self._place_pose_in_robot_base(env, device, motion)
            plan = motion.plan_to_pose(env, device, goal_xyz, goal_quat, label="PLACE")
            self._traj = plan.waypoints
            self._hold_action = plan.hold_action
            self._step_idx = 0

        action = self._playback_traj(device, motion, jaw=self.config.jaw_closed)
        if action is not None:
            return action

        print(f"[CuroboPolicy] PLACE done → OPEN ({self.config.open_steps} steps).")
        self._phase = Phase.OPEN
        self._phase_steps = 0
        return self._step_open(env, device, motion)

    def _step_open(
        self, env: gym.Env, device: torch.device, motion: MotionClient
    ) -> torch.Tensor:
        if self._phase_steps == 0:
            motion.detach()  # release collision spheres as soon as the jaw opens

        action = self._hold_with_jaw(device, self.config.jaw_open)
        self._phase_steps += 1
        if self._phase_steps < max(1, int(self.config.open_steps)):
            return action

        shapes = self._resolve_shapes(env)
        if self._shape_idx + 1 < len(shapes):
            self._shape_idx += 1
            nxt = shapes[self._shape_idx]
            print(
                f"[CuroboPolicy] OPEN done → APPROACH "
                f"shape {self._shape_idx + 1}/{len(shapes)} ({nxt.name})."
            )
            self._phase = Phase.APPROACH
            self._traj = None
            self._step_idx = 0
            self._phase_steps = 0
            motion.clear_last_goal()
            return self._step_approach(env, device, motion)

        print(
            f"[CuroboPolicy] OPEN done (all {len(shapes)} shapes) → HOME "
            f"({self.config.home_steps} steps)."
        )
        self._phase = Phase.HOME
        self._traj = None
        self._step_idx = 0
        return action

    def _step_home(
        self, env: gym.Env, device: torch.device, motion: MotionClient
    ) -> torch.Tensor:
        if self._traj is None:
            self._build_home_traj(env, device, motion)

        assert self._traj is not None
        if self._step_idx >= self._traj.shape[0]:
            print("[CuroboPolicy] HOME done → DONE.")
            self._phase = Phase.DONE
            assert self._hold_action is not None
            return self._hold_action

        action = self._traj[self._step_idx].to(device=device, dtype=torch.float32)
        self._step_idx += 1
        self._hold_action = action
        return action

    # ------------------------------------------------------------------
    # Trajectory helpers
    # ------------------------------------------------------------------

    def _build_home_traj(
        self, env: gym.Env, device: torch.device, motion: MotionClient
    ) -> None:
        """Cosine ease-in/out from current joints to home (all 6 DoF)."""
        if self._hold_action is not None:
            start = self._hold_action.detach().to(device=device, dtype=torch.float32)
        else:
            start = motion.joint_action(env, device)

        goal = torch.tensor(_HOME_JOINT_RAD, device=device, dtype=torch.float32)
        n = max(1, int(self.config.home_steps))
        t = torch.linspace(1.0 / n, 1.0, n, device=device, dtype=torch.float32)
        alpha = (1.0 - torch.cos(t * math.pi)) * 0.5
        self._traj = (start.unsqueeze(0) + alpha.unsqueeze(1) * (goal - start).unsqueeze(0)).contiguous()
        self._step_idx = 0
        print(f"[CuroboPolicy] HOME interpolating {n} steps to home joint pose.")

    def _playback_traj(
        self, device: torch.device, motion: MotionClient, *, jaw: float
    ) -> torch.Tensor | None:
        """Advance one waypoint, or None when the trajectory is finished."""
        assert self._traj is not None
        if self._step_idx >= self._traj.shape[0]:
            return None
        q = self._traj[self._step_idx]
        self._step_idx += 1
        motion.notify_joint(q)
        action = motion.q_to_action(q, device, jaw=jaw)
        self._hold_action = action
        return action

    def _hold_with_jaw(self, device: torch.device, jaw: float) -> torch.Tensor:
        assert self._hold_action is not None
        action = self._hold_action.clone()
        action[_JAW_INDEX] = float(jaw)
        self._hold_action = action
        return action

    def _jaw_is_fully_closed(self, jaw: float | torch.Tensor) -> bool:
        return float(jaw) <= float(self.config.jaw_closed) + float(self.config.jaw_empty_tol)

    # ------------------------------------------------------------------
    # Motion client + debug wiring
    # ------------------------------------------------------------------

    def _ensure_motion(self, device: torch.device) -> MotionClient:
        if self._motion is not None:
            return self._motion

        cfg = self.config
        if cfg.debug_viser:
            debug = ViserCollisionDebugViz(
                resolve_robot_yml(cfg.robot_yml),
                port=int(cfg.debug_viser_port),
            )
        else:
            debug = NullCollisionDebugViz()

        self._motion = MotionClient(
            MotionClientCfg(
                robot_yml=cfg.robot_yml,
                position_tolerance=cfg.position_tolerance,
                orientation_tolerance=cfg.orientation_tolerance,
                use_cuda_graph=cfg.use_cuda_graph,
                self_collision_check=cfg.self_collision_check,
                waypoint_stride=cfg.waypoint_stride,
            ),
            debug=debug,
        )
        self._motion.ensure_ready(device)

        if cfg.debug_viz: # TODO: rename to debug_marker to avoid confiusions with viser visualization
            self._kit_markers = KitFrameMarkers(scale=cfg.marker_frame_scale)

        return self._motion

    def _update_kit_markers(
        self, env: gym.Env, device: torch.device, motion: MotionClient
    ) -> None:
        if self._kit_markers is None:
            return
        self._kit_markers.ensure(env, motion.tool_frame)
        goal_pos_w, goal_quat = self._goal_pose_w(env, device, motion)
        self._kit_markers.update(
            env, device, goal_pos_w=goal_pos_w, goal_quat_xyzw=goal_quat
        )

    # ------------------------------------------------------------------
    # Goal poses (task-level; still on the policy until PoseStrategy extract)
    # ------------------------------------------------------------------

    def _resolve_shapes(self, env: gym.Env) -> list[ShapeInfo]:
        if self._shapes is not None:
            return self._shapes

        cfg_shapes = getattr(env.unwrapped.cfg, "shapes", None)
        if cfg_shapes:
            self._shapes = list(cfg_shapes)
        elif self.config.goal_object:
            self._shapes = [ShapeInfo(prim_path=f"{{ENV_REGEX_NS}}/{self.config.goal_object}")]
        else:
            raise ValueError(
                "CuroboPolicy needs env.cfg.shapes (from ShapeSortingEnvironment) "
                "or --goal_object <scene_entity_name>."
            )
        if self._shape_idx >= len(self._shapes):
            self._shape_idx = 0
        return self._shapes

    def _current_shape(self, env: gym.Env) -> ShapeInfo:
        return self._resolve_shapes(env)[self._shape_idx]

    def _grasp_pose_in_robot_base(
        self, env: gym.Env, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Grasp EE pose: XY standoff on base→object line, fixed Z."""
        shape = self._current_shape(env)
        obj_b = entity_position_in_robot_base(env, shape.name, device=device)
        xy = obj_b[0:2]
        rho = torch.linalg.norm(xy)
        if float(rho) < 1e-6:
            raise RuntimeError(
                f"CuroboPolicy: shape '{shape.name}' is directly above "
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

    def _hole_frame_name_for_shape(self, shape: ShapeInfo) -> str:
        from shape_sorting.shape_asset import SortingBox
        from shape_sorting.shape_forms import ShapeForm

        prefix = "shape_piece_"
        if not shape.name.startswith(prefix):
            raise RuntimeError(
                f"CuroboPolicy: cannot map shape '{shape.name}' to a lid hole; "
                f"expected a name starting with '{prefix}'."
            )
        form_value = shape.name[len(prefix) :]
        try:
            form = ShapeForm(form_value)
        except ValueError as exc:
            raise RuntimeError(
                f"CuroboPolicy: unknown form '{form_value}' in shape '{shape.name}'."
            ) from exc
        return SortingBox.hole_frame_name(form)

    def _hole_position_in_robot_base(
        self, env: gym.Env, device: torch.device
    ) -> torch.Tensor:
        """Matched lid-hole origin in the robot base frame. Shape (3,)."""
        import warp as wp
        from isaaclab.utils.math import subtract_frame_transforms
        from shape_sorting.shape_asset import SortingBox

        shape = self._current_shape(env)
        frame_name = self._hole_frame_name_for_shape(shape)
        scene = env.unwrapped.scene
        try:
            holes = scene[SortingBox.HOLE_FRAMES_SENSOR_NAME]
        except KeyError as exc:
            available = sorted(scene.keys()) if hasattr(scene, "keys") else []
            raise KeyError(
                f"Hole frames sensor not in scene. Available: {available}"
            ) from exc

        names = list(holes.data.target_frame_names)
        try:
            idx = names.index(frame_name)
        except ValueError as exc:
            raise KeyError(
                f"Hole frame '{frame_name}' for shape '{shape.name}' "
                f"not in '{SortingBox.HOLE_FRAMES_SENSOR_NAME}'. Have: {names}"
            ) from exc

        robot = scene["robot"]
        robot_pose_w = wp.to_torch(robot.data.root_pose_w)[0].to(device=device, dtype=torch.float32)
        hole_pos_w = wp.to_torch(holes.data.target_pos_w)[0, idx].to(
            device=device, dtype=torch.float32
        )
        hole_quat_w = wp.to_torch(holes.data.target_quat_w)[0, idx].to(
            device=device, dtype=torch.float32
        )

        hole_b, _ = subtract_frame_transforms(
            robot_pose_w[0:3].unsqueeze(0),
            robot_pose_w[3:7].unsqueeze(0),
            hole_pos_w.unsqueeze(0),
            hole_quat_w.unsqueeze(0),
        )
        return hole_b[0]

    def _place_pose_in_robot_base(
        self, env: gym.Env, device: torch.device, motion: MotionClient
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Place EE so the grasped object origin sits above the lid hole."""
        from isaaclab.utils.math import quat_apply

        if motion.grasp_offset_ee is None:
            raise RuntimeError(
                "CuroboPolicy PLACE requires a grasp offset; ATTACH must run first."
            )

        hole_b = self._hole_position_in_robot_base(env, device)
        obj_desired = hole_b.clone()
        obj_desired[2] = obj_desired[2] + float(self.config.place_z_offset_m)

        # Seed EE orientation from hole XY, then back out EE position so
        # R @ grasp_offset places the object at obj_desired.
        _, quat_seed = so101_ee_pose_xyzw(
            float(obj_desired[0]),
            float(obj_desired[1]),
            float(obj_desired[2]),
            tilt=_GOAL_TILT_RAD,
            roll=_GOAL_ROLL_RAD,
            device=device,
        )
        offset_b = quat_apply(
            quat_seed.unsqueeze(0),
            motion.grasp_offset_ee.to(device=device, dtype=torch.float32).unsqueeze(0),
        )[0]
        ee_pos = obj_desired - offset_b

        return so101_ee_pose_xyzw(
            float(ee_pos[0]),
            float(ee_pos[1]),
            float(ee_pos[2]),
            tilt=_GOAL_TILT_RAD,
            roll=_GOAL_ROLL_RAD,
            device=device,
        )

    def _goal_pose_w(
        self, env: gym.Env, device: torch.device, motion: MotionClient
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """World-frame active goal for Kit markers."""
        import warp as wp
        from isaaclab.utils.math import combine_frame_transforms

        num_envs = env.unwrapped.num_envs
        if motion.last_goal_xyz is None or motion.last_goal_quat_xyzw is None:
            goal_b, goal_quat_xyzw = self._grasp_pose_in_robot_base(env, device)
        else:
            goal_b = motion.last_goal_xyz
            goal_quat_xyzw = motion.last_goal_quat_xyzw

        robot = env.unwrapped.scene["robot"]
        robot_pose_w = wp.to_torch(robot.data.root_pose_w).to(device=device, dtype=torch.float32)
        t = goal_b.unsqueeze(0).expand(num_envs, -1)
        q = goal_quat_xyzw.unsqueeze(0).expand(num_envs, -1)
        return combine_frame_transforms(robot_pose_w[:, 0:3], robot_pose_w[:, 3:7], t, q)
