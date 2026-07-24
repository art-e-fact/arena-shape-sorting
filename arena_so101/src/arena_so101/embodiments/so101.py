"""SO-101 follower embodiments for Isaac Lab Arena.

USD and joint names follow the NVIDIA Sim-to-Real SO-101 workshop
(``Rotation`` … ``Jaw``). Wrist RGB is a Python ``TiledCameraCfg`` on
``Robot/gripper/gripper_cam`` — not a baked Isaac Lab sensor in the USD.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import isaaclab.envs.mdp as mdp_isaac_lab
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
    RelativeJointPositionActionCfg,
)
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg, TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.utils.cameras import ArenaCameraCfg
from isaaclab_arena.utils.pose import Pose

from arena_so101.mapping import SIM_JOINT_NAMES

_DATA_DIR = Path(__file__).parent / "data"
_USD_PATH = str(_DATA_DIR / "SO-ARM101-USD.usd")

# Arm joints only (Jaw is a separate binary gripper term for IK).
_ARM_JOINT_NAMES = (
    "Rotation",
    "Pitch",
    "Elbow",
    "Wrist_Pitch",
    "Wrist_Roll",
)

# USD Jaw limits from the NVIDIA workshop (degrees → radians).
_JAW_OPEN_RAD = math.radians(100.0)
_JAW_CLOSE_RAD = math.radians(-10.0)

def _quat_xyzw_from_euler_deg(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Intrinsic XYZ Euler (degrees) → quaternion (x, y, z, w)."""
    quat = quat_from_euler_xyz(
        torch.tensor(math.radians(roll)),
        torch.tensor(math.radians(pitch)),
        torch.tensor(math.radians(yaw)),
    )
    return tuple(quat.tolist())


# 90° yaw about Z (x, y, z, w).
_YAW_90 = _quat_xyzw_from_euler_deg(0.0, 0.0, 90.0)

_SO101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        rot=_YAW_90,
        joint_pos={
            "Rotation": -0.2736,
            "Pitch": -0.6109,
            "Elbow": -0.0745,
            "Wrist_Pitch": 1.5148,
            "Wrist_Roll": -1.6034,
            "Jaw": -0.1465,
        },
    ),
    # Gear-aware gains from the NVIDIA workshop.
    actuators={
        "rotation": ImplicitActuatorCfg(
            joint_names_expr=["Rotation"], effort_limit_sim=30, stiffness=55, damping=0.7
        ),
        "pitch": ImplicitActuatorCfg(
            joint_names_expr=["Pitch"], effort_limit_sim=30, stiffness=30, damping=0.8
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["Elbow"], effort_limit_sim=30, stiffness=25, damping=0.7
        ),
        "wrist_pitch": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Pitch"], effort_limit_sim=30, stiffness=12, damping=0.5
        ),
        "wrist_roll": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Roll"], effort_limit_sim=30, stiffness=7, damping=0.5
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["Jaw"], effort_limit_sim=30, stiffness=4, damping=0.3
        ),
    },
)

# High-PD / gravity-off copy for differential IK (same idea as FRANKA_PANDA_HIGH_PD_CFG).
# Diff-IK assumes joint targets are tracked tightly; gravity + soft PD would make EE
# lag the SE(3) command. Gravity off removes that disturbance; high PD tracks targets.
_SO101_IK_CFG = _SO101_CFG.copy()
_SO101_IK_CFG.spawn.rigid_props.disable_gravity = True
_SO101_IK_CFG.actuators = {
    "rotation": ImplicitActuatorCfg(
        joint_names_expr=["Rotation"], effort_limit_sim=30, stiffness=400, damping=80
    ),
    "pitch": ImplicitActuatorCfg(
        joint_names_expr=["Pitch"], effort_limit_sim=30, stiffness=400, damping=80
    ),
    "elbow": ImplicitActuatorCfg(
        joint_names_expr=["Elbow"], effort_limit_sim=30, stiffness=400, damping=80
    ),
    "wrist_pitch": ImplicitActuatorCfg(
        joint_names_expr=["Wrist_Pitch"], effort_limit_sim=30, stiffness=200, damping=40
    ),
    "wrist_roll": ImplicitActuatorCfg(
        joint_names_expr=["Wrist_Roll"], effort_limit_sim=30, stiffness=200, damping=40
    ),
    "gripper": ImplicitActuatorCfg(
        joint_names_expr=["Jaw"], effort_limit_sim=30, stiffness=20, damping=2
    ),
}


@configclass
class SO101SceneCfg:
    robot: ArticulationCfg = _SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # EE frame for reach/place rewards (same target as workshop).
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/gripper",
                name="gripper",
            ),
        ],
    )


@configclass
class SO101AbsJointActionsCfg:
    arm_action: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(SIM_JOINT_NAMES),
        scale=1,
        use_default_offset=False,
    )


@configclass
class SO101RelJointActionsCfg:
    arm_action: ActionTermCfg = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(SIM_JOINT_NAMES),
        scale=0.05,
        use_zero_offset=True,
    )


@configclass
class SO101IKActionsCfg:
    """Relative SE(3) arm + binary Jaw — matches keyboard / gamepad / spacemouse teleop."""

    arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=list(_ARM_JOINT_NAMES),
        body_name="gripper",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.0]),
    )

    gripper_action: ActionTermCfg = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["Jaw"],
        open_command_expr={"Jaw": _JAW_OPEN_RAD},
        close_command_expr={"Jaw": _JAW_CLOSE_RAD},
    )


@configclass
class SO101ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp_isaac_lab.last_action)
        joint_pos = ObsTerm(func=mdp_isaac_lab.joint_pos, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp_isaac_lab.joint_vel, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class SO101CameraCfg(ArenaCameraCfg):
    # Workshop ego cam: spawn at gripper mount, offset into the real lens frame.
    camera_ego: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/gripper_cam",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            projection_type="pinhole",
            f_stop=100.0,
            focal_length=13.5,
            focus_distance=0.05,
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.005, 0.06, -0.062),
            rot=_quat_xyzw_from_euler_deg(-45.0, 0.0, 0.0),
            convention="opengl",
        ),
    )


class SO101EmbodimentBase(EmbodimentBase):
    """Shared SO-101 follower setup (workshop USD)."""

    default_arm_mode = ArmMode.SINGLE_ARM

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose: Pose | None = None,
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
    ):
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms, arm_mode)
        self.scene_config = SO101SceneCfg()
        self.camera_config = SO101CameraCfg()
        self.observation_config = SO101ObservationsCfg()
        if concatenate_observation_terms:
            self.observation_config.policy.concatenate_terms = True
        self.action_config = None

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        return "gripper"

    def get_command_body_name(self) -> str:
        return "gripper"

    def get_reach_body_name(self) -> str:
        """Rigid body used for reach rewards (workshop USD link name)."""
        return "gripper"


@register_asset
class SO101AbsJointEmbodiment(SO101EmbodimentBase):
    """Absolute joint positions — preferred for SO-101 leader / joint gamepad teleop."""

    name = "so101_abs_joint"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_config = SO101AbsJointActionsCfg()


@register_asset
class SO101RelJointEmbodiment(SO101EmbodimentBase):
    """Relative joint position actions."""

    name = "so101_rel_joint"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_config = SO101RelJointActionsCfg()


@register_asset
class SO101IKEmbodiment(SO101EmbodimentBase):
    """Differential IK (relative SE(3)) arm control + binary Jaw gripper."""

    name = "so101_ik"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scene_config.robot = _SO101_IK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.action_config = SO101IKActionsCfg()

    def get_command_body_name(self) -> str:
        return self.action_config.arm_action.body_name
