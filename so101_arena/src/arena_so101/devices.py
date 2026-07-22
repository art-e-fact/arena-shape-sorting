"""Teleop device registrations for SO-101 (leader + SE3 gamepad).

Arena's built-in device library has keyboard / spacemouse / openxr but not
gamepad. We register ``gamepad`` here so ``@register_retargeter`` pairs with
``so101_ik`` resolve through ArenaEnvBuilder.

``so101_leader`` returns an Isaac Lab ``DeviceCfg`` that emits absolute joint
targets for ``so101_abs_joint`` (not SE3).
"""

from __future__ import annotations

from collections.abc import Callable

from isaaclab.devices import Se3GamepadCfg

from isaaclab_arena.assets.device_library import TeleopDeviceBase
from isaaclab_arena.assets.register import register_device

from arena_so101.leader_device import SO101LeaderDeviceCfg


@register_device
class GamepadCfg(TeleopDeviceBase):
    """Registered as ``gamepad`` — SE(3) + gripper, same contract as keyboard."""

    name = "gamepad"

    def __init__(
        self,
        sim_device: str | None = None,
        pos_sensitivity: float = 0.1,
        rot_sensitivity: float = 0.1,
    ):
        super().__init__(sim_device=sim_device)
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

    def get_device_cfg(
        self, pipeline_builder: Callable | None = None, embodiment: object | None = None
    ) -> Se3GamepadCfg:
        return Se3GamepadCfg(
            pos_sensitivity=self.pos_sensitivity,
            rot_sensitivity=self.rot_sensitivity,
        )


@register_device
class SO101LeaderCfg(TeleopDeviceBase):
    """Registered as ``so101_leader`` — absolute joints via LeRobot."""

    name = "so101_leader"

    def __init__(self, sim_device: str | None = None, port: str = "/dev/ttyACM0", leader_id: str = "leader"):
        super().__init__(sim_device=sim_device)
        self.port = port
        self.leader_id = leader_id

    def get_device_cfg(
        self, pipeline_builder: Callable | None = None, embodiment: object | None = None
    ) -> SO101LeaderDeviceCfg:
        return SO101LeaderDeviceCfg(
            port=self.port,
            leader_id=self.leader_id,
            sim_device=self.sim_device or "cpu",
        )
