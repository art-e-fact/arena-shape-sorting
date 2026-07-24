"""Teleop device registrations for SO-101 (leader + gamepad).

Arena's built-in device library has keyboard / spacemouse / openxr but not
gamepad. We register ``gamepad`` here so ``@register_retargeter`` pairs with
``so101_ik`` / ``so101_abs_joint`` resolve through ArenaEnvBuilder.

``so101_leader`` returns an Isaac Lab ``DeviceCfg`` that emits absolute joint
targets for ``so101_abs_joint`` (not SE3).
"""

from __future__ import annotations

from collections.abc import Callable

from isaaclab.devices import Se3GamepadCfg
from isaaclab.devices.device_base import DeviceCfg

from isaaclab_arena.assets.device_library import TeleopDeviceBase
from isaaclab_arena.assets.register import register_device

from arena_so101.joint_gamepad_device import SO101JointGamepadCfg
from arena_so101.leader_device import SO101LeaderDeviceCfg


@register_device
class GamepadCfg(TeleopDeviceBase):
    """Registered as ``gamepad``.

    Layout depends on the paired embodiment:
    - ``so101_abs_joint`` → absolute joint gamepad (:class:`SO101JointGamepadCfg`)
    - ``so101_ik`` → SE(3) gamepad (:class:`Se3GamepadCfg`)
    """

    name = "gamepad"

    def __init__(
        self,
        sim_device: str | None = None,
        pos_sensitivity: float = 0.1,
        rot_sensitivity: float = 0.1,
        delta_scale: float = 0.03,
    ):
        super().__init__(sim_device=sim_device)
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.delta_scale = delta_scale

    def get_device_cfg(
        self, pipeline_builder: Callable | None = None, embodiment: object | None = None
    ) -> DeviceCfg:
        emb_name = getattr(embodiment, "name", None)
        if emb_name == "so101_abs_joint":
            return SO101JointGamepadCfg(
                delta_scale=self.delta_scale,
                sim_device=self.sim_device or "cpu",
            )
        if emb_name == "so101_ik":
            return Se3GamepadCfg(
                pos_sensitivity=self.pos_sensitivity,
                rot_sensitivity=self.rot_sensitivity,
            )
        raise ValueError(
            f"Registered gamepad has no layout for embodiment {emb_name!r}. "
            "Use --embodiment so101_abs_joint (joint-space) or so101_ik (SE3)."
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
