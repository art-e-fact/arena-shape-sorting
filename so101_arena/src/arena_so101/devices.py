"""SO-101 teleop device registrations for Arena."""

from __future__ import annotations

from collections.abc import Callable

from isaaclab.devices import Se3GamepadCfg
from isaaclab_arena.assets.device_library import TeleopDeviceBase
from isaaclab_arena.assets.register import register_device


@register_device
class GamepadCfg(TeleopDeviceBase):
    """SE(3) gamepad teleop (Arena device_library has no gamepad entry)."""

    name = "gamepad"

    def __init__(self, sim_device: str | None = None, pos_sensitivity: float = 0.1, rot_sensitivity: float = 0.1):
        super().__init__(sim_device=sim_device)
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity

    def get_device_cfg(self, pipeline_builder: Callable | None = None, embodiment: object | None = None) -> Se3GamepadCfg:
        return Se3GamepadCfg(
            pos_sensitivity=self.pos_sensitivity,
            rot_sensitivity=self.rot_sensitivity,
        )


@register_device
class SO101LeaderCfg(TeleopDeviceBase):
    """Registered as ``so101_leader``. Drive sim with ``arena_so101.teleop_leader``."""

    name = "so101_leader"

    def __init__(self, sim_device: str | None = None, port: str = "/dev/ttyACM0", leader_id: str = "leader"):
        super().__init__(sim_device=sim_device)
        self.port = port
        self.leader_id = leader_id

    def get_device_cfg(self, pipeline_builder: Callable | None = None, embodiment: object | None = None):
        raise RuntimeError(
            "so101_leader is joint-space (LeRobot), not an SE3 Arena device. "
            "Use: python -m arena_so101.teleop_leader --port ... "
            f"(port={self.port!r}, id={self.leader_id!r})"
        )
