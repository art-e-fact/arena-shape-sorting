"""SO-101 leader as an Arena teleop device name.

Arena's built-in teleop path expects SE3 devices. The physical SO-101 leader
outputs joint positions via LeRobot, so use ``python -m arena_so101.teleop_leader``
for the real control loop. This registration keeps the device discoverable by name.
"""

from __future__ import annotations

from collections.abc import Callable

from isaaclab_arena.assets.device_library import TeleopDeviceBase
from isaaclab_arena.assets.register import register_device


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
