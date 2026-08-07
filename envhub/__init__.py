"""LeRobot EnvHub packaging for this repo's Isaac Lab Arena environments.

Importing this package registers our environment types with LeRobot, so the stock
LeRobot CLI scripts can build them without any wrapper script:

    lerobot-eval --env.discover_packages_path=envhub --env.type=shape_sorting_arena ...

Nothing in this package may import ``isaaclab`` at module level: LeRobot imports
the package (and every submodule) while parsing CLI arguments, long before Isaac
Sim is launched.
"""

from .config import ShapeSortingArenaEnv
from .env import make_env

__all__ = ["ShapeSortingArenaEnv", "make_env"]
