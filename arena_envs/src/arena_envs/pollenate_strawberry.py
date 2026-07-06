"""Pollenate strawberry plant environment.

Scene: ground, light, and a strawberry plant. Task success = EE reaches every
``approach_frame_*`` site discovered in the plant USD.
"""

import argparse
from pathlib import Path

from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.register import register_asset, register_environment
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

from arena_envs.pollenateable import Pollenateable

_REPO_ROOT = Path(__file__).parent.parent.parent.parent


# @register_asset
class StrawberryPlant(LibraryObject, Pollenateable):
    """Strawberry plant with pollinate approach-frame sites in its USD."""

    name = "strawberry_plant"
    tags = ["object", "pollenateable"]
    usd_path = str(_REPO_ROOT / "assets" / "strawberry_raw.usda")
    object_type = ObjectType.BASE


# @register_environment
class PollenateStrawberryEnvironment(ExampleEnvironmentBase):

    name: str = "pollenate_strawberry"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene

        from arena_envs.pollenate_task import PollenateTask

        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
        light = self.asset_registry.get_asset_by_name("light")()
        embodiment = self.asset_registry.get_asset_by_name("franka_ik")()

        plant = StrawberryPlant(
            initial_pose=Pose(position_xyz=(0.4, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
            reach_distance_threshold=args_cli.reach_distance_threshold,
        )
        scene = Scene(assets=[ground_plane, light, plant])

        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=PollenateTask(
                pollinate_object=plant,
                robot_name=embodiment.get_embodiment_name_in_scene(),
                episode_length_s=args_cli.episode_length_s,
            ),
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--reach_distance_threshold",
            type=float,
            default=0.06,
            help="EE-to-frame distance (m) under which an approach frame counts as reached.",
        )
        parser.add_argument(
            "--episode_length_s",
            type=float,
            default=30.0,
            help="Episode length in seconds.",
        )
