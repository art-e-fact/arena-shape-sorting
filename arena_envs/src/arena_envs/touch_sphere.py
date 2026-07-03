"""Minimal "touch a floating target" training environment.

Follows the built-in Arena env convention (e.g.
``isaaclab_arena_environments.press_button_rl_environment``): the module top-level is
import-light and only registers the environment; all heavy Isaac Lab / Arena imports
(including the ``Touchable`` affordance, ``TouchSphere`` asset and ``TouchTaskRL`` task,
which live in their own modules) happen inside ``get_env`` — i.e. after the sim app is up.
"""

import argparse

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase


# @register_environment
class TouchSphereEnvironment(ExampleEnvironmentBase):

    name: str = "touch_sphere"

    def get_env(self, args_cli: argparse.Namespace):
        import isaaclab_arena_examples.policy.base_rsl_rl_policy as base_rsl_rl_policy
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene

        from arena_envs.touch_assets import make_touch_spheres
        from arena_envs.touch_task import TouchTaskRL

        # Minimal scene: ground to stand on, a light, and the floating touch targets.
        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
        light = self.asset_registry.get_asset_by_name("light")()
        # concatenate_observation_terms=True gives the RL policy a flat observation vector.
        embodiment = self.asset_registry.get_asset_by_name("franka_ik")(concatenate_observation_terms=True)

        center = (args_cli.sphere_x, args_cli.sphere_y, args_cli.sphere_z)
        half = args_cli.scatter_half_extent
        half_extent = (half, half, half)

        # N floating kinematic spheres; the task scatters them with min-separation on reset.
        touch_spheres = make_touch_spheres(
            num_spheres=args_cli.num_spheres,
            region_center=center,
            region_half_extent=half_extent,
            touch_force_threshold=args_cli.touch_force_threshold,
        )
        # Task owns per-episode placement; disable each asset's default fixed-pose reset event.
        for sphere in touch_spheres:
            sphere.disable_reset_pose()

        scene = Scene(assets=[ground_plane, light, *touch_spheres])

        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=TouchTaskRL(
                touch_objects=touch_spheres,
                embodiment=embodiment,
                rl_training_mode=args_cli.rl_training_mode,
                scatter_region_center=center,
                scatter_region_half_extent=half_extent,
                min_separation=args_cli.min_separation,
            ),
            rl_framework_entry_point="rsl_rl_cfg_entry_point",
            rl_policy_cfg=f"{base_rsl_rl_policy.__name__}:RLPolicyCfg",
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--rl_training_mode", action="store_true")
        parser.add_argument("--sphere_x", type=float, default=0.5)
        parser.add_argument("--sphere_y", type=float, default=0.0)
        parser.add_argument("--sphere_z", type=float, default=0.3)
        parser.add_argument("--touch_force_threshold", type=float, default=1.0)
        parser.add_argument("--num_spheres", type=int, default=3,
                            help="Number of touch targets to spawn.")
        parser.add_argument("--scatter_half_extent", type=float, default=0.15,
                            help="Half-width (m) of the shared spawn box in each axis, centered on sphere_x/y/z.")
        parser.add_argument("--min_separation", type=float, default=0.12,
                            help="Minimum distance (m) between spheres when scattered (keep > 2*sphere_radius).")