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
        from isaaclab_arena.utils.pose import Pose

        from arena_envs.touch_assets import TouchSphere
        from arena_envs.touch_task import TouchTaskRL

        # Minimal scene: ground to stand on, a light, and the floating touch target.
        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
        light = self.asset_registry.get_asset_by_name("light")()
        # concatenate_observation_terms=True gives the RL policy a flat observation vector.
        embodiment = self.asset_registry.get_asset_by_name("franka_ik")(concatenate_observation_terms=True)

        # The target floats (kinematic) in front of the robot. The default position is a
        # reachable spot for the Franka; tune with --sphere_x/--sphere_y/--sphere_z.
        spawn_radius = args_cli.sphere_spawn_radius
        touch_sphere = TouchSphere(
            initial_pose=Pose(
                position_xyz=(args_cli.sphere_x, args_cli.sphere_y, args_cli.sphere_z),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            ),
            touch_force_threshold=args_cli.touch_force_threshold,
        )
        # When radius > 0 the task registers a per-env ball-randomisation event that
        # writes the root pose itself; disable the default fixed-pose reset event so
        # the two don't fight each other.
        if spawn_radius > 0.0:
            touch_sphere.disable_reset_pose()

        scene = Scene(assets=[ground_plane, light, touch_sphere])

        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=TouchTaskRL(
                touch_object=touch_sphere,
                embodiment=embodiment,
                rl_training_mode=args_cli.rl_training_mode,
                sphere_spawn_radius=spawn_radius,
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
        parser.add_argument(
            "--sphere_spawn_radius",
            type=float,
            default=0.2,
            help="Uniform-in-ball radius (m) around sphere_x/y/z for per-env spawn randomisation. 0 = fixed pose.",
        )