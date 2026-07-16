
# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

from shape_sorting.shape_forms import DEFAULT_FORMS, ShapeForm

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@dataclass
class ShapeSortingEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the shape-sorting pick-and-place environment."""

    enable_cameras: bool = False
    embodiment: str = "droid_rel_joint_pos"
    hdr: str | None = None
    light_intensity: float = 500.0
    additional_table_objects: list[str] = field(default_factory=list)
    rl_training_mode: bool = False
    forms: list[ShapeForm] = field(default_factory=lambda: list(DEFAULT_FORMS))
    piece_size: float = 0.05
    piece_height: float = 0.02
    box_height: float = 0.06
    clearance: float = 0.003


@register_environment
class ShapeSortingEnvironment(ArenaEnvironmentFactory[ShapeSortingEnvironmentCfg]):
    """Registered provider for the procedural shape-sorting environment."""

    name: str = "shape_sorting_test"
    _legacy_argparse_cfg_type = ShapeSortingEnvironmentCfg

    def build(self, cfg: ShapeSortingEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        import isaaclab_arena_examples.policy.base_rsl_rl_policy as base_rsl_rl_policy
        from isaaclab.envs.common import ViewerCfg

        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import IsAnchor, On
        from isaaclab_arena.scene.scene import Scene

        from shape_sorting.pick_and_place_task import PickAndPlaceTaskRL
        from shape_sorting.shape_asset import make_shape_sorting_layout

        # Step 1: Retrieve assets from the registry / build the sorting layout.
        background = self.asset_registry.get_asset_by_name("maple_table_robolab")()
        layout = make_shape_sorting_layout(
            forms=cfg.forms,
            piece_size=cfg.piece_size,
            piece_height=cfg.piece_height,
            box_height=cfg.box_height,
            clearance=cfg.clearance,
        )
        pick_up_object = layout.pick_up_piece
        destination_location = layout.box

        # Step 2: Describe spatial relationships.
        table_reference = ObjectReference(
            name="table",
            prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
            parent_asset=background,
            object_type=ObjectType.RIGID,
        )
        table_reference.add_relation(IsAnchor())

        for asset in layout.assets():
            asset.add_relation(On(table_reference))

        additional_table_objects = [
            self.asset_registry.get_asset_by_name(name)() for name in cfg.additional_table_objects
        ]
        for obj in additional_table_objects:
            obj.add_relation(On(table_reference))

        # Step 3: Configure lighting.
        light = self.asset_registry.get_asset_by_name("light")()
        light.set_intensity(cfg.light_intensity)
        if cfg.hdr is not None:
            light.add_hdr(self.hdr_registry.get_hdr_by_name(cfg.hdr)())
        directional_light = self.asset_registry.get_asset_by_name("directional_light")()

        # Step 4: Select the embodiment (flat obs vector for RL policy input).
        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(
            enable_cameras=cfg.enable_cameras,
            concatenate_observation_terms=True,
        )
        # Droid does not wire concatenate_observation_terms into its obs cfg yet.
        embodiment.observation_config.policy.concatenate_terms = True

        # Step 5: Compose the scene.
        scene = Scene(
            assets=[
                background,
                light,
                directional_light,
                table_reference,
                *layout.assets(),
                *additional_table_objects,
            ]
        )

        # Step 6: Define the RL task (dense rewards + placement success).
        task = PickAndPlaceTaskRL(
            pick_up_object=pick_up_object,
            destination_location=destination_location,
            background_scene=background,
            embodiment=embodiment,
            episode_length_s=20.0,
            rl_training_mode=cfg.rl_training_mode,
        )

        # Set viewport camera to match the robolab droid view.
        def _set_viewer_cfg(env_cfg):
            env_cfg.viewer = ViewerCfg(eye=(1.5, 0.0, 1.0), lookat=(0.2, 0.0, 0.0))
            return env_cfg

        # Step 7: Assemble the environment.
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            env_cfg_callback=_set_viewer_cfg,
            rl_framework_entry_point="rsl_rl_cfg_entry_point",
            rl_policy_cfg=f"{base_rsl_rl_policy.__name__}:RLPolicyCfg",
        )
        return isaaclab_arena_environment

    # TODO(cvolk, 2026-07-03): [typed-config-migration] Delete this CLI-only option when teleoperation runners
    # receive typed configuration instead of the environment subparser namespace.
    @staticmethod
    def _add_legacy_cli_only_args(parser: argparse.ArgumentParser) -> None:
        # Consumed directly by teleop.py and record_demos.py, not by build(cfg).
        parser.add_argument("--teleop_device", type=str, default=None)
