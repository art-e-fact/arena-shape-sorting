
# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

from shape_sorting.shape_forms import DEFAULT_EDGE_CHAMFER, DEFAULT_FORMS, DEFAULT_HOLE_CHAMFER, ShapeForm

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@dataclass
class ShapeSortingEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the shape-sorting pick-and-place environment."""

    enable_cameras: bool = False
    embodiment: str = "droid_rel_joint_pos"
    teleop_device: str | None = None
    leader_port: str = "/dev/ttyACM0"
    leader_id: str = "leader"
    leader_recalibrate: bool = False
    hdr: str | None = None
    light_intensity: float = 500.0
    additional_table_objects: list[str] = field(default_factory=list)
    rl_training_mode: bool = False
    forms: list[ShapeForm] = field(default_factory=lambda: list(DEFAULT_FORMS))
    piece_size: float = 0.03
    piece_height: float = 0.03
    box_height: float = 0.04
    clearance: float = 0.003
    edge_chamfer: float = DEFAULT_EDGE_CHAMFER
    hole_chamfer: float = DEFAULT_HOLE_CHAMFER


class ShapeSortingEnvironment(ArenaEnvironmentFactory[ShapeSortingEnvironmentCfg]):
    """Registered provider for the procedural shape-sorting environment."""

    name: str = "shape_sorting_test"
    _legacy_argparse_cfg_type = ShapeSortingEnvironmentCfg

    def build(self, cfg: ShapeSortingEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        from isaaclab.envs.common import ViewerCfg

        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import IsAnchor, On, PositionLimits
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.sorting_task import SortMultiObjectTask
        from isaaclab_arena.utils.pose import Pose

        from shape_sorting.shape_asset import make_shape_sorting_layout

        # SO-101 embodiments / devices (and Arena gamepad) live in arena_so101.
        if (
            cfg.embodiment.startswith("so101")
            or (cfg.teleop_device or "").startswith("so101")
            or cfg.teleop_device == "gamepad"
        ):
            import arena_so101

            if not hasattr(arena_so101, "register"):
                raise ImportError(
                    "import arena_so101 did not load the package (got a shadowed path). "
                    "Install with: /isaac-sim/python.sh -m pip install -e so101_arena"
                )
            arena_so101.register()

        # Retrieve assets from the registry / build the sorting layout.
        background = self.asset_registry.get_asset_by_name("maple_table_robolab")()
        layout = make_shape_sorting_layout(
            forms=cfg.forms,
            piece_size=cfg.piece_size,
            piece_height=cfg.piece_height,
            box_height=cfg.box_height,
            clearance=cfg.clearance,
            edge_chamfer=cfg.edge_chamfer,
            hole_chamfer=cfg.hole_chamfer,
        )
        pieces = layout.pieces
        box = layout.box

        # Describe spatial relationships.
        table_reference = ObjectReference(
            name="table",
            prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
            parent_asset=background,
            object_type=ObjectType.RIGID,
        )
        table_reference.add_relation(IsAnchor())

        layout.box.add_relation(On(table_reference))
        layout.box.add_relation(PositionLimits(x_min=0.5, x_max=0.55, y_min=-0.101, y_max=-0.05))

        for asset in layout.pieces:
            asset.add_relation(On(table_reference))
            asset.add_relation(PositionLimits(x_min=0.4, x_max=0.52, y_min=0.01, y_max=0.28))

        additional_table_objects = [
            self.asset_registry.get_asset_by_name(name)() for name in cfg.additional_table_objects
        ]
        for obj in additional_table_objects:
            obj.add_relation(On(table_reference))

        # Configure lighting.
        light = self.asset_registry.get_asset_by_name("light")()
        light.set_intensity(cfg.light_intensity)
        if cfg.hdr is not None:
            light.add_hdr(self.hdr_registry.get_hdr_by_name(cfg.hdr)())
        directional_light = self.asset_registry.get_asset_by_name("directional_light")()

        # Select the embodiment (flat obs vector for RL policy input).
        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(
            enable_cameras=cfg.enable_cameras,
            concatenate_observation_terms=True,
        )

        # Set the initial pose for the SO-101 to sit on the side of the table.
        if "so101" in cfg.embodiment:
            embodiment.set_initial_pose(
                Pose(
                    position_xyz=(0.236, 0.0, -0.027),
                    rotation_xyzw=embodiment.scene_config.robot.init_state.rot,
                )
            )

        # Droid does not wire concatenate_observation_terms into its obs cfg yet.
        embodiment.observation_config.policy.concatenate_terms = True

        teleop_device = None
        if cfg.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)()
            if cfg.teleop_device == "so101_leader":
                teleop_device.port = cfg.leader_port
                teleop_device.leader_id = cfg.leader_id
                teleop_device.leader_recalibrate = cfg.leader_recalibrate

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

        # Place-all task: every piece must contact the shared sorting box.
        task = SortMultiObjectTask(
            pick_up_object_list=pieces,
            destination_location_list=[box] * len(pieces),
            background_scene=background,
            episode_length_s=120.0,
        )
        task.termination_cfg.success.params["force_threshold"] = 0.1

        # Set viewport camera to match the robolab droid view.
        def _set_viewer_cfg(env_cfg):
            env_cfg.viewer = ViewerCfg(eye=(1.5, 0.0, 1.0), lookat=(0.2, 0.0, 0.0))
            return env_cfg

        # Assemble the environment.
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            env_cfg_callback=_set_viewer_cfg,
        )
        return isaaclab_arena_environment

