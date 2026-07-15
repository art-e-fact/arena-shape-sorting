"""Procedural shape-sorting manipulands."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg

import torch

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectBase, ObjectType
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose

from shape_sorting.shape_mesh import ProceduralMeshCfg, box_mesh_data, bounding_box_from_mesh_data


def make_shape_piece_spawn_cfg(
    size: tuple[float, float, float] = (0.05, 0.05, 0.05),
) -> ProceduralMeshCfg:
    """Physics/visual spawn settings for a small rigid shape piece."""
    return ProceduralMeshCfg(
        size=size,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.35, 0.15)),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=0,
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_approximation="boundingCube",
        activate_contact_sensors=True,
    )


@register_asset
class ShapePiece(Object):
    """Rigid shape piece spawned from procedurally authored USD mesh data."""

    name = "shape_piece"
    tags = ["object", "procedural"]

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        size: tuple[float, float, float] = (0.05, 0.05, 0.05),
    ):
        self.size = size
        resolved_name = instance_name if instance_name is not None else "shape_piece"
        resolved_prim = prim_path if prim_path is not None else "{ENV_REGEX_NS}/ShapePiece"
        super().__init__(
            name=resolved_name,
            prim_path=resolved_prim,
            tags=self.tags,
            object_type=ObjectType.RIGID,
            usd_path="",
            initial_pose=initial_pose,
        )

    def _generate_rigid_cfg(self) -> RigidObjectCfg:
        cfg = RigidObjectCfg(
            prim_path=self.prim_path,
            spawn=make_shape_piece_spawn_cfg(self.size),
            **self.asset_cfg_addon,
        )
        return self._add_initial_pose_to_cfg(cfg)

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Local bounds from procedural mesh vertices (no backing USD file)."""
        if self.bounding_box is None:
            self.bounding_box = bounding_box_from_mesh_data(box_mesh_data(self.size))
        return self.bounding_box

    def get_corners(self, pos: torch.Tensor) -> torch.Tensor:
        return self.get_bounding_box().get_corners_at(pos)

    def get_contact_sensor_cfg(
        self, contact_against_object: ObjectBase | None = None, usd_path: str | None = None
    ) -> ContactSensorCfg:
        """Rigid body lives on the spawned root Xform; there is no backing USD file."""
        return ObjectBase.get_contact_sensor_cfg(self, contact_against_object)
