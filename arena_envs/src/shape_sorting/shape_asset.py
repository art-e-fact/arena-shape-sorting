"""Procedural shape-sorting manipulands and sorter box."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg

import torch

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectBase, ObjectType
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose

from shape_sorting.shape_forms import DEFAULT_FORMS, ShapeForm
from shape_sorting.shape_mesh import (
    ProceduralAssemblyCfg,
    ProceduralMeshCfg,
    bounding_box_from_mesh_data,
    bounding_box_from_parts,
    build_sorting_box_parts,
    form_mesh_data,
)


def make_shape_piece_spawn_cfg(
    form: ShapeForm = ShapeForm.CUBE,
    size: float = 0.05,
    height: float = 0.05,
    color: tuple[float, float, float] = (0.85, 0.35, 0.15),
) -> ProceduralMeshCfg:
    """Physics/visual spawn settings for a small rigid shape piece."""
    return ProceduralMeshCfg(
        form=form,
        size=size,
        height=height,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.5),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=0,
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_approximation="sdf",
        activate_contact_sensors=True,
    )


def make_sorting_box_spawn_cfg(parts) -> ProceduralAssemblyCfg:
    """Physics/visual spawn settings for the multi-part sorting box."""
    return ProceduralAssemblyCfg(
        parts=parts,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.45, 0.30)),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.6),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=0,
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.35),
        activate_contact_sensors=True,
    )


_PIECE_COLORS: tuple[tuple[float, float, float], ...] = (
    (0.85, 0.35, 0.15),
    (0.20, 0.55, 0.85),
    (0.30, 0.70, 0.35),
    (0.80, 0.65, 0.15),
    (0.65, 0.30, 0.70),
    (0.90, 0.40, 0.55),
)


@register_asset
class ShapePiece(Object):
    """Rigid sorting piece spawned from a build123d form."""

    name = "shape_piece"
    tags = ["object", "procedural", "shape_sorting"]

    def __init__(
        self,
        form: ShapeForm = ShapeForm.CUBE,
        size: float = 0.05,
        height: float = 0.05,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
        color: tuple[float, float, float] | None = None,
    ):
        self.form = form
        self.size = size
        self.height = height
        self.color = color if color is not None else _PIECE_COLORS[0]
        resolved_name = instance_name if instance_name is not None else f"shape_piece_{form.value}"
        resolved_prim = prim_path if prim_path is not None else f"{{ENV_REGEX_NS}}/{resolved_name}"
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
            spawn=make_shape_piece_spawn_cfg(self.form, self.size, self.height, self.color),
            **self.asset_cfg_addon,
        )
        return self._add_initial_pose_to_cfg(cfg)

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Local bounds from procedural mesh vertices (no backing USD file)."""
        if self.bounding_box is None:
            self.bounding_box = bounding_box_from_mesh_data(form_mesh_data(self.form, self.size, self.height))
        return self.bounding_box

    def get_corners(self, pos: torch.Tensor) -> torch.Tensor:
        return self.get_bounding_box().get_corners_at(pos)

    def get_contact_sensor_cfg(
        self, contact_against_object: ObjectBase | None = None, usd_path: str | None = None
    ) -> ContactSensorCfg:
        """Rigid body lives on the spawned root Xform; there is no backing USD file."""
        return ObjectBase.get_contact_sensor_cfg(self, contact_against_object)


@register_asset
class SortingBox(Object):
    """Box that collects sorting pieces through form-matched lid holes.

    Side walls, bottom, and lid are authored as separate mesh prims under
    ``geometry/<part>/mesh`` so contact filters can target individual faces.
    Outer size is derived from ``forms``, ``piece_size``, and ``clearance``.
    """

    name = "sorting_box"
    tags = ["object", "procedural", "shape_sorting", "destination"]
    PART_NAMES: tuple[str, ...] = ("bottom", "front", "back", "left", "right", "lid")

    def __init__(
        self,
        forms: Sequence[ShapeForm] = DEFAULT_FORMS,
        piece_size: float = 0.05,
        box_height: float = 0.06,
        clearance: float = 0.003,
        wall_thickness: float = 0.008,
        lid_thickness: float = 0.006,
        bottom_thickness: float = 0.006,
        hole_gap: float = 0.012,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
    ):
        self.forms = tuple(forms)
        self.piece_size = piece_size
        self.box_height = box_height
        self.clearance = clearance
        self.wall_thickness = wall_thickness
        self.lid_thickness = lid_thickness
        self.bottom_thickness = bottom_thickness
        self.hole_gap = hole_gap

        self._parts, self.hole_centers = build_sorting_box_parts(
            forms=self.forms,
            piece_size=self.piece_size,
            box_height=self.box_height,
            clearance=self.clearance,
            wall_thickness=self.wall_thickness,
            lid_thickness=self.lid_thickness,
            bottom_thickness=self.bottom_thickness,
            hole_gap=self.hole_gap,
        )

        resolved_name = instance_name if instance_name is not None else "sorting_box"
        resolved_prim = prim_path if prim_path is not None else f"{{ENV_REGEX_NS}}/{resolved_name}"
        super().__init__(
            name=resolved_name,
            prim_path=resolved_prim,
            tags=self.tags,
            object_type=ObjectType.RIGID,
            usd_path="",
            initial_pose=initial_pose,
        )

    def part_prim_path(self, part_name: str) -> str:
        """Absolute prim path expression for a named box part (for contact filters)."""
        if part_name not in self.PART_NAMES:
            raise ValueError(f"Unknown sorting-box part '{part_name}'. Expected one of {self.PART_NAMES}.")
        return f"{self.prim_path}/geometry/{part_name}"

    def _generate_rigid_cfg(self) -> RigidObjectCfg:
        cfg = RigidObjectCfg(
            prim_path=self.prim_path,
            spawn=make_sorting_box_spawn_cfg(self._parts),
            **self.asset_cfg_addon,
        )
        return self._add_initial_pose_to_cfg(cfg)

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Local bounds from the union of all assembly parts."""
        if self.bounding_box is None:
            self.bounding_box = bounding_box_from_parts(self._parts)
        return self.bounding_box

    def get_corners(self, pos: torch.Tensor) -> torch.Tensor:
        return self.get_bounding_box().get_corners_at(pos)

    def get_contact_sensor_cfg(
        self, contact_against_object: ObjectBase | None = None, usd_path: str | None = None
    ) -> ContactSensorCfg:
        """Rigid body lives on the spawned root Xform; there is no backing USD file."""
        return ObjectBase.get_contact_sensor_cfg(self, contact_against_object)


@dataclass
class ShapeSortingLayout:
    """Box plus one piece per form, ready to drop into an Arena scene."""

    forms: tuple[ShapeForm, ...]
    piece_size: float
    piece_height: float
    box_height: float
    clearance: float
    box: SortingBox
    pieces: list[ShapePiece]

    @property
    def pick_up_piece(self) -> ShapePiece:
        """Default manipuland for single-object pick-and-place tasks."""
        return self.pieces[0]

    def assets(self) -> list[Object]:
        """All scene assets belonging to this layout."""
        return [self.box, *self.pieces]


def make_shape_sorting_layout(
    forms: Sequence[ShapeForm] = DEFAULT_FORMS,
    piece_size: float = 0.05,
    piece_height: float = 0.02,
    box_height: float = 0.06,
    clearance: float = 0.003,
    wall_thickness: float = 0.008,
    lid_thickness: float = 0.006,
    bottom_thickness: float = 0.006,
    hole_gap: float = 0.012,
) -> ShapeSortingLayout:
    """Build a sorting box and matching pieces from shared sizing parameters.

    The box derives its footprint from ``forms`` / ``piece_size`` / ``clearance``.
    Each piece uses the same ``piece_size`` and ``piece_height``, with a distinct
    color per form index.
    """
    forms_t = tuple(forms)
    box = SortingBox(
        forms=forms_t,
        piece_size=piece_size,
        box_height=box_height,
        clearance=clearance,
        wall_thickness=wall_thickness,
        lid_thickness=lid_thickness,
        bottom_thickness=bottom_thickness,
        hole_gap=hole_gap,
    )
    pieces = [
        ShapePiece(
            form=form,
            size=piece_size,
            height=piece_height,
            instance_name=f"shape_piece_{form.value}",
            color=_PIECE_COLORS[i % len(_PIECE_COLORS)],
        )
        for i, form in enumerate(forms_t)
    ]
    return ShapeSortingLayout(
        forms=forms_t,
        piece_size=piece_size,
        piece_height=piece_height,
        box_height=box_height,
        clearance=clearance,
        box=box,
        pieces=pieces,
    )
