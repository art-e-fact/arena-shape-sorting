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

from shape_sorting.shape_forms import (
    DEFAULT_EDGE_CHAMFER,
    DEFAULT_FORMS,
    DEFAULT_HOLE_CHAMFER,
    ShapeForm,
    sizes_for_equal_area,
)
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
    edge_chamfer: float = DEFAULT_EDGE_CHAMFER,
) -> ProceduralMeshCfg:
    """Physics/visual spawn settings for a small rigid shape piece."""
    return ProceduralMeshCfg(
        form=form,
        size=size,
        height=height,
        edge_chamfer=edge_chamfer,
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
        edge_chamfer: float = DEFAULT_EDGE_CHAMFER,
    ):
        self.form = form
        self.size = size
        self.height = height
        self.edge_chamfer = edge_chamfer
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
            spawn=make_shape_piece_spawn_cfg(
                self.form, self.size, self.height, self.color, edge_chamfer=self.edge_chamfer
            ),
            **self.asset_cfg_addon,
        )
        return self._add_initial_pose_to_cfg(cfg)

    def get_bounding_box(self) -> AxisAlignedBoundingBox:
        """Local bounds from procedural mesh vertices (no backing USD file)."""
        if self.bounding_box is None:
            self.bounding_box = bounding_box_from_mesh_data(
                form_mesh_data(self.form, self.size, self.height, edge_chamfer=self.edge_chamfer)
            )
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
    ``geometry/<part>/mesh``. Outer size follows a near-square hole grid over
    ``forms``, sized from ``form_sizes`` and ``clearance``.
    """

    name = "sorting_box"
    tags = ["object", "procedural", "shape_sorting", "destination"]
    PART_NAMES: tuple[str, ...] = ("bottom", "front", "back", "left", "right", "lid")
    HOLE_FRAMES_SENSOR_NAME: str = "hole_frames"
    """Default InteractiveScene key for :meth:`get_hole_frames_cfg`."""

    def __init__(
        self,
        forms: Sequence[ShapeForm] = DEFAULT_FORMS,
        form_sizes: Sequence[float] | None = None,
        piece_size: float = 0.05,
        box_height: float = 0.06,
        clearance: float = 0.003,
        wall_thickness: float = 0.008,
        lid_thickness: float = 0.006,
        bottom_thickness: float = 0.006,
        hole_gap: float = 0.012,
        hole_chamfer: float = DEFAULT_HOLE_CHAMFER,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
    ):
        self.forms = tuple(forms)
        self.piece_size = piece_size
        self.form_sizes = (
            tuple(form_sizes) if form_sizes is not None else sizes_for_equal_area(self.forms, piece_size)
        )
        if len(self.forms) != len(self.form_sizes):
            raise ValueError(
                f"forms and form_sizes length mismatch: {len(self.forms)} forms vs {len(self.form_sizes)} sizes."
            )
        self.box_height = box_height
        self.clearance = clearance
        self.wall_thickness = wall_thickness
        self.lid_thickness = lid_thickness
        self.bottom_thickness = bottom_thickness
        self.hole_gap = hole_gap
        self.hole_chamfer = hole_chamfer

        self._parts, self.hole_centers, self._cavity_aabb = build_sorting_box_parts(
            forms=self.forms,
            form_sizes=self.form_sizes,
            box_height=self.box_height,
            clearance=self.clearance,
            wall_thickness=self.wall_thickness,
            lid_thickness=self.lid_thickness,
            bottom_thickness=self.bottom_thickness,
            hole_gap=self.hole_gap,
            hole_chamfer=self.hole_chamfer,
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

    def get_inner_bounding_box(self) -> AxisAlignedBoundingBox:
        """Local AABB of the empty cavity (inside walls, above bottom, below lid)."""
        return self._cavity_aabb

    @property
    def lid_top_z(self) -> float:
        """Local Z of the lid top surface (insertion plane) in the box frame [m]."""
        return 0.5 * self.box_height

    @staticmethod
    def hole_frame_name(form: ShapeForm) -> str:
        """Scene frame name for the lid hole matching ``form``."""
        return f"hole_{form.value}"

    def get_hole_frames_cfg(
        self,
        *,
        debug_vis: bool = True,
        marker_scale: float = 0.02,
    ):
        """FrameTransformer tracking each lid hole as an offset on the box rigid body.

        Target frame names are :meth:`hole_frame_name` for each entry in
        :attr:`forms` (same order as :attr:`hole_centers`). Source and every
        target prim path are the box root — holes are not separate rigid bodies.
        """
        from isaaclab.markers.config import FRAME_MARKER_CFG
        from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg

        z = self.lid_top_z
        target_frames = [
            FrameTransformerCfg.FrameCfg(
                prim_path=self.prim_path,
                name=self.hole_frame_name(form),
                offset=OffsetCfg(pos=(float(hx), float(hy), float(z))),
            )
            for (hx, hy), form in zip(self.hole_centers, self.forms)
        ]
        cfg = FrameTransformerCfg(
            prim_path=self.prim_path,
            debug_vis=debug_vis,
            target_frames=target_frames,
        )
        if debug_vis:
            marker_cfg = FRAME_MARKER_CFG.copy()
            marker_cfg.prim_path = f"/Visuals/FrameTransformer/{self.name}_holes"
            marker_cfg.markers["frame"].scale = (marker_scale, marker_scale, marker_scale)
            cfg.visualizer_cfg = marker_cfg
        return cfg

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
    form_sizes: tuple[float, ...]
    piece_height: float
    box_height: float
    clearance: float
    edge_chamfer: float
    hole_chamfer: float
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
    edge_chamfer: float = DEFAULT_EDGE_CHAMFER,
    hole_chamfer: float = DEFAULT_HOLE_CHAMFER,
) -> ShapeSortingLayout:
    """Build a sorting box and matching pieces from shared sizing parameters.

    ``piece_size`` is the side length of an equal-area reference square. Each
    form's characteristic size is derived so all pieces share that plan area.
    The box is a near-square grid over the forms, sized from the largest
    derived size plus ``clearance``.
    """
    forms_t = tuple(forms)
    form_sizes = sizes_for_equal_area(forms_t, piece_size)
    box = SortingBox(
        forms=forms_t,
        form_sizes=form_sizes,
        piece_size=piece_size,
        box_height=box_height,
        clearance=clearance,
        wall_thickness=wall_thickness,
        lid_thickness=lid_thickness,
        bottom_thickness=bottom_thickness,
        hole_gap=hole_gap,
        hole_chamfer=hole_chamfer,
    )
    pieces = [
        ShapePiece(
            form=form,
            size=form_size,
            height=piece_height,
            instance_name=f"shape_piece_{form.value}",
            color=_PIECE_COLORS[i % len(_PIECE_COLORS)],
            edge_chamfer=edge_chamfer,
        )
        for i, (form, form_size) in enumerate(zip(forms_t, form_sizes))
    ]
    return ShapeSortingLayout(
        forms=forms_t,
        piece_size=piece_size,
        form_sizes=form_sizes,
        piece_height=piece_height,
        box_height=box_height,
        clearance=clearance,
        edge_chamfer=edge_chamfer,
        hole_chamfer=hole_chamfer,
        box=box,
        pieces=pieces,
    )
