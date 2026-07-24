"""Procedural mesh spawning via the OpenUSD Python API.

Builds triangle meshes at spawn time and wires them into Isaac Lab's rigid-body
hierarchy (``Xform`` root + ``geometry/...`` children). Used by
:class:`shape_sorting.shape_asset.ShapePiece` and
:class:`shape_sorting.shape_asset.SortingBox`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import MISSING, dataclass
from typing import TYPE_CHECKING

from isaaclab.sim import schemas
from isaaclab.sim.spawners import materials
from isaaclab.sim.spawners.meshes.meshes_cfg import MeshCfg
from isaaclab.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from isaaclab.sim.utils import bind_physics_material, bind_visual_material, clone, create_prim, get_current_stage
from isaaclab.utils import configclass

from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

from shape_sorting.shape_forms import (
    DEFAULT_EDGE_CHAMFER,
    DEFAULT_HOLE_CHAMFER,
    ShapeForm,
    hole_cutter,
    piece_solid,
    tessellate_solid,
    tessellation_tolerance,
)

if TYPE_CHECKING:
    from pxr import Usd


@dataclass
class MeshData:
    """Triangle mesh in local coordinates."""

    vertices: tuple[tuple[float, float, float], ...]
    face_vertex_counts: tuple[int, ...]
    face_vertex_indices: tuple[int, ...]

    def __repr__(self) -> str:
        n_faces = len(self.face_vertex_counts)
        n_indices = len(self.face_vertex_indices)
        return f"MeshData(n_vertices={len(self.vertices)}, n_faces={n_faces}, n_indices={n_indices})"


@dataclass
class MeshPart:
    """Named mesh under a rigid assembly, with its own collision approximation."""

    name: str
    mesh: MeshData
    collision_approximation: str = "convexHull"

    def __repr__(self) -> str:
        return f"MeshPart(name={self.name!r}, mesh={self.mesh!r}, collision_approximation={self.collision_approximation!r})"


def bounding_box_from_mesh_data(mesh: MeshData) -> AxisAlignedBoundingBox:
    """Axis-aligned bounds of mesh vertices in local coordinates."""
    xs, ys, zs = zip(*mesh.vertices)
    return AxisAlignedBoundingBox(
        min_point=(min(xs), min(ys), min(zs)),
        max_point=(max(xs), max(ys), max(zs)),
    )


def bounding_box_from_parts(parts: Sequence[MeshPart]) -> AxisAlignedBoundingBox:
    """Union AABB of every part mesh."""
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for part in parts:
        for x, y, z in part.mesh.vertices:
            xs.append(x)
            ys.append(y)
            zs.append(z)
    return AxisAlignedBoundingBox(
        min_point=(min(xs), min(ys), min(zs)),
        max_point=(max(xs), max(ys), max(zs)),
    )


def mesh_data_from_solid(solid, tolerance: float = 1e-3) -> MeshData:
    """Convert a build123d solid into :class:`MeshData` via tessellation."""
    vertices, triangles = tessellate_solid(solid, tolerance=tolerance)
    return MeshData(
        vertices=tuple((float(v.X), float(v.Y), float(v.Z)) for v in vertices),
        face_vertex_counts=(3,) * len(triangles),
        face_vertex_indices=tuple(index for tri in triangles for index in tri),
    )


def form_mesh_data(
    form: ShapeForm,
    size: float,
    height: float,
    edge_chamfer: float = DEFAULT_EDGE_CHAMFER,
    tolerance: float | None = None,
) -> MeshData:
    """Tessellate a sorting piece of the given form."""
    if tolerance is None:
        tolerance = tessellation_tolerance(edge_chamfer=edge_chamfer)
    return mesh_data_from_solid(piece_solid(form, size, height, edge_chamfer=edge_chamfer), tolerance=tolerance)


def write_usd_mesh(stage: Usd.Stage, prim_path: str, mesh: MeshData) -> None:
    """Author a ``UsdGeom.Mesh`` prim from triangle data."""
    from pxr import Gf, UsdGeom, Vt

    mesh_api = UsdGeom.Mesh.Define(stage, prim_path)
    mesh_api.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(v[0], v[1], v[2]) for v in mesh.vertices]))
    mesh_api.CreateFaceVertexCountsAttr(Vt.IntArray(list(mesh.face_vertex_counts)))
    mesh_api.CreateFaceVertexIndicesAttr(Vt.IntArray(list(mesh.face_vertex_indices)))
    mesh_api.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.bilinear)


def _apply_mesh_collision(stage: Usd.Stage, mesh_prim_path: str, approximation: str, cfg) -> None:
    from pxr import UsdPhysics

    from isaaclab.sim.schemas import schemas_cfg

    if cfg.collision_props is None:
        return

    if approximation == "sdf":
        # MeshCollisionAPI(approximation=sdf) alone is not enough; PhysX also needs
        # PhysxSDFMeshCollisionAPI (resolution, etc.) for cooking / debug viz.
        schemas.define_mesh_collision_properties(
            mesh_prim_path, schemas_cfg.SDFMeshPropertiesCfg(), stage=stage
        )
    else:
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(stage.GetPrimAtPath(mesh_prim_path))
        mesh_collision_api.GetApproximationAttr().Set(approximation)

    schemas.define_collision_properties(mesh_prim_path, cfg.collision_props, stage=stage)


def _bind_materials(stage: Usd.Stage, cfg, geom_prim_path: str, mesh_prim_path: str) -> None:
    if cfg.visual_material is not None:
        material_path = (
            f"{geom_prim_path}/{cfg.visual_material_path}"
            if not cfg.visual_material_path.startswith("/")
            else cfg.visual_material_path
        )
        cfg.visual_material.func(material_path, cfg.visual_material)
        bind_visual_material(mesh_prim_path, material_path, stage=stage)

    if cfg.physics_material is not None:
        material_path = (
            f"{geom_prim_path}/{cfg.physics_material_path}"
            if not cfg.physics_material_path.startswith("/")
            else cfg.physics_material_path
        )
        cfg.physics_material.func(material_path, cfg.physics_material)
        bind_physics_material(mesh_prim_path, material_path, stage=stage)


def _finalize_rigid_root(prim_path: str, cfg, stage: Usd.Stage) -> None:
    if cfg.mass_props is not None:
        schemas.define_mass_properties(prim_path, cfg.mass_props, stage=stage)
    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props, stage=stage)


@configclass
class ProceduralMeshCfg(MeshCfg):
    """Spawn config for a single procedurally authored triangle mesh."""

    func: Callable | str = "shape_sorting.shape_mesh:spawn_procedural_mesh"

    form: ShapeForm = ShapeForm.CUBE
    """Plan-view silhouette of the piece."""

    size: float = 0.05
    """Characteristic plan-view size (m)."""

    height: float = 0.05
    """Extrusion along Z (m)."""

    edge_chamfer: float = DEFAULT_EDGE_CHAMFER
    """Top/bottom plan-edge chamfer on the piece (m)."""

    mesh_data: MeshData | None = None
    """Explicit mesh; when set, overrides ``form`` / ``size`` / ``height``."""

    collision_approximation: str = "convexHull"
    """PhysX mesh collision approximation."""


@configclass
class ProceduralAssemblyCfg(RigidObjectSpawnerCfg):
    """Spawn config for a multi-part rigid assembly (e.g. a sorting box)."""

    func: Callable | str = "shape_sorting.shape_mesh:spawn_procedural_assembly"

    parts: tuple[MeshPart, ...] = MISSING
    """Named mesh parts authored under ``geometry/<name>/mesh``."""

    visual_material_path: str = "material"
    """Relative material path under each part geometry prim."""

    visual_material: materials.VisualMaterialCfg | None = None
    """Optional visual material applied to every part mesh."""

    physics_material_path: str = "material"
    """Relative physics-material path under each part geometry prim."""

    physics_material: materials.PhysicsMaterialCfg | None = None
    """Optional physics material applied to every part mesh."""


def _mesh_for_piece_cfg(cfg: ProceduralMeshCfg) -> MeshData:
    if cfg.mesh_data is not None:
        return cfg.mesh_data
    return form_mesh_data(cfg.form, cfg.size, cfg.height, edge_chamfer=cfg.edge_chamfer)


@clone
def spawn_procedural_mesh(
    prim_path: str,
    cfg: ProceduralMeshCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a rigid triangle mesh authored with the OpenUSD Python API."""
    stage = get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation, stage=stage)

    geom_prim_path = f"{prim_path}/geometry"
    mesh_prim_path = f"{geom_prim_path}/mesh"
    write_usd_mesh(stage, mesh_prim_path, _mesh_for_piece_cfg(cfg))
    _apply_mesh_collision(stage, mesh_prim_path, cfg.collision_approximation, cfg)
    _bind_materials(stage, cfg, geom_prim_path, mesh_prim_path)
    _finalize_rigid_root(prim_path, cfg, stage)
    return stage.GetPrimAtPath(prim_path)


@clone
def spawn_procedural_assembly(
    prim_path: str,
    cfg: ProceduralAssemblyCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a rigid body with one named mesh prim per assembly part."""
    stage = get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation, stage=stage)

    geom_root = f"{prim_path}/geometry"
    for part in cfg.parts:
        part_geom = f"{geom_root}/{part.name}"
        mesh_prim_path = f"{part_geom}/mesh"
        write_usd_mesh(stage, mesh_prim_path, part.mesh)
        _apply_mesh_collision(stage, mesh_prim_path, part.collision_approximation, cfg)
        _bind_materials(stage, cfg, part_geom, mesh_prim_path)

    _finalize_rigid_root(prim_path, cfg, stage)
    return stage.GetPrimAtPath(prim_path)


def ideal_hole_grid(n: int) -> tuple[int, int]:
    """Near-square ``(cols, rows)`` that fits ``n`` holes (cols >= rows when tied)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return cols, rows


def build_sorting_box_parts(
    forms: Sequence[ShapeForm],
    form_sizes: Sequence[float],
    box_height: float,
    clearance: float,
    wall_thickness: float = 0.008,
    lid_thickness: float = 0.006,
    bottom_thickness: float = 0.006,
    hole_gap: float = 0.012,
    hole_chamfer: float = DEFAULT_HOLE_CHAMFER,
    tolerance: float | None = None,
) -> tuple[tuple[MeshPart, ...], tuple[tuple[float, float], ...]]:
    """Build wall / lid / bottom meshes and return hole XY centers in box frame.

    Holes are packed in a near-square grid; outer XY size follows ``cols`` /
    ``rows``, the largest form, and ``clearance``. AABB is origin-centered.
    """
    from build123d import Align, Axis, Box, BuildPart, Location, Mode, Select, add, chamfer

    from shape_sorting.shape_forms import clamp_hole_chamfer

    if not forms:
        raise ValueError("Sorting box requires at least one form.")
    if len(forms) != len(form_sizes):
        raise ValueError(
            f"forms and form_sizes length mismatch: {len(forms)} forms vs {len(form_sizes)} sizes."
        )

    clamped_hole_chamfer = clamp_hole_chamfer(hole_chamfer, lid_thickness)
    if tolerance is None:
        tolerance = tessellation_tolerance(hole_chamfer=clamped_hole_chamfer)

    max_form_size = max(form_sizes)
    hole_span = max_form_size + 2.0 * clearance
    pitch = hole_span + hole_gap  # center-to-center spacing
    n = len(forms)
    cols, rows = ideal_hole_grid(n)

    # Margin of hole_gap around the hole cells on each side.
    inner_x = cols * pitch + hole_gap
    inner_y = rows * pitch + hole_gap
    outer_x = inner_x + 2.0 * wall_thickness
    outer_y = inner_y + 2.0 * wall_thickness

    if box_height <= lid_thickness + bottom_thickness:
        raise ValueError(
            f"box_height ({box_height}) must exceed lid + bottom thickness "
            f"({lid_thickness + bottom_thickness})."
        )

    # Row-major grid, centered on origin (x right, y back).
    hole_centers = tuple(
        (
            (i % cols - (cols - 1) / 2.0) * pitch,
            ((rows - 1) / 2.0 - i // cols) * pitch,
        )
        for i in range(n)
    )
    z_bottom = -0.5 * box_height + 0.5 * bottom_thickness
    z_lid = 0.5 * box_height - 0.5 * lid_thickness
    wall_height = box_height - lid_thickness - bottom_thickness
    z_wall = -0.5 * box_height + bottom_thickness + 0.5 * wall_height

    bottom = Box(
        outer_x,
        outer_y,
        bottom_thickness,
        align=(Align.CENTER, Align.CENTER, Align.CENTER),
    ).translate((0.0, 0.0, z_bottom))

    front = Box(
        outer_x,
        wall_thickness,
        wall_height,
        align=(Align.CENTER, Align.CENTER, Align.CENTER),
    ).translate((0.0, -0.5 * (outer_y - wall_thickness), z_wall))
    back = Box(
        outer_x,
        wall_thickness,
        wall_height,
        align=(Align.CENTER, Align.CENTER, Align.CENTER),
    ).translate((0.0, 0.5 * (outer_y - wall_thickness), z_wall))
    left = Box(
        wall_thickness,
        inner_y,
        wall_height,
        align=(Align.CENTER, Align.CENTER, Align.CENTER),
    ).translate((-0.5 * (outer_x - wall_thickness), 0.0, z_wall))
    right = Box(
        wall_thickness,
        inner_y,
        wall_height,
        align=(Align.CENTER, Align.CENTER, Align.CENTER),
    ).translate((0.5 * (outer_x - wall_thickness), 0.0, z_wall))

    with BuildPart() as lid_builder:
        Box(
            outer_x,
            outer_y,
            lid_thickness,
            align=(Align.CENTER, Align.CENTER, Align.CENTER),
        )
        for (hx, hy), form, form_size in zip(hole_centers, forms, form_sizes):
            cutter = hole_cutter(form, form_size, clearance, cut_depth=lid_thickness * 2.0)
            add(cutter.moved(Location((hx, hy, 0.0))), mode=Mode.SUBTRACT)
            if clamped_hole_chamfer > 0.0:
                try:
                    top_rim = lid_builder.edges(Select.LAST).group_by(Axis.Z)[-1]
                    if top_rim:
                        chamfer(top_rim, length=clamped_hole_chamfer)
                except Exception:
                    pass
    lid = lid_builder.part.translate((0.0, 0.0, z_lid))

    parts = (
        MeshPart("bottom", mesh_data_from_solid(bottom, tolerance), "boundingCube"),
        MeshPart("front", mesh_data_from_solid(front, tolerance), "boundingCube"),
        MeshPart("back", mesh_data_from_solid(back, tolerance), "boundingCube"),
        MeshPart("left", mesh_data_from_solid(left, tolerance), "boundingCube"),
        MeshPart("right", mesh_data_from_solid(right, tolerance), "boundingCube"),
        # Concave lid: keep holes traversable for insertion.
        MeshPart("lid", mesh_data_from_solid(lid, tolerance), "sdf"),
    )
    return parts, hole_centers
