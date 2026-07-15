"""Procedural mesh spawning via the OpenUSD Python API.

Builds triangle meshes at spawn time and wires them into Isaac Lab's rigid-body
hierarchy (``Xform`` root + ``geometry/mesh`` child). Used by
:class:`shape_sorting.shape_asset.ShapePiece`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING, dataclass
from typing import TYPE_CHECKING

from isaaclab.sim import schemas
from isaaclab.sim.spawners.meshes.meshes_cfg import MeshCfg
from isaaclab.sim.utils import bind_physics_material, bind_visual_material, clone, create_prim, get_current_stage
from isaaclab.utils import configclass

from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

if TYPE_CHECKING:
    from pxr import Usd


@dataclass(frozen=True)
class MeshData:
    """Triangle mesh in local coordinates."""

    vertices: tuple[tuple[float, float, float], ...]
    face_vertex_counts: tuple[int, ...]
    face_vertex_indices: tuple[int, ...]


def bounding_box_from_mesh_data(mesh: MeshData) -> AxisAlignedBoundingBox:
    """Axis-aligned bounds of mesh vertices in local coordinates."""
    xs, ys, zs = zip(*mesh.vertices)
    return AxisAlignedBoundingBox(
        min_point=(min(xs), min(ys), min(zs)),
        max_point=(max(xs), max(ys), max(zs)),
    )


def box_mesh_data(size: tuple[float, float, float]) -> MeshData:
    """Return a centered axis-aligned box as 12 triangles."""
    sx, sy, sz = size
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
    vertices = (
        (-hx, -hy, -hz),
        (hx, -hy, -hz),
        (hx, hy, -hz),
        (-hx, hy, -hz),
        (-hx, -hy, hz),
        (hx, -hy, hz),
        (hx, hy, hz),
        (-hx, hy, hz),
    )
    triangles = (
        (0, 1, 2),
        (0, 2, 3),
        (4, 6, 5),
        (4, 7, 6),
        (0, 4, 5),
        (0, 5, 1),
        (2, 6, 7),
        (2, 7, 3),
        (0, 3, 7),
        (0, 7, 4),
        (1, 5, 6),
        (1, 6, 2),
    )
    return MeshData(
        vertices=vertices,
        face_vertex_counts=(3,) * len(triangles),
        face_vertex_indices=tuple(index for tri in triangles for index in tri),
    )


def write_usd_mesh(stage: Usd.Stage, prim_path: str, mesh: MeshData) -> None:
    """Author a ``UsdGeom.Mesh`` prim from triangle data."""
    from pxr import Gf, UsdGeom, Vt

    mesh_api = UsdGeom.Mesh.Define(stage, prim_path)
    mesh_api.CreatePointsAttr(
        Vt.Vec3fArray([Gf.Vec3f(v[0], v[1], v[2]) for v in mesh.vertices])
    )
    mesh_api.CreateFaceVertexCountsAttr(Vt.IntArray(list(mesh.face_vertex_counts)))
    mesh_api.CreateFaceVertexIndicesAttr(Vt.IntArray(list(mesh.face_vertex_indices)))
    mesh_api.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.bilinear)


@configclass
class ProceduralMeshCfg(MeshCfg):
    """Spawn config for procedurally authored triangle meshes."""

    func: Callable | str = "shape_sorting.shape_mesh:spawn_procedural_mesh"

    size: tuple[float, float, float] = MISSING
    """Box extent (m) when using the built-in cube builder."""

    mesh_data: MeshData | None = None
    """Explicit mesh; when set, overrides the cube built from ``size``."""

    collision_approximation: str = "boundingCube"
    """PhysX mesh collision approximation (e.g. ``boundingCube``, ``convexHull``)."""


def _mesh_for_cfg(cfg: ProceduralMeshCfg) -> MeshData:
    if cfg.mesh_data is not None:
        return cfg.mesh_data
    return box_mesh_data(cfg.size)


@clone
def spawn_procedural_mesh(
    prim_path: str,
    cfg: ProceduralMeshCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a rigid triangle mesh authored with the OpenUSD Python API."""
    from pxr import Usd, UsdPhysics

    stage = get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation, stage=stage)

    geom_prim_path = f"{prim_path}/geometry"
    mesh_prim_path = f"{geom_prim_path}/mesh"
    write_usd_mesh(stage, mesh_prim_path, _mesh_for_cfg(cfg))

    if cfg.collision_props is not None:
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(stage.GetPrimAtPath(mesh_prim_path))
        mesh_collision_api.GetApproximationAttr().Set(cfg.collision_approximation)
        schemas.define_collision_properties(mesh_prim_path, cfg.collision_props, stage=stage)

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

    if cfg.mass_props is not None:
        schemas.define_mass_properties(prim_path, cfg.mass_props, stage=stage)
    if cfg.rigid_props is not None:
        schemas.define_rigid_body_properties(prim_path, cfg.rigid_props, stage=stage)

    return stage.GetPrimAtPath(prim_path)
