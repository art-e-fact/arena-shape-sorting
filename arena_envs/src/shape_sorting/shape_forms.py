"""Shape form definitions and build123d solid / hole builders.

``size`` is the plan-view characteristic length (bounding-circle diameter for
polygons, side length for the square, outer width for the cross). ``height`` is
the Z extrusion. Hole cutters inflate the plan profile by ``clearance``.
"""

from __future__ import annotations

import math
from enum import Enum

from build123d import (
    Axis,
    BuildPart,
    BuildSketch,
    Circle,
    Mode,
    Polygon,
    Rectangle,
    RegularPolygon,
    Vector,
    chamfer,
    extrude,
    offset,
)

DEFAULT_EDGE_CHAMFER = 0.001
"""Default top/bottom plan-edge chamfer on sorting pieces (m)."""

DEFAULT_HOLE_CHAMFER = 0.001
"""Default lead-in chamfer on lid hole rims (m)."""


class ShapeForm(Enum):
    """Plan-view silhouette used for pieces and matching sorter holes."""

    CUBE = "cube"
    CYLINDER = "cylinder"
    TRIANGLE = "triangle"
    HEXAGON = "hexagon"
    STAR = "star"
    CROSS = "cross"


_CHAMFER_SKIP_FORMS = frozenset({ShapeForm.STAR, ShapeForm.CROSS})

DEFAULT_FORMS: tuple[ShapeForm, ...] = (
    ShapeForm.CUBE,
    ShapeForm.CYLINDER,
    ShapeForm.TRIANGLE,
    ShapeForm.HEXAGON,
)


def _center_on_origin(solid):
    """Translate a solid so its AABB center sits at the origin."""
    center = solid.bounding_box().center()
    return solid.translate((-center.X, -center.Y, -center.Z))


def _star_points(outer_r: float, inner_r: float, n: int = 5) -> list[Vector]:
    pts: list[Vector] = []
    for i in range(n * 2):
        r = outer_r if i % 2 == 0 else inner_r
        angle = i * math.pi / n - math.pi / 2
        pts.append(Vector(r * math.cos(angle), r * math.sin(angle)))
    return pts


def clamp_piece_edge_chamfer(edge_chamfer: float, size: float, height: float) -> float:
    """Clamp piece chamfer so it stays below local width and height limits."""
    if edge_chamfer <= 0.0:
        return 0.0
    return min(edge_chamfer, height * 0.45, size * 0.08)


def clamp_hole_chamfer(hole_chamfer: float, lid_thickness: float) -> float:
    """Clamp lid hole rim chamfer to a fraction of lid thickness."""
    if hole_chamfer <= 0.0:
        return 0.0
    return min(hole_chamfer, lid_thickness * 0.45)


def tessellation_tolerance(*, edge_chamfer: float = 0.0, hole_chamfer: float = 0.0) -> float:
    """Use a finer tessellation tolerance when chamfers are active."""
    if edge_chamfer > 0.0 or hole_chamfer > 0.0:
        return 5e-4
    return 1e-3


def _chamfer_top_bottom_plan_edges(builder: BuildPart, length: float) -> None:
    """Chamfer the highest and lowest Z edge groups (plan perimeters)."""
    if length <= 0.0:
        return
    edges_by_z = builder.edges().group_by(Axis.Z)
    if not edges_by_z:
        return
    chamfer(edges_by_z[-1], length=length)
    if len(edges_by_z) > 1:
        chamfer(edges_by_z[0], length=length)


def add_form_profile(form: ShapeForm, size: float) -> None:
    """Add the plan-view profile of ``form`` into the active :class:`BuildSketch`."""
    half = size * 0.5
    if form is ShapeForm.CUBE:
        Rectangle(size, size)
    elif form is ShapeForm.CYLINDER:
        Circle(half)
    elif form is ShapeForm.TRIANGLE:
        RegularPolygon(half, 3)
    elif form is ShapeForm.HEXAGON:
        RegularPolygon(half, 6)
    elif form is ShapeForm.STAR:
        Polygon(*_star_points(half, half * 0.38))
    elif form is ShapeForm.CROSS:
        arm = size * 0.32
        Rectangle(size, arm)
        Rectangle(arm, size, mode=Mode.ADD)
    else:
        raise ValueError(f"Unsupported shape form: {form}")


def piece_solid(
    form: ShapeForm,
    size: float,
    height: float,
    edge_chamfer: float = DEFAULT_EDGE_CHAMFER,
):
    """Return a centered solid for a manipulable sorting piece.

    When ``edge_chamfer`` is positive, top and bottom plan perimeters are chamfered.
    """
    clamped = clamp_piece_edge_chamfer(edge_chamfer, size, height)
    with BuildPart() as part:
        with BuildSketch():
            add_form_profile(form, size)
        extrude(amount=height)
        if clamped > 0.0 and form not in _CHAMFER_SKIP_FORMS:
            try:
                _chamfer_top_bottom_plan_edges(part, clamped)
            except Exception:
                # Re-entrant silhouettes can fail at large chamfers; keep the sharp solid.
                pass
    return _center_on_origin(part.part)


def hole_cutter(form: ShapeForm, size: float, clearance: float, cut_depth: float):
    """Centered prism used to cut a clearance-inflated hole through the lid."""
    with BuildPart() as cutter:
        with BuildSketch():
            add_form_profile(form, size)
            offset(amount=clearance)
        extrude(amount=cut_depth)
    return _center_on_origin(cutter.part)


def tessellate_solid(solid, tolerance: float = 1e-3):
    """Tessellate a build123d solid into ``(vertices, triangles)``."""
    return solid.tessellate(tolerance=tolerance)
