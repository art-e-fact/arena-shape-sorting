"""Shape form definitions and build123d solid / hole builders.

``size`` is the plan-view characteristic length (bounding-circle diameter for
polygons, side length for the square, outer width for the cross). ``height`` is
the Z extrusion. Hole cutters inflate the plan profile by ``clearance``.
"""

from __future__ import annotations

import math
from enum import Enum

from build123d import (
    Align,
    BuildPart,
    BuildSketch,
    Box,
    Circle,
    Cylinder,
    Mode,
    Polygon,
    Rectangle,
    RegularPolygon,
    Vector,
    extrude,
    offset,
)


class ShapeForm(Enum):
    """Plan-view silhouette used for pieces and matching sorter holes."""

    CUBE = "cube"
    CYLINDER = "cylinder"
    TRIANGLE = "triangle"
    HEXAGON = "hexagon"
    STAR = "star"
    CROSS = "cross"


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


def piece_solid(form: ShapeForm, size: float, height: float):
    """Return a centered solid for a manipulable sorting piece."""
    if form is ShapeForm.CUBE:
        return Box(size, size, height, align=(Align.CENTER, Align.CENTER, Align.CENTER))
    if form is ShapeForm.CYLINDER:
        return Cylinder(size * 0.5, height, align=(Align.CENTER, Align.CENTER, Align.CENTER))

    with BuildPart() as part:
        with BuildSketch():
            add_form_profile(form, size)
        extrude(amount=height)
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
