"""cuRobo pick-and-place policy that scripts SO-101 shape-sorting demonstrations.

Design rules — keep them when changing this file
------------------------------------------------
This policy grows a recovery at a time; these rules are what keeps it readable. Break
one only on purpose, and update this block when you do.

1. **Phases are motions, not situations.** There are three: ``GO`` (a planned approach —
   move to a hover pose, then descend to contact), ``JAW`` (open or close in place, then
   check what that achieved) and ``UP`` (back out along the descent). ``HOME``/``DONE``
   end the episode. A new behaviour is a new branch in :meth:`CuroboPolicy._decide`, not
   a new phase.
2. **Decide in one place, from the world — once it is still.** ``_decide`` runs whenever
   the arm is clear of contact and reads the world — is something in the jaws, is the
   piece in the box, how is it lying — through :func:`next_goal`, a plain decision table.
   ``UP`` holds until the pieces stop moving first, so nothing is decided about a piece
   in mid-fall. No flag carries "what to do next" from one phase into another, and no
   pose is remembered: the only memory is per-piece retry counters, plus two per-attempt
   facts the *recording* needs (was this attempt already cut, how tipped was the piece
   before it was grasped). One exception, on purpose: a piece let go misaligned on purpose
   (``place_release_misaligned_prob``) is left where it lands and the arm climbs out,
   while an accidental failed drop is closed on again — and once the jaw is open, the
   world no longer shows which of the two it was.
3. **Plan whole approaches, and only from clear poses.** A candidate counts only if both
   its hover move and its descent plan. After contact the arm leaves by replaying the
   descent — it never asks cuRobo to start from a pose that touches something, which
   cuRobo refuses and which used to freeze the arm after a release.
4. **Every failure has an exit that is not a freeze.** A candidate that does not plan
   gives way to the next; a piece with none goes to the back of the queue, because
   moving the others can free it up; a full round without progress ends the episode.
5. **Record only what should be imitated.** Every attempt that does not do what it set
   out to is cut — at the moment it is caught where that gives the recovery a better
   first frame, otherwise by ``_decide`` — and "checkpoint" happens only once a piece is
   verifiably in the box. A clip must never confirm a mistake by accident.

How a piece moves
-----------------
::

    GO(grasp) → JAW(close) → UP → decide → GO(insert) → JAW(open) → UP → decide → …

``_decide`` (via :func:`next_goal`)::

    holding, upright, tries left      → INSERT   into its hole
    holding, tipped or out of tries    → PARK     on a free table spot, standing up
    empty,   piece in the box          → finish   "checkpoint", on to the next piece
    empty,   out of tries              → defer    to the back of the queue
    empty,   otherwise                 → GRASP    wherever the piece lies now — on the
                                                  table (on its side, too), on the lid,
                                                  wedged in its hole

Where a mistake is caught, and what follows:

* jaw closes on nothing → "cut", UP with the jaw opening, grasp again;
* the held piece is off its hole at the bottom of the insert descent (XY, yaw folded into
  its symmetry, tilt) → "cut", UP still holding it, re-aim or park;
* the released piece never turns up in the cavity → "cut", JAW closes on it again — it is
  still between the jaws, so that needs no plan — then as above (another "cut" if that
  close finds nothing);
* with ``--place_release_misaligned_prob``, a piece caught off its hole is let go anyway,
  as a trained policy playing its chunk open-loop would → "cut" after the settle wait (or
  once it turns up in the box: letting go there was still the mistake), UP with the jaw
  open, grasp it where it lies. Only while a regrasp and another insert are in budget, so
  a let-go piece is never deferred where it lies;
* a grasp that let the piece slip, or tipped an upright piece in the jaws → "cut" in
  ``_decide``; giving up on a piece that used any tries, or dropping one in mid-air
  because nothing plans → "cut" too.

**Grasp candidates.** The SO-101 wrist rolls about the gripper's own axis, and that axis
is vertical in a grasp pose, so the jaw can close along any yaw. Candidates point it along
the piece's face normals (:func:`shape_sorting.shape_forms.face_normal_yaws`) — squeezing
two flats, never two corners — starting from the radial grasp this policy was tuned with,
so a neighbour sitting on that line no longer blocks the piece. On a tipped piece only the
faces that stayed vertical count, so a piece lying on its side is squeezed across its
sides — never on its end faces, which would leave a finger under it once it stands.

**Standing a piece up.** A piece that comes up tipped — pulled out of its hole at an
angle, or picked up lying on its side — is parked with the wrist pitched so that it
lands standing (:func:`upright_tilt_rolls`). The grip fixes the piece's axis in the
gripper; its yaw is free, and choosing it puts the gripper in the arm's plane, where the
5-DoF wrist can reach — about the pan axis, which is not the base origin. If that tilt
does not plan it tries leaning 30°, from which the piece still falls onto its base, and
last the old way: gripper upright, gravity levels anything tipped under ~45°. A tilted
gripper reaches the table only further out (``_SETDOWN_MIN_REACH_M``), so those spots
are further out. Bounded like any retry: each go is a grasp.

**Demo events.** ``pop_demo_events()`` reports "checkpoint" (verified sub-step —
everything recorded so far is worth keeping) and "cut" (own mistake detected — drop back
to the last checkpoint and start a new clip here). ``generate_policy_demos`` turns these
into LeRobot episode boundaries; see ``demo_clips``. Other runners can ignore them.
``--grasp_perturb_prob`` and ``--place_perturb_prob`` make first attempts miss on purpose
so the recoveries get recorded, and ``--miss_notice_delay_max_steps`` delays noticing a
missed grasp so those clips also start mid-transport. The env's ``--drop_on_hole_prob``
starts a piece stuck in its hole or lying on the lid instead — the state a trained policy leaves
after letting go in the wrong place — and this policy needs nothing special for it:
``_decide`` sees a piece that is not in the box and grasps it where it lies. The queue
just starts with whatever starts on the box (:meth:`CuroboPolicy._pending`).

Plans via ``MotionClient`` (see ``curobo_motion``); collision-world sync lives there too.
Requires ``--embodiment so101_abs_joint``.

Example::

    python -m shape_sorting.run_policy --enable_cameras --viz kit \\
      --policy_type shape_sorting.curobo_policy.CuroboPolicy \\
      --num_steps 3000 \\
      --external_environment_class_path shape_sorting.shape_sorting_env:ShapeSortingEnvironment \\
      shape_sorting_test --embodiment so101_abs_joint
"""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, NamedTuple

import gymnasium as gym
import torch
from gymnasium.spaces.dict import Dict as GymSpacesDict

from arena_so101 import JAW_CLOSE_RAD, JAW_OPEN_RAD
from arena_so101.mapping import SIM_JOINT_NAMES
from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg
from shape_sorting.curobo_motion import MotionClient, MotionClientCfg, resolve_robot_yml
from shape_sorting.curobo_viz import KitFrameMarkers, NullCollisionDebugViz, ViserCollisionDebugViz
from shape_sorting.curobo_world import entity_pose_in_robot_base, entity_position_in_robot_base
from shape_sorting.shape_forms import ShapeForm, face_normal_yaws, place_yaw_offsets, yaw_symmetry_order
from shape_sorting.shape_sorting_env import ShapeInfo

if TYPE_CHECKING:
    EePose = tuple[torch.Tensor, torch.Tensor]
    """``(position (3,), quat_xyzw (4,))`` in the robot base frame."""
    Candidate = tuple[str, EePose, EePose]
    """``(label, hover pose, contact pose)``."""

# Grasp pose relative to goal_object (robot base frame). Tuned with the ``gripper`` link
# origin as the cuRobo tool frame; the tool frame is now ``tcp`` between the jaw tips, 7 mm
# behind and 102 mm below it (arena_so101.TCP_OFFSET), so the standoff carries those 7 mm
# and ``grasp_height_m`` the 102 mm: the physical grasp is unchanged.
_GOAL_XY_STANDOFF_M = 0.037
_GOAL_TILT_RAD = 0.0
_GOAL_ROLL_RAD = 0.0

_JAW_INDEX = SIM_JOINT_NAMES.index("Jaw")

# Object-origin height above its resting height when parked on the table. Just enough
# that the piece is dropped rather than pressed into the table by the position-controlled
# arm.
_PARK_RELEASE_CLEARANCE_M = 0.002

# A piece let go of with an edge on the table falls onto its base while its centre is over
# that base: tipped under atan(r/h) ≈ 47° for a 30 mm cylinder or hexagon — which matches
# what parking did live (47° came up level, 53° and 72° landed on a side). So a piece that
# cannot be set down upright is set down leaning this much, with margin to spare, which
# asks that much less of the wrist near the table.
_SETDOWN_LEAN_RAD = math.radians(30.0)
# Past this the gripper would point up — into the table, for a set-down.
_SETDOWN_MAX_TILT_RAD = math.radians(90.0)
# A tilted gripper sits ~0.1 m nearer the base than the piece it holds, so a set-down
# needs the spot further out. Offline cuRobo sweep, piece 4 mm over the table: 0.20 m
# out reaches 30° of tilt, 0.25 m 45-60°, 0.30 m 75°; the same in every direction once
# the arm plane is taken about the pan axis. Only with the fingertips pointing away
# from the base — pointing back, nothing past 15° planned — and, past 45°, only with the
# wrist rolled to about -90..-135° (or +135°): the other rolls put the gripper's bulky
# side into the table. A piece lying flat can go down on either end, which offers both
# signs, so it always gets one that works; one held at 45-70° gets whichever its grip
# gave, and with the wrong one it takes the long way — parked as held, lands lying, is
# picked up lying and stood up from there.
_SETDOWN_MIN_REACH_M = 0.24
# ...and a stood-up piece does not stay where it is let go: it rocks onto its base, toward
# the robot. Live, 10-14 mm — enough, from a spot behind the box, to end up against the
# wall, where the next grasp closed on piece and wall together and lifted the box.
_SETDOWN_DRIFT_M = 0.015

# The shoulder-pan axis in the robot base frame (URDF joint "Rotation"). The arm's plane
# turns about it, not about the base origin 3 cm away: next to the box that is 7° of
# azimuth, which an upright gripper absorbs in its roll and a tilted one cannot.
# ponytail: copied from SO-ARM101-USD.urdf; read it off the cuRobo kinematics if the
# robot model changes.
_PAN_AXIS_XY = (0.0207909, -0.0230745)

# Half-height and largest plan radius of a piece at the default 30 mm equal-area size (the
# cube's half-diagonal, 21.2 mm, is the widest). Used to release a tipped piece high enough
# that its lowest corner clears the table, and to keep parked pieces apart.
# ponytail: default piece size only; read piece_size/piece_height off the env cfg if those
# become variable.
_PIECE_HALF_HEIGHT_M = 0.015
_PIECE_MAX_RADIUS_M = 0.022

# The fingertip collision spheres' lowest point below the cuRobo tool frame (``tcp``, between
# the jaw tips), and the gap kept between them and the lid when grasping a piece that sits
# on it or in a hole.
_FINGERTIP_BELOW_TOOL_M = 0.003
_LID_FINGERTIP_CLEARANCE_M = 0.003

# Where a piece is set down to be picked up level: clear of the box wall (SortingBox's
# default wall_thickness) and of every other piece's outline by _PARK_CLEARANCE_M.
_BOX_WALL_M = 0.008
_PARK_CLEARANCE_M = 0.01
_PARK_MAX_SPOTS = 3
# ...and no nearer the robot's base than pieces are ever placed. Closer in, the arm folds
# over itself for a top-down grasp; live, a regrasp 12 cm from the base knocked the cube
# over and shoved the box.
_PARK_MIN_REACH_M = 0.14

# "Still" for _decide: pieces spawn 1 cm up and drop, and one released on the lid can take
# a while to tip into its final pose. Bounded, so a piece that keeps rolling cannot stall
# the episode.
_STILL_SPEED_M_S = 0.02
_SETTLE_MAX_STEPS = 30

_HOME_JOINT_RAD = tuple(
    math.radians(v)
    for v in (
        -1.6e-05,  # shoulder_pan / Rotation
        4.5e-03,   # shoulder_lift / Pitch
        5.5e-03,   # elbow_flex / Elbow
        2.1e-03,   # wrist_flex / Wrist_Pitch
        -4.6e-06,  # wrist_roll / Wrist_Roll
        0.200,     # gripper / Jaw
    )
)


class Phase(Enum):
    """What the arm is doing. Motions only — see the design rules above."""

    GO = auto()
    JAW = auto()
    UP = auto()
    HOME = auto()
    DONE = auto()


_PHASE_STEPS: dict[Phase, str] = {
    Phase.GO: "_step_go",
    Phase.JAW: "_step_jaw",
    Phase.UP: "_step_up",
    Phase.HOME: "_step_home",
}
"""Handler per phase. ``Phase.DONE`` is the one phase with no step — it just holds.

A table rather than an if/elif chain so a phase added without a handler fails a test
instead of silently freezing the arm (``tests/test_insert_align.py``).
"""


class Goal(Enum):
    """Where the current ``GO`` is taking the arm."""

    GRASP = auto()   # the current piece, wherever it lies
    INSERT = auto()  # the current piece's hole
    PARK = auto()    # a free table spot — see park_spots()


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _tilt_of(quat_xyzw: torch.Tensor) -> float:
    """Angle between a body's own +Z and the base +Z [rad]."""
    return math.acos(max(-1.0, min(1.0, float(_rot_from_quat_xyzw(quat_xyzw)[2, 2]))))


def standing_quat(quat_xyzw: torch.Tensor, *, any_face: bool = False) -> torch.Tensor:
    """The same piece with its axis relabelled as +Z the way it points up.

    Everything here reads "up" off a piece's own +Z, but every piece is a prism, the same
    solid upside down: one standing on its top *is* standing, and fits its hole.
    Relabelling the axes — a symmetry of the solid — lets every rule see it that way,
    instead of picking it up only to stand it up again. ``any_face`` picks whichever
    body axis is nearest vertical instead: a cube as tall as it is wide is the same solid
    on every face, so on its side it is standing too.
    """
    from isaaclab.utils.math import quat_from_matrix

    rot = _rot_from_quat_xyzw(quat_xyzw)
    i = max(range(3) if any_face else (2,), key=lambda k: abs(float(rot[2, k])))
    z = rot[:, i] * (1.0 if float(rot[2, i]) >= 0.0 else -1.0)
    x = rot[:, (i + 1) % 3]
    return quat_from_matrix(torch.stack([x, torch.linalg.cross(z, x), z], dim=1).unsqueeze(0))[0]


def _raised(xyz: torch.Tensor, dz: float) -> torch.Tensor:
    out = xyz.clone()
    out[2] += float(dz)
    return out


def _rot_from_quat_xyzw(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """3x3 rotation matrix from an xyzw quaternion. Columns are the body axes."""
    x, y, z, w = (float(v) for v in quat_xyzw.reshape(4))
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-9:
        raise ValueError(f"Degenerate quaternion: {(x, y, z, w)}")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=torch.float32,
    )


def alignment_error(
    obj_pos: torch.Tensor,
    obj_quat_xyzw: torch.Tensor,
    hole_pos: torch.Tensor,
    hole_quat_xyzw: torch.Tensor,
    *,
    symmetry_order: int | None,
) -> tuple[float, float, float]:
    """How far a piece is from dropping into its hole: ``(xy_m, yaw_rad, tilt_rad)``.

    ``yaw`` is folded into the piece's own rotational symmetry, so a cube 90° off reads
    0; ``symmetry_order=None`` (the cylinder) reports 0 always. ``tilt`` is the angle
    between the piece's +Z and the base +Z — the error a 5-DoF arm cannot correct, and
    the one that says the piece caught on the rim.

    Plain tensor math on purpose: no Isaac Sim import, so it is unit-testable offline
    (``tests/test_insert_align.py``).
    """
    dx = float(obj_pos[0]) - float(hole_pos[0])
    dy = float(obj_pos[1]) - float(hole_pos[1])
    xy = math.hypot(dx, dy)

    rot_obj = _rot_from_quat_xyzw(obj_quat_xyzw)
    tilt = math.acos(max(-1.0, min(1.0, float(rot_obj[2, 2]))))
    if symmetry_order is None:
        return xy, 0.0, tilt

    rot_hole = _rot_from_quat_xyzw(hole_quat_xyzw)
    delta = math.atan2(float(rot_obj[1, 0]), float(rot_obj[0, 0])) - math.atan2(
        float(rot_hole[1, 0]), float(rot_hole[0, 0])
    )
    # Folding by the symmetry period also wraps, so no separate wrap-to-pi is needed.
    period = 2.0 * math.pi / int(symmetry_order)
    return xy, abs(delta - period * round(delta / period)), tilt


def next_goal(
    *,
    holding: bool,
    upright: bool = True,
    in_box: bool = False,
    insert_tries_left: bool = True,
    grasp_tries_left: bool = True,
) -> str:
    """The decision table ``CuroboPolicy._decide`` runs on.

    Returns "insert", "park", "finish", "defer" or "grasp". A plain function so the whole
    policy can be read off a few lines and tested without a simulator. ``upright`` only
    matters while holding; ``in_box`` and ``grasp_tries_left`` only while empty. How a piece
    lies is not a question: whatever its pose, it is grasped where it lies and, if it comes
    up tipped, set down standing (``CuroboPolicy._park_candidates``).
    """
    if holding:
        return "insert" if upright and insert_tries_left else "park"
    if in_box:
        return "finish"
    if not (grasp_tries_left and insert_tries_left):
        return "defer"
    return "grasp"


def grasp_jaw_yaws(
    obj_quat_xyzw: torch.Tensor,
    face_normals: tuple[float, ...] | None,
    *,
    radial_yaw: float,
    max_tilt: float,
) -> list[float]:
    """Absolute jaw yaws [rad] to grasp a piece with, best first.

    The jaws close along the gripper's X axis, so pointing it along a face normal squeezes
    two flats instead of two corners — a cube held by its diagonal turns in the jaws and
    goes into the hole crooked. Only normals that are still roughly horizontal count: on a
    tipped piece those are the faces that stayed vertical (if none do, the least tipped
    ones). A round piece has a flat everywhere: upright it gets four evenly spaced yaws,
    which still matter for getting the fingers past a neighbour; tipped, the same four
    turned to its tilt axis, so the jaws close across a circle rather than squeeze the
    ellipse it shows along the tilt, and the filter below keeps the two along the axis.

    Ordered by angle from ``radial_yaw`` (the base→piece direction), because the radial
    grasp is the one ``grasp_height_m`` and the standoff were tuned with.
    """
    rot = _rot_from_quat_xyzw(obj_quat_xyzw)
    up = rot[:, 2]
    if face_normals is None and math.hypot(float(up[0]), float(up[1])) > 1e-3:
        axis = rot.T @ torch.tensor([-float(up[1]), float(up[0]), 0.0])  # tilt axis, body frame
        face_normals = tuple(math.atan2(float(axis[1]), float(axis[0])) + k * math.pi / 2.0 for k in range(4))
    if face_normals is None:
        yaws = [radial_yaw + k * math.pi / 2.0 for k in range(4)]
    else:
        world = [rot @ torch.tensor([math.cos(p), math.sin(p), 0.0]) for p in face_normals]
        slope = [abs(float(n[2])) for n in world]
        limit = max(math.sin(max_tilt), min(slope) + 1e-6)
        yaws = [math.atan2(float(n[1]), float(n[0])) for n, s in zip(world, slope) if s <= limit]
    # A hexagon on its side shows two sloped faces per side: same yaw, one plan is enough.
    unique = {round(math.degrees(y)) % 360: _wrap_to_pi(y) for y in yaws}
    return sorted(unique.values(), key=lambda y: abs(_wrap_to_pi(y - radial_yaw)))


def grasp_tool_xy_roll(
    piece_x: float, piece_y: float, jaw_yaw: float, standoff: float
) -> tuple[float, float, float]:
    """Tool-frame XY and roll that close the jaws along ``jaw_yaw`` around a piece.

    The fixed finger is on the tool's -X side, so the tool stands off the piece by
    ``standoff`` against the jaw direction — the geometry the radial grasp was tuned with,
    now rotated as a whole. The roll is relative to the yaw ``so101_ee_pose_xyzw`` derives
    from the tool position, and every value of it is reachable: the SO-101 wrist rolls
    about the gripper's own axis, which is vertical in a grasp pose. At
    ``jaw_yaw = atan2(piece_y, piece_x)`` this reproduces the original grasp exactly.
    """
    tool_x = piece_x - standoff * math.cos(jaw_yaw)
    tool_y = piece_y - standoff * math.sin(jaw_yaw)
    return tool_x, tool_y, _wrap_to_pi(jaw_yaw - math.atan2(tool_y, tool_x))


def upright_tilt_rolls(
    piece_z_in_tool: torch.Tensor, *, lean: float = 0.0, max_tilt: float = math.pi / 2.0
) -> list[tuple[float, float]]:
    """``(tilt, roll)`` for :func:`so101_ee_pose_xyzw` that stand a held piece up.

    ``piece_z_in_tool`` is the piece's own axis in the tool frame — fixed for as long as
    the grip holds, so this is a choice of gripper orientation alone. Standing the piece
    up leaves one freedom, its yaw, and that is exactly what makes it reachable: the yaw
    is chosen so the gripper axis ends up in the arm's vertical plane, the only place a
    5-DoF wrist can put it. Solving ``Ry(tilt) Rz(roll) a = +Z`` gives
    ``tilt = ±acos(a_z)`` with ``roll = -atan2(a_y, a_x)`` for the minus branch (fingertips
    pointing away from the base) and ``π`` minus that for the plus branch.

    ``lean`` stops that much short, tipping the piece in the arm plane: set down leaning,
    it still falls onto its base (below ~45° here — see ``_SETDOWN_LEAN_RAD``) and it
    asks for less wrist tilt near the table. Either end may go down, as every piece is a
    prism. Least tilt first, fingertips-away first among equals; past ``max_tilt`` the
    gripper would point up, into the table.
    """
    out = []
    for end in (1.0, -1.0):
        ax, ay, az = (end * float(v) for v in piece_z_in_tool.reshape(3))
        tilt = max(0.0, math.acos(max(-1.0, min(1.0, az))) - lean)
        roll = -math.atan2(ay, ax)
        if tilt <= max_tilt + 1e-6:
            out += [(-tilt, _wrap_to_pi(roll)), (tilt, _wrap_to_pi(math.pi + roll))]
    return sorted(out, key=lambda tr: (round(abs(tr[0]), 6), tr[0] > 0))


class BoxFootprint(NamedTuple):
    """The sorting box seen from above, in the robot base frame."""

    center: tuple[float, float]
    yaw: float
    lo: tuple[float, float]
    """Cavity AABB corners in the box frame (the success term's ``aabb_min/max``)."""
    hi: tuple[float, float]
    lid_z: float
    table_z: float
    """The table top — the box stands on it, and the lid is as far above its root."""

    def outside_by(self, xy: tuple[float, float]) -> float:
        """How far ``xy`` lies outside the cavity footprint [m]; negative over the cavity.

        The larger of the two per-axis distances, which is the true distance except past a
        corner, where it is slightly less — safe for every use here.
        """
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx, dy = xy[0] - self.center[0], xy[1] - self.center[1]
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        return max(self.lo[0] - lx, lx - self.hi[0], self.lo[1] - ly, ly - self.hi[1])

    def ring(self, off: float, spacing: float = 0.03) -> list[tuple[float, float]]:
        """Points every ~``spacing`` along the cavity footprint grown by ``off``."""
        x0, y0, x1, y1 = self.lo[0] - off, self.lo[1] - off, self.hi[0] + off, self.hi[1] + off
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
        local = []
        for (ax, ay), (bx, by) in itertools.pairwise(corners):
            n = max(1, round(math.hypot(bx - ax, by - ay) / spacing))
            local += [(ax + (bx - ax) * i / n, ay + (by - ay) * i / n) for i in range(n)]
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return [(self.center[0] + c * x - s * y, self.center[1] + s * x + c * y) for x, y in local]


def park_spots(
    box: BoxFootprint,
    *,
    here_xy: tuple[float, float],
    others_xy: list[tuple[float, float]],
    min_reach: float = _PARK_MIN_REACH_M,
    clearance: float = _PARK_CLEARANCE_M,
) -> list[tuple[float, float]]:
    """Free table spots to set a piece down on, best first. XY in the robot base frame.

    ``here_xy`` — right under the lifted piece — comes first: a piece lifted off the table
    goes straight back where it was, which nothing else can have taken. A piece lifted off
    the lid or out of a hole has no such spot, so a ring around the box follows, nearest
    first. A spot must stand ``clearance`` clear of the box wall and of the other pieces,
    and at least ``min_reach`` out from the robot's base; whether the arm can reach it is
    the planner's call.
    """
    clear = _BOX_WALL_M + _PIECE_MAX_RADIUS_M + clearance
    gap = 2.0 * _PIECE_MAX_RADIUS_M + clearance
    ring = sorted(box.ring(clear), key=lambda p: math.dist(p, here_xy))
    return [
        p
        for p in [tuple(here_xy), *ring]
        if box.outside_by(p) >= clear - 1e-6
        and math.hypot(*p) >= min_reach
        and all(math.dist(p, o) >= gap for o in others_xy)
    ]


def so101_ee_pose_xyzw(
    x: float,
    y: float,
    z: float,
    tilt: float = 0.0,
    roll: float = 0.0,
    *,
    pivot_xy: tuple[float, float] = (0.0, 0.0),
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a reachable SO-101 EE pose from 5-DoF task parameters.

    Position ``(x, y, z)`` is in the robot base frame. At ``tilt = roll = 0`` the
    tool frame is FLU in the arm plane: X forward, Y left, Z up.

    The arm plane's azimuth is measured about ``pivot_xy``. A tilted pose is reachable
    only about the real pan axis, :data:`_PAN_AXIS_XY`. At tilt 0 the azimuth merely
    relabels the roll, and every such pose here is built about the base origin, which is
    what the grasp and insert rolls were derived and tuned with.
    """
    from isaaclab.utils.math import quat_from_matrix

    pos = torch.tensor([x, y, z], device=device, dtype=dtype)
    rho = math.hypot(x - pivot_xy[0], y - pivot_xy[1])
    psi = 0.0 if rho < 1e-8 else math.atan2(y - pivot_xy[1], x - pivot_xy[0])

    x0 = torch.tensor([math.cos(psi), math.sin(psi), 0.0], device=device, dtype=dtype)
    z0 = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    y0 = torch.linalg.cross(z0, x0)
    y0 = y0 / y0.norm().clamp_min(1e-8)
    rot0 = torch.stack([x0, y0, z0], dim=1)

    ct, st = math.cos(tilt), math.sin(tilt)
    cr, sr = math.cos(roll), math.sin(roll)
    rot_y = torch.tensor(
        [[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]],
        device=device,
        dtype=dtype,
    )
    rot_z = torch.tensor(
        [[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    )
    rot = rot0 @ rot_y @ rot_z
    return pos, quat_from_matrix(rot.unsqueeze(0))[0].to(dtype=dtype)


@dataclass
class CuroboPolicyCfg(PolicyCfg):
    """Configure the cuRobo shape-sorting policy for SO-101 abs-joint control."""

    robot_yml: str = ""
    """Path to cuRobo ``so101.yml``. Empty uses the generated package default."""

    goal_object: str = ""
    """Optional single scene entity to grasp when ``env.cfg.shapes`` is absent."""

    place_object: str = "sorting_box"
    """Unused; place XY/Z come from ``hole_frames`` matching the current shape."""

    grasp_height_m: float = -0.002
    """Tool-frame height above the piece's origin for the grasp pose [m].

    Calibration knob: the cuRobo tool frame is the ``tcp`` link between the jaw tips (the
    fingertip collision spheres reach 3 mm below it), so this decides *where on the piece*
    the jaws grip. The default puts the fingertips 5 mm below the piece's centre — for a
    piece standing on the table, ~1 cm above the table; 1 cm lower and there is no
    collision-free solution at all. (Numbers here were tuned with the ``gripper`` link
    origin as the tool frame, 0.102 m higher: 0.10 then is -0.002 now.)

    Measured from the piece's live pose, so it holds wherever the piece lies. On the lid
    or wedged in a hole, the grasp is raised as far as it takes for the fingertips to
    clear the lid. (This replaced an absolute ``grasp_z_m=0.145``: same grasp on the
    table, but one that needed the piece's resting height on record.)

    Gripping higher is tempting — it keeps the jaws further above the lid while the piece
    is seated — but it measured worse: 0/9 failed insertions here against 3/13 at 0.004
    over 3 rollouts each (unpaired, so weak — pass ``--placement_seed`` to compare
    properly), with no clearance problem to fix at either height. The likely mechanism
    is that the piece hangs lower below the grip and tilts further when its bottom
    catches the hole rim. Raise it only together with a deeper
    :attr:`place_z_offset_m`, and re-measure."""

    approach_height_m: float = 0.04
    """Height of the pre-grasp hover above the grasp pose [m].

    Every grasp comes straight down from here, and ``UP`` replays that descent to leave,
    so the arm never plans from a pose with its fingers around a piece. Must clear the
    open jaw over the neighbours; the planner rejects a hover that does not."""

    place_z_offset_m: float = 0.02
    """Object-origin height above the matched lid hole at release [m].

    The hole frame sits on the lid top, so at the defaults this leaves the piece's bottom
    about 5 mm clear of the lid (0.020 offset − 0.015 half-height) and it drops the rest
    of the way, the rim chamfer finishing the alignment. 0.015 would put the bottom
    exactly on the lid; below that it is inside the hole at release.

    Lowering it to 0.011 was tried and measured worse — 4 failed insertions in 22
    against 1 in 19, paired on ``--placement_seed`` — because the jaws are rigid and
    position-controlled, so they jam a slightly misaligned piece onto the rim where
    dropping lets gravity correct it. Worth re-measuring now: that comparison was run
    when a failed insert ended the rollout, and jams are now recovered from rather than
    fatal, so the trade it lost on has changed."""

    place_hover_z_offset_m: float = 0.045
    """Object-origin height above the lid hole at the end of the hover move [m].

    The gap to :attr:`place_z_offset_m` is the insert descent, and ``UP`` replays it
    backwards, so this is how far the gripper lifts before anything is planned again.
    That climb-out is the point: planning straight from the release pose can start inside
    the box's collision margin, and cuRobo then fails every attempt on the spot.
    Measured free — the 25 mm default matches a no-retreat run release for release."""

    position_tolerance: float = 0.015
    """cuRobo position convergence tolerance [m]."""

    orientation_tolerance: float = 0.1
    """Orientation tolerance [rad]."""

    jaw_open: float = JAW_OPEN_RAD
    """Jaw command while approaching a piece and when releasing it [rad]."""

    jaw_closed: float = JAW_CLOSE_RAD
    """Jaw command while grasping and carrying [rad]. Below the fully-closed position on
    purpose, so the jaw keeps squeezing whatever it holds."""

    close_steps: int = 12
    """Sim steps to hold the closing jaw before checking the grasp (~0.4 s at 30 Hz)."""

    jaw_empty_tol: float = math.radians(5.0)
    """A measured jaw within this of ``jaw_closed`` means it closed on nothing [rad]."""

    nearest_first: bool = False
    """Work the pieces nearest the box first, instead of in ``env.cfg.shapes`` order.

    A fixed order shows each piece in one context only — the last one is only ever picked
    with the others already in — so a trained policy that leaves a piece behind faces a
    scene no demonstration had. Distance to the box is visible to the cameras, so the next
    piece stays a single answer (a random order would make it a coin flip at every pick);
    and the layout is random, so every piece still comes first, middle and last. Pieces
    that start on the box or knocked over go first regardless."""

    max_grasp_retries: int = 5
    """Grasp attempts per piece before it goes to the back of the queue.

    Counts attempts that actually moved; a candidate that does not plan costs nothing.
    Reset when the piece comes back round, which only happens after another piece went
    in — so it cannot loop."""

    grasp_perturb_prob: float = 0.0
    """Probability that the *first* grasp attempt on a piece is deliberately offset.

    Retries always use the true pose, so the recorded recovery is a correct
    demonstration. 0 disables perturbation (identical behaviour to before)."""

    grasp_perturb_xy_m: float = 0.02
    """Half-width of the uniform per-axis XY offset applied to a perturbed grasp [m].

    Calibration knob: too small and every attempt still succeeds, too large and the
    plan fails outright so the jaw closes in place instead of near-missing the piece.
    Check the share of perturbed attempts that plan successfully in the logs."""

    insert_settle_steps: int = 15
    """Extra open-jaw steps to wait for a released piece to land inside the box.

    The check runs once as soon as ``open_steps`` elapses, so a clean insert costs no
    extra frames. A piece that has not turned up by the end (~0.5 s at 30 Hz) is closed
    on again and recovered — or, if it was let go misaligned on purpose, left where it
    lies and grasped again from above."""

    miss_notice_delay_max_steps: int = 0
    """Max steps to keep carrying an empty gripper before noticing a missed grasp.

    The delay is sampled uniformly in ``[0, max]`` per miss, so instant retries are still
    produced. 0 keeps the instant retry only. A non-zero value makes recovery clips start
    mid-transport, which is the state a trained policy actually lands in — see
    ``training_research.local/recovery-gap.md``. Always noticed before the descent
    starts, so an empty gripper is never lowered onto anything."""

    max_insert_retries: int = 2
    """Failed inserts per piece before it is parked and sent to the back of the queue.

    A recovery is lift → re-aim (or park and re-grasp level) → place again. The piece
    gets a fresh budget when it comes back round, which only happens after another piece
    went in, so it cannot loop. 0 means one attempt and no recovery."""

    insert_align_xy_tol_m: float = 0.005
    """Max piece-to-hole XY offset that still counts as insertable [m].

    Calibration knob, checked while the piece is still gripped. Measured on two headless
    runs: 16 clean inserts released at 0.0-5.0 mm, and a cylinder released at 6.7 mm did
    *not* fall in — so the earlier 8 mm let a sure failure through to the slower
    post-release recovery. The limit for the cylinder sits somewhere in 5-6.7 mm; 5 mm
    errs toward re-aiming, which is the cheap mistake here (a re-aim, against a regrip
    from the lid). Every insert logs its alignment, so each run adds to the sample."""

    insert_align_yaw_tol_rad: float = math.radians(10.0)
    """Max piece-to-hole yaw error, folded into the piece's symmetry [rad].

    Calibration knob. A 30 mm square in a 3 mm-clearance hole binds at about 13°
    (``s*(cos+sin) <= s + 2c``), so the default is just inside the geometric limit.
    Meaningless for the cylinder, which reports 0 yaw error by construction."""

    insert_align_tilt_tol_rad: float = math.radians(20.0)
    """Tilt off vertical the arm can no longer work around [rad].

    Guessed, not derived — a real number would need the rim chamfer, the drop height and
    the piece silhouette. Deliberately generous: a slightly tipped piece still drops in,
    and levelling one takes a set-down and a regrasp, so a tight value only buys
    pointless parking. One number for every question that reduces to it:

    * end of the insert descent — is the held piece too tilted to let go of?
    * ``_decide`` while holding — can this grasp be re-aimed (under) or must the piece be
      put down and picked up level (over)? Tilt is the only error re-aiming cannot fix.
    * grasp candidates — which of a tipped piece's faces are still vertical enough to
      squeeze?
    """

    place_perturb_prob: float = 0.0
    """Probability that the *first* placement of a piece is deliberately misaligned.

    The counterpart of :attr:`grasp_perturb_prob` for inserts. Retries always use the
    true pose, so the recorded recovery is a correct demonstration. 0 disables it."""

    place_perturb_xy_m: float = 0.012
    """Half-width of the uniform per-axis XY offset on a perturbed placement [m]."""

    place_perturb_yaw_rad: float = math.radians(45.0)
    """Half-width of the uniform yaw offset on a perturbed placement [rad].

    Yaw is the failure a trained policy actually produces on non-circular pieces, and a
    cube arriving 45° off its hole is the case its recovery most needs to have seen — so
    the default spans the cube's whole folded range. Sampling uniformly rather than at
    the bound keeps a mix of near misses (which still drop in) and clear ones."""

    place_release_misaligned_prob: float = 0.0
    """Probability that a placement caught misaligned is let go anyway, then recovered from above.

    A trained policy plays its action chunk open-loop: it opens the jaw wherever the descent
    ends and climbs out, as every clean insert taught it. What it then has to recover from is
    an open jaw above a piece lying on the lid or wedged in its hole — not a misaligned piece
    still in the jaws, which is all lifting it back out records. With this, the piece is let
    go, the arm climbs out with the jaw open and ``_decide`` grasps it where it settled. The
    release itself is cut, so only the climb-out and the recovery are recorded. Pair it with
    :attr:`place_perturb_prob`, which makes the misaligned placements. 0 disables it.

    Only while the piece has a regrasp and another insert left in its budget; past that it
    is lifted back out and parked as before. Otherwise ``_decide`` would defer a piece left
    lying on the box, and record walking away from it."""

    open_steps: int = 12
    """Sim steps to hold the opening jaw before checking the release (~0.4 s at 30 Hz)."""

    home_steps: int = 18
    """Sim steps to cosine-interpolate to the home joint pose (~0.6 s at 30 Hz)."""

    use_cuda_graph: bool = False
    """Enable cuRobo CUDA graphs."""

    self_collision_check: bool = False
    """Enable self-collision costs."""

    optimizer_collision_activation_distance: float = 0.03
    """Soft collision-cost activation distance [m]; larger keeps more clearance."""

    waypoint_stride: int = 3
    """Play every N-th interpolated waypoint (1 = all).

    One waypoint is consumed per env step, so this sets the arm's wall-clock speed:
    ``interpolation_dt (0.025 s) * stride / step_dt``. At the default 30 Hz env rate,
    stride 3 plays the plan at ~2.25x its planned speed — matching the 2.5x that stride 2
    gave at 50 Hz. Raise it with the env rate, or the demos stretch out and every episode
    carries more near-duplicate frames."""

    debug_viz: bool = False
    """Draw goal + EE frame markers in the Kit viewport."""

    debug_viser: bool = False
    """Open a Viser page with collision meshes + robot spheres."""

    debug_viser_port: int = 8080
    """Viser HTTP port when ``debug_viser`` is enabled."""

    marker_frame_scale: float = 0.08
    """World-scale of the frame marker axes [m]."""


@register_policy
class CuroboPolicy(PolicyBase[CuroboPolicyCfg]):
    """Sort every ``env.cfg.shapes`` piece into its hole, recovering from its own misses."""

    name = "curobo_reach"

    def __init__(self, config: CuroboPolicyCfg):
        super().__init__(config)
        self._motion: MotionClient | None = None
        self._kit_markers: KitFrameMarkers | None = None
        self._demo_events: list[str] = []
        self.reset()

    # ------------------------------------------------------------------
    # PolicyBase
    # ------------------------------------------------------------------

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._motion is not None:
            self._motion.detach()
            self._motion.clear_last_goal()
        # The arm: what it is doing right now. UP with nothing to replay waits for the
        # pieces to land and goes to _decide, which is exactly what the first step of an
        # episode needs.
        self._phase = Phase.UP
        self._goal = Goal.GRASP
        self._traj: torch.Tensor | None = None
        self._step_idx = 0
        self._descent_start = 0
        self._phase_steps = 0
        self._jaw_cmd = float(self.config.jaw_open)
        self._hold_action: torch.Tensor | None = None
        # The task: which pieces are left, and how the current one is going.
        self._shapes: list[ShapeInfo] | None = None
        self._todo: list[ShapeInfo] | None = None
        self._stalled = 0
        self._grasp_attempts = 0
        self._insert_attempts = 0
        self._miss_cut_in: int | None = None
        # Recording facts about the current attempt. "Already cut" starts true: there is no
        # attempt yet, so nothing to cut.
        self._attempt_cut = True
        self._grasp_tilt = 0.0
        self._let_go_misaligned = False
        # policy_runner never drains these; clear so they cannot leak across episodes.
        self._demo_events.clear()

    def is_demonstration_ended(self) -> bool:
        """True after the scripted sequence has finished (HOME → DONE)."""
        return self._phase is Phase.DONE

    def pop_demo_events(self) -> list[str]:
        """Return and clear the recording events raised during the last ``get_action``.

        "checkpoint" — frames recorded so far are verified, keep them.
        "cut" — drop back to the last checkpoint; this step starts a new clip.
        """
        events, self._demo_events = self._demo_events, []
        return events

    def close(self) -> None:
        if self._motion is not None:
            self._motion.close()
            self._motion = None
        if self._kit_markers is not None:
            self._kit_markers.close()
            self._kit_markers = None
        self._traj = None
        self._shapes = None

    def get_action(self, env: gym.Env, observation: GymSpacesDict) -> torch.Tensor:
        device = torch.device(env.unwrapped.device)
        num_envs = env.unwrapped.num_envs
        action_dim = env.action_space.shape[-1]
        if action_dim != len(SIM_JOINT_NAMES):
            raise RuntimeError(
                f"CuroboPolicy expects so101_abs_joint (action dim {len(SIM_JOINT_NAMES)}), "
                f"got action dim {action_dim}. Pass `--embodiment so101_abs_joint`."
            )

        motion = self._ensure_motion(device)
        step = _PHASE_STEPS.get(self._phase)
        if step is not None:
            action = getattr(self, step)(env, device, motion)
        else:  # DONE
            action = self._hold(env, device, motion)

        self._update_kit_markers(env, device, motion)
        return action.unsqueeze(0).expand(num_envs, -1).contiguous()

    # ------------------------------------------------------------------
    # Motions — each plays one thing and hands over. Only _decide chooses.
    # ------------------------------------------------------------------

    def _start_go(self, env: gym.Env, device: torch.device, motion: MotionClient) -> torch.Tensor:
        self._phase = Phase.GO
        return self._step_go(env, device, motion)

    def _step_go(self, env: gym.Env, device: torch.device, motion: MotionClient) -> torch.Tensor:
        """Play the planned approach — hover move, then descent — and act on arrival."""
        assert self._traj is not None
        if self._miss_cut_in is not None:
            # Carrying a missed piece on purpose. Notice before the descent starts, so an
            # empty gripper is never lowered onto the box or the table.
            if self._miss_cut_in <= 0 or self._step_idx >= self._descent_start - 1:
                self._miss_cut_in = None
                self._jaw_cmd = float(self.config.jaw_open)  # known empty from here on
                self._cut("GO: noticed the empty gripper mid-transport")
                return self._decide(env, device, motion)
            self._miss_cut_in -= 1

        carrying = self._goal is not Goal.GRASP
        action = self._playback_traj(
            device, motion, jaw=self.config.jaw_closed if carrying else self.config.jaw_open
        )
        if action is not None:
            return action

        if self._goal is Goal.GRASP:
            return self._start_jaw(env, device, motion, self.config.jaw_closed)
        if self._goal is Goal.INSERT:
            # Last look before letting go, while the piece can still be taken back out.
            reason = self._insert_misaligned(env, device)
            release_anyway = float(self.config.place_release_misaligned_prob)
            in_budget = (  # after this failure: else _decide defers it where it lies
                self._insert_attempts < max(0, int(self.config.max_insert_retries))
                and self._grasp_attempts < max(1, int(self.config.max_grasp_retries))
            )
            if (
                reason is not None and in_budget
                and release_anyway > 0.0 and random.random() < release_anyway
            ):
                # What a trained policy does: let go wherever the descent ended. Not cut yet —
                # _after_release cuts once the release is over, so it is never imitated.
                print(f"[CuroboPolicy] GO: {reason} — letting go anyway, on purpose.")
                self._let_go_misaligned = True
            elif reason is not None:
                self._insert_failed(f"GO: {reason}")
                return self._start_up(env, device, motion)
        return self._start_jaw(env, device, motion, self.config.jaw_open)

    def _start_jaw(
        self, env: gym.Env, device: torch.device, motion: MotionClient, jaw: float
    ) -> torch.Tensor:
        self._phase = Phase.JAW
        self._jaw_cmd = float(jaw)
        self._phase_steps = 0
        return self._step_jaw(env, device, motion)

    def _step_jaw(self, env: gym.Env, device: torch.device, motion: MotionClient) -> torch.Tensor:
        """Hold the arm still while the jaw moves, then see what that achieved."""
        opening = self._jaw_cmd == float(self.config.jaw_open)
        if opening and self._phase_steps == 0:
            motion.detach()  # whatever was held is part of the world again
        action = self._hold_with_jaw(device, self._jaw_cmd)
        self._phase_steps += 1
        steps = self.config.open_steps if opening else self.config.close_steps
        if self._phase_steps < max(1, int(steps)):
            return action
        if opening:
            return self._after_release(env, device, motion, action)
        return self._after_close(env, device, motion)

    def _after_close(self, env: gym.Env, device: torch.device, motion: MotionClient) -> torch.Tensor:
        """The jaw has closed — on the piece, or on nothing."""
        shape = self._current_shape(env)
        jaw = float(motion.measured_jaw(env, device))
        if not self._jaw_is_fully_closed(jaw):
            print(f"[CuroboPolicy] JAW closed on {shape.name} (jaw={jaw:.3f}) → UP.")
            return self._start_up(env, device, motion)

        if self._goal is Goal.GRASP:
            delay = self._sample_miss_notice_delay()
            print(
                f"[CuroboPolicy] JAW closed on nothing (jaw={jaw:.3f}) — missed {shape.name}, "
                f"attempt {self._grasp_attempts}/{self.config.max_grasp_retries}"
                + (f"; carrying on for {delay} step(s) before noticing." if delay else ".")
            )
            if delay > 0:
                # Pretend it worked, so the recovery clip starts mid-transport — the state a
                # trained policy that missed actually finds itself in. Attach the piece here,
                # at the bottom: its offset is then what a real grasp would have measured, so
                # the empty carry flies at the height of a real one instead of 4 cm above it.
                self._miss_cut_in = delay
                motion.attach(env, device, shape.name)
                return self._start_up(env, device, motion)
            self._cut(f"JAW: missed {shape.name}")
        else:
            # A regrip after a failed drop, and the piece had already slid out of reach.
            # Cut again so closing on air is not what the next clip confirms; it starts from
            # the empty jaw at the box instead, which is a state worth recovering from.
            self._cut(f"JAW: regrip closed on nothing — {shape.name} is out of reach")
        self._jaw_cmd = float(self.config.jaw_open)
        return self._start_up(env, device, motion)

    def _after_release(
        self, env: gym.Env, device: torch.device, motion: MotionClient, action: torch.Tensor
    ) -> torch.Tensor:
        """The jaw has opened. Only an insert has anything to verify."""
        if self._goal is not Goal.INSERT:
            return self._start_up(env, device, motion)
        inserted = self._piece_inserted(env)
        if not inserted:
            deadline = max(1, int(self.config.open_steps)) + max(
                1, int(self.config.insert_settle_steps)
            )
            if self._phase_steps < deadline:
                return action
            # Inside the cavity but still bouncing is a slow success, not a failure.
            inserted = self._piece_inserted(env, require_settled=False)
        name = self._current_shape(env).name
        if inserted:
            if self._let_go_misaligned:
                # It went in anyway, but letting go there was still the mistake: keep the
                # climb-out, not the release that happened to work.
                self._cut(f"JAW: {name} dropped in although it was let go misaligned")
            return self._start_up(env, device, motion)
        if self._let_go_misaligned:
            # Leave it where it settled and climb out with the jaw open, as a trained policy
            # would; _decide then grasps it where it lies. The cut lands here, after the settle
            # wait, so the recovery clip starts after the release rather than with it.
            self._insert_failed(f"JAW: {name} was let go misaligned and is not in the box")
            return self._start_up(env, device, motion)
        # It did not fall in, and it is still between the open jaws: close on it again
        # rather than walk away. That needs no plan, and nothing else is reachable from
        # down here anyway.
        self._insert_failed(
            f"JAW: {name} is not in the box after {self.config.insert_settle_steps} settle step(s)"
        )
        return self._start_jaw(env, device, motion, self.config.jaw_closed)

    def _start_up(self, env: gym.Env, device: torch.device, motion: MotionClient) -> torch.Tensor:
        self._phase = Phase.UP
        self._phase_steps = 0
        return self._step_up(env, device, motion)

    def _step_up(self, env: gym.Env, device: torch.device, motion: MotionClient) -> torch.Tensor:
        """Back out along the descent, the way the arm came in, then decide.

        A replay, not a plan: at the bottom the jaws straddle a piece or sit inside the
        lid's collision margin, and cuRobo refuses to plan out of a start state like that —
        which is how the arm used to freeze after a release. With no descent to undo (the
        first step of an episode, or letting go in mid-air) this goes straight on.

        Then it holds until nothing is moving: pieces spawn 1 cm up and drop, one let go
        of on the lid tips into its final pose, and a decision about either before it
        lands would plan a grasp around where the piece was.
        """
        if self._traj is None or self._step_idx <= self._descent_start:
            # The first step of an episode always holds: every piece was just teleported
            # into place and reads zero velocity, mid-air or not.
            first_step = self._hold_action is None
            if first_step or (self._phase_steps < _SETTLE_MAX_STEPS and not self._world_still(env)):
                self._phase_steps += 1
                return self._hold(env, device, motion)
            return self._decide(env, device, motion)
        action = self._playback_traj_reversed(device, motion, jaw=self._jaw_cmd)
        assert action is not None  # _step_idx > _descent_start >= 0
        return action


    # ------------------------------------------------------------------
    # Home and trajectory playback
    # ------------------------------------------------------------------

    def _step_home(
        self, env: gym.Env, device: torch.device, motion: MotionClient
    ) -> torch.Tensor:
        if self._traj is None:
            self._build_home_traj(env, device, motion)

        assert self._traj is not None
        if self._step_idx >= self._traj.shape[0]:
            print("[CuroboPolicy] HOME done → DONE.")
            self._phase = Phase.DONE
            assert self._hold_action is not None
            return self._hold_action

        action = self._traj[self._step_idx].to(device=device, dtype=torch.float32)
        self._step_idx += 1
        self._hold_action = action
        return action

    def _build_home_traj(
        self, env: gym.Env, device: torch.device, motion: MotionClient
    ) -> None:
        """Cosine ease-in/out from current joints to home (all 6 DoF)."""
        if self._hold_action is not None:
            start = self._hold_action.detach().to(device=device, dtype=torch.float32)
        else:
            start = motion.joint_action(env, device)

        goal = torch.tensor(_HOME_JOINT_RAD, device=device, dtype=torch.float32)
        n = max(1, int(self.config.home_steps))
        t = torch.linspace(1.0 / n, 1.0, n, device=device, dtype=torch.float32)
        alpha = (1.0 - torch.cos(t * math.pi)) * 0.5
        self._traj = (start.unsqueeze(0) + alpha.unsqueeze(1) * (goal - start).unsqueeze(0)).contiguous()
        self._step_idx = 0
        print(f"[CuroboPolicy] HOME interpolating {n} steps to home joint pose.")

    def _playback_traj(
        self, device: torch.device, motion: MotionClient, *, jaw: float
    ) -> torch.Tensor | None:
        """Advance one waypoint, or None when the trajectory is finished."""
        assert self._traj is not None
        if self._step_idx >= self._traj.shape[0]:
            return None
        q = self._traj[self._step_idx]
        self._step_idx += 1
        motion.notify_joint(q)
        action = motion.q_to_action(q, device, jaw=jaw)
        self._hold_action = action
        return action

    def _playback_traj_reversed(
        self, device: torch.device, motion: MotionClient, *, jaw: float
    ) -> torch.Tensor | None:
        """Walk the current trajectory back toward its start, or None once past it."""
        assert self._traj is not None
        self._step_idx -= 1
        if self._step_idx < 0:
            return None
        q = self._traj[self._step_idx]
        motion.notify_joint(q)
        action = motion.q_to_action(q, device, jaw=jaw)
        self._hold_action = action
        return action

    def _hold(self, env: gym.Env, device: torch.device, motion: MotionClient) -> torch.Tensor:
        """Stay where the last action put the arm (or where it is, before any action)."""
        if self._hold_action is None:
            self._hold_action = motion.joint_action(env, device)
        return self._hold_action

    def _hold_with_jaw(self, device: torch.device, jaw: float) -> torch.Tensor:
        assert self._hold_action is not None
        action = self._hold_action.clone()
        action[_JAW_INDEX] = float(jaw)
        self._hold_action = action
        return action

    def _jaw_is_fully_closed(self, jaw: float | torch.Tensor) -> bool:
        return float(jaw) <= float(self.config.jaw_closed) + float(self.config.jaw_empty_tol)

    # ------------------------------------------------------------------
    # Decisions — the only place that chooses what happens next
    # ------------------------------------------------------------------

    def _decide(self, env: gym.Env, device: torch.device, motion: MotionClient) -> torch.Tensor:
        """Pick the next motion from the world. Runs only with the arm clear of contact."""
        self._traj = None
        pretending = self._miss_cut_in is not None
        holding = pretending or self._holding(env, device, motion)
        name = self._current_shape(env).name
        if not pretending and not holding and self._jaw_cmd == float(self.config.jaw_closed):
            # The jaw was closed on the piece and now holds nothing: it slipped out on the
            # way up — after a grasp, a re-aim or a regrip alike. Cut even if this attempt
            # was cut already: the lift that lost it is a new mistake after that cut.
            self._cut(f"decide: {name} slipped out of the jaws")
        elif self._goal is Goal.GRASP and holding and not pretending and not self._attempt_cut:
            # Tipping an upright piece in the jaws is a bad grasp. Tipped *before* the grasp
            # (perched on the lid) is different: then holding it at all is the recovery.
            tol = float(self.config.insert_align_tilt_tol_rad)
            if self._piece_tilt(env, device) > tol >= self._grasp_tilt:
                self._cut(f"decide: the grasp of {name} tipped it in the jaws")
        if holding:
            return self._decide_holding(env, device, motion)
        return self._decide_empty(env, device, motion)

    def _decide_holding(
        self, env: gym.Env, device: torch.device, motion: MotionClient
    ) -> torch.Tensor:
        shape = self._current_shape(env)
        if not self._has_box(env):
            # --goal_object smoke test: no box, so picking it up was the whole test.
            print(f"[CuroboPolicy] decide: grasped {shape.name}, and there is no box → HOME.")
            self._finish_piece(shape)
            self._phase = Phase.HOME
            return self._step_home(env, device, motion)
        if self._miss_cut_in is not None:
            # A pretended grasp (attached at the bottom, in _after_close): carry the
            # "piece" toward its hole until GO notices. Tilt, budgets and parking mean
            # nothing for a piece that is not really there.
            if self._plan_go(env, device, motion, Goal.INSERT):
                return self._start_go(env, device, motion)
            self._miss_cut_in = None
            self._jaw_cmd = float(self.config.jaw_open)
            self._cut("decide: nowhere to carry the pretended grasp — noticing the miss now")
            return self._decide_empty(env, device, motion)

        # Re-measured every time: the piece may have moved in the jaws since the grasp — a
        # failed insert means exactly that — and aiming from a stale offset repeats the
        # same miss. After a fresh grasp it is simply the first measurement.
        motion.attach(env, device, shape.name)
        tilt = self._piece_tilt(env, device)
        goal = next_goal(
            holding=True,
            upright=tilt <= float(self.config.insert_align_tilt_tol_rad),
            insert_tries_left=self._insert_tries_left(),
        )
        if goal == "insert" and self._plan_go(env, device, motion, Goal.INSERT):
            return self._start_go(env, device, motion)

        why = "no insert plan" if goal == "insert" else (
            f"tilted {math.degrees(tilt):.0f}°" if tilt > self.config.insert_align_tilt_tol_rad
            else "out of insert tries"
        )
        if goal == "insert":
            # Lifted it and cannot place it: putting it back down is not a demonstration.
            self._cut(f"decide: no insert plan for {shape.name}")
        print(f"[CuroboPolicy] decide: parking {shape.name} ({why}).")
        if self._plan_go(env, device, motion, Goal.PARK):
            return self._start_go(env, device, motion)
        self._cut(f"decide: nowhere to put {shape.name} down — letting go here")
        self._goal = Goal.PARK
        return self._start_jaw(env, device, motion, self.config.jaw_open)

    def _decide_empty(
        self, env: gym.Env, device: torch.device, motion: MotionClient
    ) -> torch.Tensor:
        motion.detach()  # nothing in the jaws: every piece is an obstacle again
        todo = self._pending(env)
        while todo:
            shape = todo[0]
            in_box = self._piece_inserted(env, require_settled=False)
            goal = next_goal(
                holding=False,
                in_box=in_box,
                insert_tries_left=self._insert_tries_left(),
                grasp_tries_left=self._grasp_attempts < max(1, int(self.config.max_grasp_retries)),
            )
            if goal == "finish":
                self._finish_piece(shape)
                continue
            if goal == "grasp" and self._plan_go(env, device, motion, Goal.GRASP):
                self._grasp_attempts += 1
                return self._start_go(env, device, motion)
            why = "no grasp plan" if goal == "grasp" else "out of tries"
            if not self._defer_piece(shape, why):
                break

        if not todo:
            print("[CuroboPolicy] decide: every piece is in the box → HOME.")
            self._phase = Phase.HOME
            return self._step_home(env, device, motion)
        print(
            f"[CuroboPolicy] decide: {len(todo)} piece(s) left and none workable since the "
            f"last one went in → DONE."
        )
        self._phase = Phase.DONE
        return self._hold(env, device, motion)

    def _finish_piece(self, shape: ShapeInfo) -> None:
        """The current piece is in the box: confirm the recording and move on."""
        assert self._todo is not None and self._todo[0] is shape
        self._todo.pop(0)
        # Confirmed here, with the arm clear of the box, which is also where the next clip
        # has to start from.
        self._demo_events.append("checkpoint")
        self._stalled = 0
        self._grasp_attempts = self._insert_attempts = 0
        print(f"[CuroboPolicy] decide: {shape.name} is in the box ({len(self._todo)} left).")

    def _defer_piece(self, shape: ShapeInfo, reason: str) -> bool:
        """Send the current piece to the back of the queue.

        False once every remaining piece has been deferred since the last one went in:
        nothing about the scene has changed since, so trying again would repeat itself.
        """
        assert self._todo is not None and self._todo[0] is shape
        if self._grasp_attempts or self._insert_attempts:
            # It was tried and it did not work out: whatever was recorded since the last
            # cut (lifting it back out, parking it) leads nowhere, and "put it down and
            # walk away" would contradict the recoveries recorded from the same states.
            self._cut(f"decide: giving up on {shape.name} for now")
        self._todo.append(self._todo.pop(0))
        self._stalled += 1
        self._grasp_attempts = self._insert_attempts = 0
        print(
            f"[CuroboPolicy] decide: deferring {shape.name} ({reason}); "
            f"{self._stalled}/{len(self._todo)} deferred since the last insert."
        )
        return self._stalled < len(self._todo)

    def _insert_failed(self, reason: str) -> None:
        """Book a placement that will not end in the box: cut the clip, spend a try.

        The physical recovery always runs — it has to, the piece is in or over the box.
        The budget is enforced by ``_decide``, which parks the piece once it is spent.
        """
        self._insert_attempts += 1
        self._cut(
            f"{reason} — insert recovery "
            f"{self._insert_attempts}/{self.config.max_insert_retries}"
        )

    def _cut(self, reason: str) -> None:
        """Drop everything recorded since the last checkpoint; this step starts a new clip."""
        self._demo_events.append("cut")
        self._attempt_cut = True
        print(f"[CuroboPolicy] {reason} → cut.")

    def _insert_tries_left(self) -> bool:
        return self._insert_attempts <= max(0, int(self.config.max_insert_retries))

    def _holding(self, env: gym.Env, device: torch.device, motion: MotionClient) -> bool:
        """Something is between the closed jaws — measured, not remembered."""
        if self._jaw_cmd != float(self.config.jaw_closed):
            return False
        return not self._jaw_is_fully_closed(motion.measured_jaw(env, device))

    def _pending(self, env: gym.Env) -> list[ShapeInfo]:
        """Pieces not yet in the box, current one first. Built on first use."""
        if self._todo is None:
            cfg_shapes = getattr(env.unwrapped.cfg, "shapes", None)
            if cfg_shapes:
                self._todo = list(cfg_shapes)
                if self._has_box(env):
                    # Whatever starts on the box — stuck in its hole, fallen onto the lid —
                    # or knocked over goes first: it is what a policy that let go in the
                    # wrong place faces next, and left for later it gets knocked about by
                    # the other inserts (live, a lying cylinder brushed by the arm rolled
                    # out of reach). Once, here: re-sorting at every decision would undo
                    # deferrals.
                    device = torch.device(env.unwrapped.device)
                    first = {
                        s.name for s in self._todo
                        if self._on_the_box(env, device, s.name) or self._tipped(env, device, s.name)
                    }
                    # Keys up front: list.sort() empties the list while it runs, so a key that
                    # looks at the queue (the current piece, its hole) would find it empty.
                    keys = {
                        s.name: (
                            s.name not in first,
                            self._box_distance(env, device, s.name) if self.config.nearest_first else 0.0,
                        )
                        for s in self._todo
                    }
                    self._todo.sort(key=lambda s: keys[s.name])
            elif self.config.goal_object:
                self._todo = [ShapeInfo(prim_path=f"{{ENV_REGEX_NS}}/{self.config.goal_object}")]
            else:
                raise ValueError(
                    "CuroboPolicy needs env.cfg.shapes (from ShapeSortingEnvironment) "
                    "or --goal_object <scene_entity_name>."
                )
        return self._todo

    def _current_shape(self, env: gym.Env) -> ShapeInfo:
        return self._pending(env)[0]

    # ------------------------------------------------------------------
    # Planning — whole approaches, best candidate first
    # ------------------------------------------------------------------

    def _plan_go(
        self, env: gym.Env, device: torch.device, motion: MotionClient, goal: Goal
    ) -> bool:
        """Plan the hover move and the descent for the first candidate where both work.

        Both legs plan before anything moves, so a candidate can never strand the arm
        halfway, and the descent is planned from the hover plan's own end state. For an
        insert the box is hidden for the descent only: its last few mm sit inside the
        planner's collision margin around the lid, where the optimiser would otherwise
        buy clearance by drifting off the hole axis.
        """
        make = {
            Goal.GRASP: self._grasp_candidates,
            Goal.INSERT: self._insert_candidates,
            Goal.PARK: self._park_candidates,
        }[goal]
        candidates = make(env, device, motion)
        hide = self._box_entity_name(env) if goal is Goal.INSERT else None
        shape = self._current_shape(env)
        for i, (label, hover, contact) in enumerate(candidates):
            tag = f"{goal.name} {shape.name} [{i + 1}/{len(candidates)} {label}]"
            move = motion.plan_to_pose(env, device, *hover, label=f"{tag} hover")
            if not move.success:
                continue
            with motion.obstacles_hidden(hide):
                descent = motion.plan_to_pose(
                    env, device, *contact, label=f"{tag} descent", start=move.waypoints[-1]
                )
            if not descent.success:
                continue
            self._goal = goal
            self._attempt_cut = False
            self._let_go_misaligned = False
            self._traj = torch.cat([move.waypoints, descent.waypoints])
            self._descent_start = int(move.waypoints.shape[0])
            self._step_idx = 0
            print(f"[CuroboPolicy] {tag}: planned.")
            return True
        print(
            f"[CuroboPolicy] {goal.name} {shape.name}: none of {len(candidates)} "
            f"candidate(s) planned."
        )
        return False

    def _grasp_candidates(
        self, env: gym.Env, device: torch.device, motion: MotionClient
    ) -> list[Candidate]:
        """One grasp per jaw yaw that squeezes two flats of the piece as it lies now."""
        shape = self._current_shape(env)
        xyz, quat_xyzw = self._piece_pose(env, device)
        px, py = float(xyz[0]), float(xyz[1])
        z = float(xyz[2]) + float(self.config.grasp_height_m)
        box = self._box_footprint(env, device) if self._has_box(env) else None
        if box is not None and box.outside_by((px, py)) < 0.0:
            # On the lid or wedged in a hole: grip it higher up rather than put the
            # fingertips through the lid, which no candidate would plan.
            z = max(z, box.lid_z + _FINGERTIP_BELOW_TOOL_M + _LID_FINGERTIP_CLEARANCE_M)

        self._grasp_tilt = _tilt_of(quat_xyzw)
        radial = math.atan2(py, px)
        yaws = grasp_jaw_yaws(
            quat_xyzw,
            self._face_normals(shape),
            radial_yaw=radial,
            max_tilt=float(self.config.insert_align_tilt_tol_rad),
        )
        out = [
            self._grasp_candidate(px, py, z, yaw, f"jaw {math.degrees(_wrap_to_pi(yaw - radial)):+.0f}°", device)
            for yaw in yaws
        ]
        dx, dy = self._sample_grasp_offset()
        if (dx, dy) != (0.0, 0.0) and yaws:
            # First in line, with the true candidates behind it: a perturbed grasp that
            # does not plan falls back to a correct one instead of deferring the piece.
            print(f"[CuroboPolicy] GRASP perturbed on purpose by ({dx:+.3f}, {dy:+.3f}) m.")
            out.insert(0, self._grasp_candidate(px + dx, py + dy, z, yaws[0], "perturbed", device))
        return out

    def _grasp_candidate(
        self, px: float, py: float, z: float, jaw_yaw: float, label: str, device: torch.device
    ) -> Candidate:
        tool_x, tool_y, roll = grasp_tool_xy_roll(px, py, jaw_yaw, _GOAL_XY_STANDOFF_M)
        contact = so101_ee_pose_xyzw(tool_x, tool_y, z, tilt=_GOAL_TILT_RAD, roll=roll, device=device)
        hover = so101_ee_pose_xyzw(
            tool_x, tool_y, z + float(self.config.approach_height_m),
            tilt=_GOAL_TILT_RAD, roll=roll, device=device,
        )
        return label, hover, contact

    def _insert_candidates(
        self, env: gym.Env, device: torch.device, motion: MotionClient
    ) -> list[Candidate]:
        """One placement over the hole per roll that lines the held piece up with it."""
        hole, _ = self._hole_pose_in_robot_base(env, device)
        rolls = self._place_roll_candidates(env, device, motion)

        def candidate(label: str, base: torch.Tensor, roll: float) -> Candidate:
            return (
                label,
                self._place_pose_in_robot_base(
                    env, device, motion, roll=roll,
                    obj_desired=_raised(base, self.config.place_hover_z_offset_m),
                ),
                self._place_pose_in_robot_base(
                    env, device, motion, roll=roll,
                    obj_desired=_raised(base, self.config.place_z_offset_m),
                ),
            )

        out = [candidate(f"roll {math.degrees(r):+.0f}°", hole, r) for r in rolls]
        dx, dy, dyaw = self._sample_place_perturb()
        if (dx, dy, dyaw) != (0.0, 0.0, 0.0) and rolls:
            # First in line, with the true candidates behind it — same as the grasp: a
            # perturbed placement that does not plan falls back to a correct one instead
            # of parking a piece that was held perfectly well.
            print(
                f"[CuroboPolicy] INSERT perturbed on purpose by ({dx:+.3f}, {dy:+.3f}) m "
                f"and {math.degrees(dyaw):+.0f}° — expect an insert recovery."
            )
            shifted = hole.clone()
            shifted[0] += dx
            shifted[1] += dy
            out.insert(0, candidate("perturbed", shifted, _wrap_to_pi(rolls[0] + dyaw)))
        return out

    def _park_candidates(
        self, env: gym.Env, device: torch.device, motion: MotionClient
    ) -> list[Candidate]:
        """Down onto the best few free table spots (:func:`park_spots`), standing up.

        A tipped piece is first offered to the wrist: tilted so the piece comes down
        upright (:func:`upright_tilt_rolls`), then leaning by ``_SETDOWN_LEAN_RAD`` for
        less wrist tilt. Last, the gripper upright, a few rolls — the piece comes down as
        tipped as it is held and gravity has to level it, which works under ~45° only.
        Any orientation at any spot beats the next orientation at the best spot.
        """
        if not self._has_box(env):
            return []
        box = self._box_footprint(env, device)
        todo = self._pending(env)
        here = entity_position_in_robot_base(env, todo[0].name, device=device)
        here_xy = (float(here[0]), float(here[1]))
        others = [entity_position_in_robot_base(env, s.name, device=device) for s in todo[1:]]
        others_xy = [(float(o[0]), float(o[1])) for o in others]

        held_tilt = self._piece_tilt(env, device)
        ways: list[tuple[str, float, float, float]] = []  # label, tool tilt, roll, piece tilt
        if held_tilt > float(self.config.insert_align_tilt_tol_rad):
            a = self._piece_axis_in_tool(env, device, motion)
            slack = float(self.config.orientation_tolerance)  # what the planner may leave
            for lean, how in ((0.0, "upright"), (_SETDOWN_LEAN_RAD, "leaning")):
                ways += [
                    (f"{how} tilt {math.degrees(t):+.0f}° roll {math.degrees(r):+.0f}°", t, r, lean + slack)
                    for t, r in upright_tilt_rolls(a, lean=lean, max_tilt=_SETDOWN_MAX_TILT_RAD)
                    if t < 0.0  # fingertips away from the base: the other way never plans
                ]
        ways += [
            (f"as held roll {math.degrees(r):+.0f}°", _GOAL_TILT_RAD, r, held_tilt)
            for r in (_GOAL_ROLL_RAD, math.pi / 2.0, -math.pi / 2.0)
        ]

        h, rad = _PIECE_HALF_HEIGHT_M, _PIECE_MAX_RADIUS_M
        out: list[Candidate] = []
        for label, tool_tilt, roll, piece_tilt in ways:
            reach, drift = (_SETDOWN_MIN_REACH_M, _SETDOWN_DRIFT_M) if tool_tilt else (_PARK_MIN_REACH_M, 0.0)
            spots = park_spots(
                box, here_xy=here_xy, others_xy=others_xy,
                min_reach=reach, clearance=_PARK_CLEARANCE_M + drift,
            )
            # Tipped by t, a piece's lowest bottom corner sits h·cos t + R·sin t below its
            # origin instead of h: release that much higher, so it drops a few mm onto an
            # edge and falls level rather than being pressed into the table.
            lift = _PARK_RELEASE_CLEARANCE_M + max(
                0.0, h * math.cos(piece_tilt) + rad * math.sin(piece_tilt) - h
            )
            for i, (x, y) in enumerate(spots[:_PARK_MAX_SPOTS]):
                base = torch.tensor([x, y, box.table_z + h], device=device, dtype=torch.float32)
                where = "where it was" if (x, y) == here_xy else f"spot {i + 1}"
                out.append((
                    f"{where} {label}",
                    self._place_pose_in_robot_base(
                        env, device, motion, roll=roll, tilt=tool_tilt,
                        obj_desired=_raised(base, self.config.place_hover_z_offset_m),
                    ),
                    self._place_pose_in_robot_base(
                        env, device, motion, roll=roll, tilt=tool_tilt,
                        obj_desired=_raised(base, lift),
                    ),
                ))
        return out

    def _piece_axis_in_tool(
        self, env: gym.Env, device: torch.device, motion: MotionClient
    ) -> torch.Tensor:
        """The held piece's own +Z in the tool frame — what the grip fixed, measured now."""
        from isaaclab.utils.math import convert_quat

        ee = motion.ee_pose(motion.planner_joint_state(env, device))
        rot_ee = _rot_from_quat_xyzw(convert_quat(ee.quaternion.view(-1)[:4], to="xyzw"))
        return rot_ee.T @ _rot_from_quat_xyzw(self._piece_pose(env, device)[1])[:, 2]

    def _face_normals(self, shape: ShapeInfo) -> tuple[float, ...] | None:
        try:
            return face_normal_yaws(self._shape_form(shape))
        except RuntimeError:
            return None  # --goal_object: not a shape piece, grasp it like a round one

    def _sample_grasp_offset(self) -> tuple[float, float]:
        """XY offset for this attempt: only a piece's very first attempt is perturbed.

        Re-grasping during an insert recovery counts as a retry, so it is never
        perturbed either: stacking two deliberate mistakes would record a recovery that
        itself fails, which is the opposite of the demonstration the clip is for.
        """
        if self._grasp_attempts != 0 or self._insert_attempts > 0:
            return (0.0, 0.0)
        if random.random() >= float(self.config.grasp_perturb_prob):
            return (0.0, 0.0)
        m = float(self.config.grasp_perturb_xy_m)
        return (random.uniform(-m, m), random.uniform(-m, m))


    # ------------------------------------------------------------------
    # Sensing — what the world says, measured each time
    # ------------------------------------------------------------------

    def _piece_inserted(self, env: gym.Env, *, require_settled: bool = True) -> bool:
        """Whether the current piece sits inside the box, per the success term.

        ``require_settled=False`` drops the velocity part of that term, which separates
        "still bouncing in the cavity" from "never got in".
        """
        params = getattr(env.unwrapped.cfg, "piece_in_box_params", None)
        if params is None:
            return False  # --goal_object smoke test: no box for anything to be in.

        from isaaclab.managers import SceneEntityCfg
        from shape_sorting.predicates import objects_centers_inside_aabb

        per_piece = {
            **params,
            "object_cfg_list": [SceneEntityCfg(self._current_shape(env).name)],
        }
        if not require_settled:
            per_piece["velocity_threshold"] = float("inf")
        return bool(objects_centers_inside_aabb(env.unwrapped, **per_piece)[0])

    def _insert_misaligned(self, env: gym.Env, device: torch.device) -> str | None:
        """Why the gripped piece would not drop into its hole, or None if it would."""
        if not self._has_box(env):
            return None  # --goal_object smoke test: no hole to line up with.

        shape = self._current_shape(env)
        obj_xyz, obj_quat_xyzw = self._piece_pose(env, device)
        hole_pos, hole_quat = self._hole_pose_in_robot_base(env, device)
        xy, yaw, tilt = alignment_error(
            obj_xyz,
            obj_quat_xyzw,
            hole_pos,
            hole_quat,
            symmetry_order=yaw_symmetry_order(self._shape_form(shape)),
        )
        # Logged on every insert, not just the failures: this is the only place the
        # natural alignment spread is observable, and the tolerances above need it.
        print(
            f"[CuroboPolicy] INSERT alignment: xy={xy * 1000:.1f} mm "
            f"yaw={math.degrees(yaw):.1f}° tilt={math.degrees(tilt):.1f}°"
        )
        cfg = self.config
        if xy > float(cfg.insert_align_xy_tol_m):
            return f"{xy * 1000:.0f} mm off the hole axis"
        if yaw > float(cfg.insert_align_yaw_tol_rad):
            return f"yawed {math.degrees(yaw):.0f}° against the hole"
        if tilt > float(cfg.insert_align_tilt_tol_rad):
            return f"tilted {math.degrees(tilt):.0f}° (caught on the rim)"
        return None

    def _piece_tilt(self, env: gym.Env, device: torch.device) -> float:
        """Angle between the current piece's own +Z and the base +Z [rad]."""
        return _tilt_of(self._piece_pose(env, device)[1])

    def _piece_pose(self, env: gym.Env, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """The current piece's ``(position (3,), quat_xyzw (4,))`` in the robot base frame.

        The one place piece orientation is read, so that a piece on its top — or a cube on
        its side — reads as the standing piece it is (:func:`standing_quat`).
        """
        return self._pose_of(env, device, self._current_shape(env).name)

    def _tipped(self, env: gym.Env, device: torch.device, name: str) -> bool:
        """Tipped past what an insert tolerates — leaning on something, or on its side."""
        tilt = _tilt_of(self._pose_of(env, device, name)[1])
        return tilt > float(self.config.insert_align_tilt_tol_rad)

    def _pose_of(
        self, env: gym.Env, device: torch.device, name: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from isaaclab.utils.math import convert_quat

        pose = entity_pose_in_robot_base(env, name, device=device)
        quat = convert_quat(pose.quaternion.view(-1)[:4], to="xyzw")
        # ponytail: assumes piece_height == piece_size (the defaults) — a taller or flatter
        # "cube" is not the same solid on its side — and profiles symmetric about their own
        # X, which every regular form here is; upside down, the others would be mirrored.
        cube = name == f"shape_piece_{ShapeForm.CUBE.value}"
        return pose.position.view(3), standing_quat(quat, any_face=cube)

    def _on_the_box(self, env: gym.Env, device: torch.device, name: str) -> bool:
        """Over the cavity's footprint: on the lid, stuck in a hole, or already in the box."""
        xyz = entity_position_in_robot_base(env, name, device=device)
        return self._box_footprint(env, device).outside_by((float(xyz[0]), float(xyz[1]))) < 0.0

    def _box_distance(self, env: gym.Env, device: torch.device, name: str) -> float:
        """XY distance from a piece to the box origin [m]."""
        piece = entity_position_in_robot_base(env, name, device=device)
        box = entity_position_in_robot_base(env, self._box_entity_name(env), device=device)
        return math.hypot(float(piece[0] - box[0]), float(piece[1] - box[1]))

    def _world_still(self, env: gym.Env) -> bool:
        """Nothing that ``_decide`` looks at is moving: the pieces left, and the box."""
        import warp as wp

        todo = self._pending(env)
        held = self._jaw_cmd == float(self.config.jaw_closed) or self._miss_cut_in is not None
        names = [s.name for s in (todo[1:] if held else todo)]  # the held one moves with the arm
        if self._has_box(env):
            names.append(self._box_entity_name(env))
        scene = env.unwrapped.scene
        return all(
            float(torch.linalg.vector_norm(wp.to_torch(scene[n].data.root_lin_vel_w)[0]))
            < _STILL_SPEED_M_S
            for n in names
        )

    def _box_footprint(self, env: gym.Env, device: torch.device) -> BoxFootprint:
        """Where the box, its cavity, its lid and the table under it are, right now."""
        from isaaclab.utils.math import convert_quat

        pose = entity_pose_in_robot_base(env, self._box_entity_name(env), device=device)
        xyz = pose.position.view(3)
        lid_z = float(self._hole_pose_in_robot_base(env, device)[0][2])
        params = env.unwrapped.cfg.piece_in_box_params
        lo, hi = params["aabb_min"], params["aabb_max"]
        return BoxFootprint(
            center=(float(xyz[0]), float(xyz[1])),
            yaw=self._yaw_from_quat_xyzw(convert_quat(pose.quaternion.view(-1)[:4], to="xyzw")),
            lo=(float(lo[0]), float(lo[1])),
            hi=(float(hi[0]), float(hi[1])),
            lid_z=lid_z,
            table_z=2.0 * float(xyz[2]) - lid_z,
        )

    @staticmethod
    def _has_box(env: gym.Env) -> bool:
        """False in the ``--goal_object`` smoke test, which has no box."""
        return getattr(env.unwrapped.cfg, "piece_in_box_params", None) is not None

    @staticmethod
    def _box_entity_name(env: gym.Env) -> str:
        """Scene entity name of the sorting box, from the success-term container."""
        params = getattr(env.unwrapped.cfg, "piece_in_box_params", None)
        if params is None:
            raise RuntimeError("Sorting box unknown: env.cfg.piece_in_box_params is missing.")
        return str(params["container_cfg"].name)

    def _sample_place_perturb(self) -> tuple[float, float, float]:
        """``(dx, dy, dyaw)`` misalignment built into this placement on purpose.

        Only a piece's first placement is ever perturbed, so every recorded recovery
        ends in a correct demonstration. Mirrors :meth:`_sample_grasp_offset`.
        """
        if self._insert_attempts > 0:
            return (0.0, 0.0, 0.0)
        if random.random() >= float(self.config.place_perturb_prob):
            return (0.0, 0.0, 0.0)
        m = float(self.config.place_perturb_xy_m)
        yaw = float(self.config.place_perturb_yaw_rad)
        return (random.uniform(-m, m), random.uniform(-m, m), random.uniform(-yaw, yaw))

    def _sample_miss_notice_delay(self) -> int:
        """Steps to keep carrying an empty gripper before noticing the miss."""
        return random.randint(0, max(0, int(self.config.miss_notice_delay_max_steps)))

    # ------------------------------------------------------------------
    # Motion client + debug wiring
    # ------------------------------------------------------------------

    def _ensure_motion(self, device: torch.device) -> MotionClient:
        if self._motion is not None:
            return self._motion

        cfg = self.config
        if cfg.debug_viser:
            debug = ViserCollisionDebugViz(
                resolve_robot_yml(cfg.robot_yml),
                port=int(cfg.debug_viser_port),
            )
        else:
            debug = NullCollisionDebugViz()

        self._motion = MotionClient(
            MotionClientCfg(
                robot_yml=cfg.robot_yml,
                position_tolerance=cfg.position_tolerance,
                orientation_tolerance=cfg.orientation_tolerance,
                use_cuda_graph=cfg.use_cuda_graph,
                self_collision_check=cfg.self_collision_check,
                optimizer_collision_activation_distance=(
                    cfg.optimizer_collision_activation_distance
                ),
                waypoint_stride=cfg.waypoint_stride,
            ),
            debug=debug,
        )
        self._motion.ensure_ready(device)

        if cfg.debug_viz: # TODO: rename to debug_marker to avoid confiusions with viser visualization
            self._kit_markers = KitFrameMarkers(scale=cfg.marker_frame_scale)

        return self._motion

    def _update_kit_markers(
        self, env: gym.Env, device: torch.device, motion: MotionClient
    ) -> None:
        if self._kit_markers is None or motion.last_goal_xyz is None:
            return
        self._kit_markers.ensure(env, motion.tool_frame)
        goal_pos_w, goal_quat = self._goal_pose_w(env, device, motion)
        self._kit_markers.update(
            env, device, goal_pos_w=goal_pos_w, goal_quat_xyzw=goal_quat
        )

    def _goal_pose_w(
        self, env: gym.Env, device: torch.device, motion: MotionClient
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """World-frame pose of the last planned goal, for the Kit markers."""
        import warp as wp
        from isaaclab.utils.math import combine_frame_transforms

        assert motion.last_goal_xyz is not None and motion.last_goal_quat_xyzw is not None
        num_envs = env.unwrapped.num_envs
        robot = env.unwrapped.scene["robot"]
        robot_pose_w = wp.to_torch(robot.data.root_pose_w).to(device=device, dtype=torch.float32)
        t = motion.last_goal_xyz.unsqueeze(0).expand(num_envs, -1)
        q = motion.last_goal_quat_xyzw.unsqueeze(0).expand(num_envs, -1)
        return combine_frame_transforms(robot_pose_w[:, 0:3], robot_pose_w[:, 3:7], t, q)

    # ------------------------------------------------------------------
    # Goal poses
    # ------------------------------------------------------------------

    def _shape_form(self, shape: ShapeInfo) -> ShapeForm:
        prefix = "shape_piece_"
        if not shape.name.startswith(prefix):
            raise RuntimeError(
                f"CuroboPolicy: cannot map shape '{shape.name}' to a form; "
                f"expected a name starting with '{prefix}'."
            )
        form_value = shape.name[len(prefix) :]
        try:
            return ShapeForm(form_value)
        except ValueError as exc:
            raise RuntimeError(
                f"CuroboPolicy: unknown form '{form_value}' in shape '{shape.name}'."
            ) from exc

    def _hole_frame_name_for_shape(self, shape: ShapeInfo) -> str:
        from shape_sorting.shape_asset import SortingBox

        return SortingBox.hole_frame_name(self._shape_form(shape))

    def _hole_pose_in_robot_base(
        self, env: gym.Env, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Matched lid-hole pose in the robot base frame. ``(pos (3,), quat_xyzw (4,))``."""
        import warp as wp
        from isaaclab.utils.math import subtract_frame_transforms
        from shape_sorting.shape_asset import SortingBox

        shape = self._current_shape(env)
        frame_name = self._hole_frame_name_for_shape(shape)
        scene = env.unwrapped.scene
        try:
            holes = scene[SortingBox.HOLE_FRAMES_SENSOR_NAME]
        except KeyError as exc:
            available = sorted(scene.keys()) if hasattr(scene, "keys") else []
            raise KeyError(
                f"Hole frames sensor not in scene. Available: {available}"
            ) from exc

        names = list(holes.data.target_frame_names)
        try:
            idx = names.index(frame_name)
        except ValueError as exc:
            raise KeyError(
                f"Hole frame '{frame_name}' for shape '{shape.name}' "
                f"not in '{SortingBox.HOLE_FRAMES_SENSOR_NAME}'. Have: {names}"
            ) from exc

        robot = scene["robot"]
        robot_pose_w = wp.to_torch(robot.data.root_pose_w)[0].to(device=device, dtype=torch.float32)
        hole_pos_w = wp.to_torch(holes.data.target_pos_w)[0, idx].to(
            device=device, dtype=torch.float32
        )
        hole_quat_w = wp.to_torch(holes.data.target_quat_w)[0, idx].to(
            device=device, dtype=torch.float32
        )

        hole_b, hole_quat_b = subtract_frame_transforms(
            robot_pose_w[0:3].unsqueeze(0),
            robot_pose_w[3:7].unsqueeze(0),
            hole_pos_w.unsqueeze(0),
            hole_quat_w.unsqueeze(0),
        )
        return hole_b[0], hole_quat_b[0]

    @staticmethod
    def _yaw_from_quat_xyzw(quat_xyzw: torch.Tensor) -> float:
        from isaaclab.utils.math import euler_xyz_from_quat

        _, _, yaw = euler_xyz_from_quat(quat_xyzw.reshape(1, 4))
        return float(yaw[0])

    def _place_roll_candidates(
        self, env: gym.Env, device: torch.device, motion: MotionClient
    ) -> list[float]:
        """SO-101 roll angles that align the grasped piece to the hole (up to symmetry)."""
        form = self._shape_form(self._current_shape(env))
        if yaw_symmetry_order(form) is None:
            # Continuous symmetry: keep nominal EE roll (already reachable).
            return [_GOAL_ROLL_RAD]

        hole_pos, hole_quat = self._hole_pose_in_robot_base(env, device)
        obj_desired = hole_pos.clone()
        obj_desired[2] = obj_desired[2] + float(self.config.place_hover_z_offset_m)
        _, quat_ee0 = so101_ee_pose_xyzw(
            float(obj_desired[0]),
            float(obj_desired[1]),
            float(obj_desired[2]),
            tilt=_GOAL_TILT_RAD,
            roll=_GOAL_ROLL_RAD,
            device=device,
        )

        from isaaclab.utils.math import convert_quat

        _, obj_quat_xyzw = self._piece_pose(env, device)
        q_ee = motion.ee_pose(motion.planner_joint_state(env, device))
        ee_quat_xyzw = convert_quat(q_ee.quaternion.view(-1)[:4], to="xyzw")

        psi_hole = self._yaw_from_quat_xyzw(hole_quat)
        psi_obj = self._yaw_from_quat_xyzw(obj_quat_xyzw)
        psi_ee = self._yaw_from_quat_xyzw(ee_quat_xyzw)
        psi_ee0 = self._yaw_from_quat_xyzw(quat_ee0)
        psi_rel = _wrap_to_pi(psi_obj - psi_ee)
        roll_0 = _wrap_to_pi(psi_hole - psi_ee0 - psi_rel)

        rolls = [_wrap_to_pi(roll_0 + dyaw) for dyaw in place_yaw_offsets(form)]
        rolls.sort(key=lambda r: abs(r))
        return rolls

    def _place_pose_in_robot_base(
        self,
        env: gym.Env,
        device: torch.device,
        motion: MotionClient,
        *,
        roll: float,
        obj_desired: torch.Tensor,
        tilt: float = _GOAL_TILT_RAD,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Place EE so the grasped object origin lands on ``obj_desired`` (robot base).

        Target-agnostic: the caller decides whether that point is above a lid hole or
        above a spot on the table.
        """
        from isaaclab.utils.math import quat_apply

        if motion.grasp_offset_ee is None:
            raise RuntimeError(
                "CuroboPolicy: placing needs a grasp offset — attach() the held piece first."
            )

        # The tool's yaw follows the tool's own XY, which depends on where the offset puts
        # it: iterate. Once was enough for an upright tool (a few mm of slack), not for a
        # tilted one, whose offset swings out sideways.
        offset = motion.grasp_offset_ee.to(device=device, dtype=torch.float32).unsqueeze(0)
        pivot = _PAN_AXIS_XY if tilt else (0.0, 0.0)  # see so101_ee_pose_xyzw
        ee_pos = obj_desired
        for _ in range(4):
            _, quat = so101_ee_pose_xyzw(*ee_pos.tolist(), tilt=tilt, roll=roll, pivot_xy=pivot, device=device)
            ee_pos = obj_desired - quat_apply(quat.unsqueeze(0), offset)[0]
        return so101_ee_pose_xyzw(*ee_pos.tolist(), tilt=tilt, roll=roll, pivot_xy=pivot, device=device)
