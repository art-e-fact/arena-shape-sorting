# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Drives the real CuroboPolicy phase machine against a scripted world. Run: python test_decide.py

Only the sensing and the planner are faked; GO / JAW / UP / _decide run for real. The
world decides what each plan, grasp, alignment and drop *does*, so a scenario can script
the failures seen in real runs and the fuzz test can throw everything at it at once.
Needs torch + arena_so101 (the Arena venv), but no Isaac Sim app.
"""

from __future__ import annotations

import math
import random
import sys
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shape_sorting.curobo_motion import PlanResult
from shape_sorting.curobo_policy import (
    _PIECE_MAX_RADIUS_M,
    _SETTLE_MAX_STEPS,
    BoxFootprint,
    CuroboPolicy,
    CuroboPolicyCfg,
    Goal,
    Phase,
    _rot_from_quat_xyzw,
    _tilt_of,
    grasp_jaw_yaws,
    grasp_tool_xy_roll,
    next_goal,
    park_spots,
    so101_ee_pose_xyzw,
    standing_quat,
    upright_tilt_rolls,
)
from shape_sorting.shape_forms import ShapeForm, face_normal_yaws
from shape_sorting.shape_sorting_env import ShapeInfo

CUBE, CYL, HEX = "shape_piece_cube", "shape_piece_cylinder", "shape_piece_hexagon"
_POSE = (torch.zeros(3), torch.tensor([0.0, 0.0, 0.0, 1.0]))


class World:
    """What really happens. Each rule gets (piece, n) — n counts that rule's calls per piece."""

    def __init__(self, pieces, **rules):
        self.in_box: set[str] = set()
        self.on_box: set[str] = set()  # where a piece starts: on the lid or in a hole
        self.tilt = {p: 0.0 for p in pieces}
        self.holding = False
        self.calls: Counter = Counter()
        self.rules = {
            "plan_ok": lambda goal, piece, w: True,
            "grasp_holds": lambda piece, n: True,
            "regrip_holds": lambda piece, n: True,
            "aligned": lambda piece, n: True,
            "drops_in": lambda piece, n: True,
            "wedge_tilt": lambda piece, n: 0.0,
            "park_tilt": lambda piece, n: 0.0,
            "keeps_hold": lambda piece, n: True,
            "still": lambda piece, n: True,  # asked once per step while UP waits to decide
            **rules,
        }

    def ask(self, rule, piece, *extra):
        self.calls[(rule, piece)] += 1
        return self.rules[rule](piece, self.calls[(rule, piece)], *extra)


class FakeMotion:
    def __init__(self, world: World, policy: CuroboPolicy):
        self.w, self.p = world, policy
        self.last_goal_xyz = None
        self.plans: list[tuple[str, bool]] = []
        self.attached_in: list[Phase] = []

    def plan_to_pose(self, env, device, pos, quat, *, label="plan", start=None):
        goal, piece = label.split()[:2]
        ok = bool(self.w.rules["plan_ok"](goal, piece, self.w))
        self.plans.append((label, ok))
        n = 3 if start is None else 2  # hover move, then descent
        waypoints = torch.full((n, 5), float(len(self.plans)))
        return PlanResult(waypoints=waypoints, success=ok, hold_action=torch.zeros(6))

    @contextmanager
    def obstacles_hidden(self, name):
        yield []

    def attach(self, env, device, name):
        assert name == self.p._todo[0].name, "attached something other than the current piece"
        self.attached_in.append(self.p._phase)

    def detach(self):
        p = self.p
        if p._phase is not Phase.JAW or p._jaw_cmd != p.config.jaw_open:
            return  # a bookkeeping detach, not a release
        piece = p._todo[0].name
        assert self.w.holding or p._goal is Goal.PARK, "released over the box while holding nothing"
        if p._goal is Goal.INSERT and self.w.ask("drops_in", piece):
            self.w.in_box.add(piece)
            self.w.tilt[piece] = 0.0
        elif p._goal is Goal.INSERT:
            self.w.tilt[piece] = self.w.ask("wedge_tilt", piece)
        else:
            self.w.tilt[piece] = self.w.ask("park_tilt", piece)
        self.w.holding = False

    def measured_jaw(self, env, device):
        p = self.p
        piece = p._todo[0].name
        if p._phase is Phase.JAW:  # the one read right after a close
            rule = "grasp_holds" if p._goal is Goal.GRASP else "regrip_holds"
            self.w.holding = bool(self.w.ask(rule, piece))
        elif self.w.holding and not self.w.ask("keeps_hold", piece):
            self.w.holding = False  # slipped out on the way up
        return torch.tensor(0.19 if self.w.holding else -0.172)

    def joint_action(self, env, device):
        return torch.zeros(6)

    def q_to_action(self, q, device, *, jaw):
        return torch.cat([q.to(torch.float32), torch.tensor([float(jaw)])])

    def notify_joint(self, q):
        pass

    def clear_last_goal(self):
        pass


def make(pieces=(CUBE, CYL, HEX), cfg=None, box=True, **rules):
    policy = CuroboPolicy(cfg or CuroboPolicyCfg())
    world = World(pieces, **rules)
    motion = FakeMotion(world, policy)
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            device="cpu",
            num_envs=1,
            cfg=SimpleNamespace(
                shapes=[ShapeInfo(prim_path=f"{{ENV_REGEX_NS}}/{p}") for p in pieces],
                **({"piece_in_box_params": {}} if box else {}),
            ),
        ),
        action_space=SimpleNamespace(shape=(1, 6)),
    )
    def two(env, device, motion):
        return [("a", _POSE, _POSE), ("b", _POSE, _POSE)]

    def grasp_candidates(env, device, motion):
        policy._grasp_tilt = world.tilt[current()]
        return two(env, device, motion)

    def current():
        return policy._todo[0].name

    policy._ensure_motion = lambda device: motion
    policy._grasp_candidates = grasp_candidates
    policy._insert_candidates = policy._park_candidates = two
    policy._box_entity_name = lambda env: "sorting_box"
    policy._piece_inserted = lambda env, require_settled=True: current() in world.in_box
    policy._piece_tilt = lambda env, device: world.tilt[current()]
    policy._world_still = lambda env: bool(world.ask("still", policy._pending(env)[0].name))
    policy._on_the_box = lambda env, device, name: name in world.on_box | world.in_box
    policy._tipped = lambda env, device, name: world.tilt[name] > policy.config.insert_align_tilt_tol_rad
    policy._insert_misaligned = lambda env, device: (
        None if world.ask("aligned", current()) else "misaligned"
    )
    return policy, world, env, motion


def run(policy, env, max_steps=20_000):
    events = []
    for t in range(max_steps):
        policy.get_action(env, None)
        events += [(t, e) for e in policy.pop_demo_events()]
        if policy.is_demonstration_ended():
            return t, events
    raise AssertionError(f"still running after {max_steps} steps in {policy._phase}")


def kinds(events):
    return [e for _, e in events]


# --- the decision table ------------------------------------------------------------

def test_next_goal_table():
    for holding in (True, False):
        for upright in (True, False):
            for in_box in (True, False):
                for ins in (True, False):
                    for grs in (True, False):
                        got = next_goal(holding=holding, upright=upright, in_box=in_box,
                                        insert_tries_left=ins, grasp_tries_left=grs)
                        if holding:
                            want = "insert" if upright and ins else "park"
                        elif in_box:
                            want = "finish"
                        else:
                            want = "grasp" if ins and grs else "defer"
                        assert got == want, (holding, upright, in_box, ins, grs, got)


# --- grasp geometry ------------------------------------------------------------------

def _yaw_quat(deg):
    h = math.radians(deg) / 2
    return torch.tensor([0.0, 0.0, math.sin(h), math.cos(h)])


def _tilt_quat(deg):  # about base X
    h = math.radians(deg) / 2
    return torch.tensor([math.sin(h), 0.0, 0.0, math.cos(h)])


def test_radial_grasp_is_reproduced_exactly():
    px, py = 0.099, -0.211
    tx, ty, roll = grasp_tool_xy_roll(px, py, math.atan2(py, px), 0.03)
    rho = math.hypot(px, py)
    assert abs(tx - px * (1 - 0.03 / rho)) < 1e-12 and abs(ty - py * (1 - 0.03 / rho)) < 1e-12
    assert abs(roll) < 1e-12, "the tuned grasp must stay roll 0"


def test_cube_at_45_degrees_is_squeezed_across_flats():
    yaws = grasp_jaw_yaws(_yaw_quat(45), face_normal_yaws(ShapeForm.CUBE), radial_yaw=0.0,
                          max_tilt=math.radians(20))
    offsets = sorted(round(math.degrees(y)) % 90 for y in yaws)
    assert offsets == [45, 45, 45, 45], "every candidate must be face-aligned, never diagonal"
    assert abs(abs(math.degrees(yaws[0])) - 45) < 1e-6, "best first: 45° off radial"


def test_hexagon_is_gripped_on_flats_not_vertices():
    yaws = grasp_jaw_yaws(_yaw_quat(0), face_normal_yaws(ShapeForm.HEXAGON), radial_yaw=0.0,
                          max_tilt=math.radians(20))
    assert sorted(round(math.degrees(y)) % 60 for y in yaws) == [30] * 6


def test_tipped_cube_only_offers_faces_that_stayed_vertical():
    # Tipped 40° about base X: the ±X faces stay vertical, ±Y now slope 40°.
    yaws = grasp_jaw_yaws(_tilt_quat(40), face_normal_yaws(ShapeForm.CUBE), radial_yaw=0.0,
                          max_tilt=math.radians(20))
    assert sorted(round(math.degrees(y)) % 360 for y in yaws) == [0, 180]


def test_a_cube_on_its_side_reads_as_a_standing_cube():
    # 30 mm tall and 30 mm wide: on any face it is the same solid, and fits its hole.
    for deg in (90.0, -90.0, 180.0):
        assert _tilt_of(standing_quat(_tilt_quat(deg), any_face=True)) < 1e-4, deg
    assert abs(math.degrees(_tilt_of(standing_quat(_tilt_quat(30.0), any_face=True))) - 30.0) < 1e-3
    # Only a relabelling of its own axes — the solid itself must not turn.
    q = torch.tensor([0.3, -0.5, 0.2, 0.8])
    q = q / q.norm()
    for any_face in (True, False):
        relabel = _rot_from_quat_xyzw(q).T @ _rot_from_quat_xyzw(standing_quat(q, any_face=any_face))
        assert torch.allclose(relabel.abs().sum(0), torch.ones(3), atol=1e-4), relabel
        assert abs(float(torch.linalg.det(relabel)) - 1.0) < 1e-4


def test_a_prism_on_its_top_reads_as_standing_but_on_its_side_does_not():
    # Live: a hexagon stood back up landed on its other end, read "tilted 179°", and was
    # parked again and again. Upside down it is the same prism, and fits its hole.
    assert _tilt_of(standing_quat(_tilt_quat(179.0))) < math.radians(1.01)
    assert abs(math.degrees(_tilt_of(standing_quat(_tilt_quat(90.0)))) - 90.0) < 1e-3
    assert abs(math.degrees(_tilt_of(standing_quat(_tilt_quat(150.0)))) - 30.0) < 1e-3
    # Yaw is read off the body X axis, which the flip keeps.
    rot = _rot_from_quat_xyzw(standing_quat(torch.tensor([1.0, 0.0, 0.0, 0.0])))  # 180° about X
    assert torch.allclose(rot[:, 0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6), rot


def test_round_piece_gets_four_yaws_radial_first():
    yaws = grasp_jaw_yaws(_yaw_quat(0), None, radial_yaw=0.3, max_tilt=math.radians(20))
    assert len(yaws) == 4 and abs(yaws[0] - 0.3) < 1e-9


def test_tipped_round_piece_is_gripped_across_its_tilt_axis():
    # Tipped 30° about base X: along X its section is still a circle; along Y it is an
    # ellipse the closing jaws would squeeze up and out.
    yaws = grasp_jaw_yaws(_tilt_quat(30), None, radial_yaw=0.3, max_tilt=math.radians(20))
    assert sorted(round(math.degrees(y)) % 360 for y in yaws) == [0, 180], yaws


# --- scenarios from real runs --------------------------------------------------------

def test_happy_path_inserts_everything_and_goes_home():
    policy, world, env, _ = make()
    _, events = run(policy, env)
    assert world.in_box == {CUBE, CYL, HEX}
    assert kinds(events) == ["checkpoint"] * 3


def test_boxed_in_piece_waits_for_its_neighbour():
    # The logged failure: the cube sits between the cylinder and the hexagon on the
    # radial line and no grasp plans until the cylinder is gone.
    def plan_ok(goal, piece, w):
        return not (goal == "GRASP" and piece == CUBE and CYL not in w.in_box)

    policy, world, env, _ = make(plan_ok=plan_ok)
    _, events = run(policy, env)
    assert world.in_box == {CUBE, CYL, HEX}
    assert kinds(events) == ["checkpoint"] * 3, "a deferral is not a mistake: nothing to cut"


def test_permanently_blocked_piece_ends_the_episode_instead_of_freezing():
    policy, world, env, _ = make(plan_ok=lambda goal, piece, w: piece != CUBE)
    run(policy, env)
    assert world.in_box == {CYL, HEX}
    assert policy._phase is Phase.DONE


def test_nothing_plannable_is_done_on_the_first_step():
    policy, _, env, _ = make(plan_ok=lambda *a: False)
    t, events = run(policy, env)
    assert t == 1 and events == []  # step 0 always holds: see the next test


def test_upright_misalignment_is_re_aimed_without_parking():
    policy, world, env, motion = make(aligned=lambda piece, n: not (piece == CUBE and n == 1))
    _, events = run(policy, env)
    assert world.in_box == {CUBE, CYL, HEX}
    assert kinds(events) == ["cut", "checkpoint", "checkpoint", "checkpoint"]
    assert not any(label.startswith("PARK") for label, _ in motion.plans)


def test_tilted_in_the_jaws_is_parked_then_regrasped():
    def aligned(piece, n):
        if piece == CUBE and n == 1:
            world.tilt[CUBE] = math.radians(35)  # caught the rim and pivoted in the jaws
            return False
        return True

    policy, world, env, motion = make(aligned=aligned)
    _, events = run(policy, env)
    assert world.in_box == {CUBE, CYL, HEX}
    assert any(label.startswith("PARK") for label, _ in motion.plans)
    assert kinds(events)[0] == "cut"


def test_cube_wedged_at_40_degrees_after_release_is_recovered():
    # The reported stuck case: released 45° off, it wedged in the hole at 40° and the
    # regrip closed on nothing. It must be grasped where it lies, parked, and retried.
    policy, world, env, motion = make(
        drops_in=lambda piece, n: not (piece == CUBE and n == 1),
        wedge_tilt=lambda piece, n: math.radians(40),
        regrip_holds=lambda piece, n: False,
    )
    run(policy, env)
    assert world.in_box == {CUBE, CYL, HEX}
    assert any(label.startswith(f"PARK {CUBE}") for label, _ in motion.plans)


def test_piece_that_never_comes_up_level_is_bounded():
    policy, world, env, _ = make(
        drops_in=lambda piece, n: piece != CUBE,
        wedge_tilt=lambda piece, n: math.radians(40),
        park_tilt=lambda piece, n: math.radians(90),  # lands on its side every time
    )
    run(policy, env)
    assert world.in_box == {CYL, HEX} and policy._phase is Phase.DONE


def test_pretended_miss_never_lowers_an_empty_gripper():
    cfg = CuroboPolicyCfg(miss_notice_delay_max_steps=40)
    random.seed(3)
    policy, world, env, _ = make(cfg=cfg, grasp_holds=lambda piece, n: not (piece == CUBE and n == 1))
    _, events = run(policy, env)  # FakeMotion.detach asserts no release while empty
    assert world.in_box == {CUBE, CYL, HEX}
    assert kinds(events).count("cut") == 1


# --- a mistake must never be confirmed by the next checkpoint --------------------------

def test_grasp_that_slips_on_the_way_up_is_cut():
    policy, world, env, _ = make(keeps_hold=lambda piece, n: not (piece == CUBE and n == 1))
    _, events = run(policy, env)
    assert world.in_box == {CUBE, CYL, HEX}
    assert kinds(events) == ["cut", "checkpoint", "checkpoint", "checkpoint"]


def test_grasp_that_tips_an_upright_piece_is_cut():
    def grasp_holds(piece, n):
        if piece == CUBE and n == 1:
            world.tilt[CUBE] = math.radians(35)  # pinched by a corner
        return True

    policy, world, env, _ = make(grasp_holds=grasp_holds, park_tilt=lambda piece, n: 0.0)
    _, events = run(policy, env)
    assert world.in_box == {CUBE, CYL, HEX}
    assert kinds(events)[0] == "cut"


def test_grasping_an_already_wedged_piece_is_the_recovery_not_a_mistake():
    # Wedged at 40° after a failed drop; the regrip misses. Grasping it where it lies
    # and parking it is exactly the demonstration wanted, so exactly two cuts: the failed
    # drop and the empty regrip — none for holding the (already tipped) piece.
    policy, world, env, _ = make(
        drops_in=lambda piece, n: not (piece == CUBE and n == 1),
        wedge_tilt=lambda piece, n: math.radians(40),
        regrip_holds=lambda piece, n: False,
    )
    _, events = run(policy, env)
    assert world.in_box == {CUBE, CYL, HEX}
    assert kinds(events) == ["cut", "cut", "checkpoint", "checkpoint", "checkpoint"]


def test_giving_up_on_a_tried_piece_is_cut_but_an_unplannable_one_is_not():
    policy, world, env, _ = make(aligned=lambda piece, n: piece != CUBE)
    _, events = run(policy, env)
    assert world.in_box == {CYL, HEX}
    # 3 failed inserts (budget 2), then the deferral cut; later rounds repeat that.
    assert kinds(events)[:4] == ["cut", "cut", "cut", "cut"]

    policy, world, env, _ = make(plan_ok=lambda goal, piece, w: piece != CUBE)
    _, events = run(policy, env)
    assert "cut" not in kinds(events), "nothing was tried, so nothing to drop"


def test_piece_that_keeps_landing_on_its_side_is_retried_within_the_grasp_budget():
    policy, _, env, motion = make(
        drops_in=lambda piece, n: piece != CUBE,
        wedge_tilt=lambda piece, n: math.radians(40),
        park_tilt=lambda piece, n: math.radians(90),
    )
    run(policy, env)
    parks = [label for label, ok in motion.plans if label.startswith(f"PARK {CUBE}") and ok]
    # Lying down is no reason to give up any more — but each try is a grasp, and the
    # grasp budget per round still bounds them. Two rounds: before and after the others.
    budget = policy.config.max_grasp_retries
    assert budget < len(parks) // 2 <= 2 * budget, len(parks) // 2


def test_slip_after_an_insert_recovery_is_cut_too():
    # Misaligned at the bottom (cut), lifted — and it slips out on the way up. Losing it
    # is a second mistake after that cut, not part of the recovery.
    lifted = []

    def aligned(piece, n):
        ok = not (piece == CUBE and n == 1)
        if not ok:
            lifted.append(piece)
        return ok

    def keeps_hold(piece, n):
        return not (lifted and piece == CUBE and n == 2)  # n=1 is the lift after the grasp

    policy, world, env, _ = make(aligned=aligned, keeps_hold=keeps_hold)
    _, events = run(policy, env)
    assert world.in_box == {CUBE, CYL, HEX}
    assert kinds(events)[:3] == ["cut", "cut", "checkpoint"], kinds(events)


def test_pretended_carry_is_attached_at_the_bottom_like_a_real_grasp():
    cfg = CuroboPolicyCfg(miss_notice_delay_max_steps=40)
    random.seed(3)
    policy, _, env, motion = make(cfg=cfg, grasp_holds=lambda piece, n: not (piece == CUBE and n == 1))
    run(policy, env)
    # Real grasps attach at the hover in _decide (UP); the pretend attaches in JAW, at
    # the contact pose, and is not re-attached higher up.
    assert Phase.JAW in motion.attached_in
    assert motion.attached_in.count(Phase.JAW) == 1


def test_goal_object_mode_grasps_then_goes_home():
    policy, _, env, motion = make(pieces=(CUBE,), box=False)
    del policy._piece_inserted  # the real one: no box means nothing is "in the box"
    _, events = run(policy, env)
    assert any(label.startswith("GRASP") and ok for label, ok in motion.plans), "it must move"
    assert policy._phase is Phase.DONE and kinds(events) == ["checkpoint"]


# --- pieces that start in the wrong place (--drop_on_hole_prob) ------------------------

def test_piece_that_starts_wedged_in_its_hole_is_taken_out_and_reinserted_first():
    # What a trained policy leaves behind after letting go off-centre, and what the env's
    # drop event recreates: nothing went wrong in *this* episode, so nothing is cut. The
    # stuck piece is dealt with before the others, though it is last in the list.
    policy, world, env, motion = make()
    world.tilt[HEX] = math.radians(35)
    world.on_box.add(HEX)
    _, events = run(policy, env)
    assert world.in_box == {CUBE, CYL, HEX}
    assert motion.plans[0][0].startswith(f"GRASP {HEX}"), motion.plans[0]
    assert any(label.startswith(f"PARK {HEX}") for label, _ in motion.plans), "levelled first"
    assert kinds(events) == ["checkpoint"] * 3


def test_piece_that_starts_in_the_box_is_simply_finished():
    policy, world, env, motion = make()
    world.in_box.add(CUBE)
    _, events = run(policy, env)
    assert not any(CUBE in label for label, _ in motion.plans)
    assert kinds(events) == ["checkpoint"] * 3


def test_a_piece_lying_on_its_side_is_picked_up_and_stood_up_first():
    # Not a mistake of this episode, so nothing to cut: grasp it lying, set it down
    # standing, grasp it again and insert it — before the others, whose inserts would
    # otherwise brush it (live, a lying cylinder rolled out of reach that way).
    policy, world, env, motion = make()
    world.tilt[HEX] = math.radians(90)
    _, events = run(policy, env)
    assert world.in_box == {CUBE, CYL, HEX}
    assert motion.plans[0][0].startswith(f"GRASP {HEX}"), motion.plans[0]
    assert any(label.startswith(f"PARK {HEX}") for label, _ in motion.plans)
    assert kinds(events) == ["checkpoint"] * 3


def test_a_piece_stuck_steeply_in_its_hole_is_still_tried():
    policy, world, env, motion = make()
    world.tilt[HEX] = math.radians(70)
    world.on_box.add(HEX)
    run(policy, env)
    assert world.in_box == {CUBE, CYL, HEX}
    assert motion.plans[0][0].startswith(f"GRASP {HEX}"), motion.plans[0]


def test_nothing_is_decided_until_the_dropped_pieces_have_landed():
    # Step 0 holds without asking: just-teleported pieces read zero velocity mid-air —
    # the live bug where every grasp was planned around pieces still falling.
    policy, world, env, motion = make(still=lambda piece, n: n > 4)
    for t in range(6):
        policy.get_action(env, None)
        assert bool(motion.plans) == (t == 5), t
    assert world.calls[("still", CUBE)] == 5
    # ...and a piece that never stops rolling only delays the start, bounded.
    policy, _, env, motion = make(still=lambda piece, n: False)
    for _ in range(_SETTLE_MAX_STEPS):
        policy.get_action(env, None)
    assert motion.plans == []
    policy.get_action(env, None)
    assert motion.plans, "decides anyway once the wait runs out"


def test_the_real_still_check_skips_the_held_piece_but_not_the_box():
    import warp as wp

    wp.init()
    speed = {CUBE: 0.5, CYL: 0.0, HEX: 0.0, "sorting_box": 0.0}

    def env():
        scene = {n: SimpleNamespace(data=SimpleNamespace(
            root_lin_vel_w=wp.from_torch(torch.tensor([[v, 0.0, 0.0]])))) for n, v in speed.items()}
        cfg = SimpleNamespace(shapes=[ShapeInfo(prim_path=f"/{p}") for p in (CUBE, CYL, HEX)],
                              piece_in_box_params={"container_cfg": SimpleNamespace(name="sorting_box")})
        return SimpleNamespace(unwrapped=SimpleNamespace(scene=scene, cfg=cfg, device="cpu"))

    policy = CuroboPolicy(CuroboPolicyCfg())
    policy._on_the_box = policy._tipped = lambda *a: False  # queue order is not what this checks
    assert not policy._world_still(env()), "the current piece is moving"
    policy._jaw_cmd = policy.config.jaw_closed
    assert policy._world_still(env()), "...because it is in the jaws, moving with the arm"
    speed["sorting_box"] = 0.5
    assert not policy._world_still(env()), "the box counts too"


# --- where a piece is set down --------------------------------------------------------

_BOX = BoxFootprint(center=(0.17, -0.10), yaw=0.3, lo=(-0.04, -0.04), hi=(0.04, 0.04),
                    lid_z=0.07, table_z=0.03)


def test_a_piece_lifted_off_the_table_goes_back_where_it_was():
    here = (0.22, 0.10)
    assert park_spots(_BOX, here_xy=here, others_xy=[(0.25, 0.15)])[0] == here


def test_a_piece_lifted_out_of_its_hole_goes_next_to_the_box_clear_of_everything():
    here = (0.19, -0.09)  # over the lid
    neighbour = _BOX.ring(0.04)[3]
    spots = park_spots(_BOX, here_xy=here, others_xy=[neighbour])
    assert spots and here not in spots
    for p in spots:
        assert math.hypot(*p) >= 0.14, "not in the cramped strip by the robot's base"
        assert _BOX.outside_by(p) >= 0.03, "past the wall, room for the piece"
        assert math.dist(p, neighbour) >= 2 * _PIECE_MAX_RADIUS_M, "not on top of the neighbour"
    dists = [math.dist(p, here) for p in spots]
    assert dists == sorted(dists), "nearest first: shortest carry"
    # A stood-up piece rocks toward the robot after release: those spots keep more room.
    roomy = park_spots(_BOX, here_xy=here, others_xy=[neighbour], clearance=0.025)
    assert roomy and all(_BOX.outside_by(p) >= 0.055 - 1e-6 for p in roomy)


def test_a_held_piece_is_set_down_standing_with_a_reachable_wrist():
    # The grip fixes the piece's axis in the tool frame; any (tilt, roll) returned must
    # stand it up — to within the lean — through so101_ee_pose_xyzw, i.e. a pose the
    # 5-DoF arm can actually take, wherever on the table that is.
    gen = torch.Generator().manual_seed(0)
    rng = random.Random(0)
    for _ in range(200):
        a = torch.randn(3, generator=gen)
        a = a / a.norm()
        for lean in (0.0, math.radians(25)):
            ways = upright_tilt_rolls(a, lean=lean)
            assert ways and all(abs(t) <= math.pi / 2 + 1e-6 for t, _ in ways)
            assert [abs(t) for t, _ in ways] == sorted(abs(t) for t, _ in ways), "least tilt first"
            for t, r in ways:
                _, q = so101_ee_pose_xyzw(rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), 0.1, tilt=t, roll=r)
                up = abs(float((_rot_from_quat_xyzw(q) @ a)[2]))  # either end down
                assert math.acos(min(1.0, up)) <= lean + 1e-3, (a, lean, t, r)

    # Pulled out of its hole tipped 72° about the jaw axis: pitch the wrist 72°, fingertips
    # away from the base, jaws turned tangential so the pitch turns the piece about them.
    t, r = upright_tilt_rolls(torch.tensor([0.0, math.sin(math.radians(72)), math.cos(math.radians(72))]))[0]
    assert abs(math.degrees(t) + 72) < 1e-3 and abs(abs(math.degrees(r)) - 90) < 1e-3, (t, r)
    # Already upright in the jaws: the gripper stays vertical.
    assert abs(upright_tilt_rolls(torch.tensor([0.0, 0.0, 1.0]))[0][0]) < 1e-9


def test_a_lying_hexagon_is_gripped_across_its_sides_once_per_side():
    # Squeezing its end faces would leave a finger under it once it stands. Across, the
    # two sloped faces per side are one jaw direction, and one plan is enough for it.
    yaws = grasp_jaw_yaws(_tilt_quat(90), face_normal_yaws(ShapeForm.HEXAGON), radial_yaw=0.0,
                          max_tilt=math.radians(20))
    assert sorted(round(math.degrees(y)) % 360 for y in yaws) == [0, 180], yaws


# --- fuzz: throw every failure at it at once ------------------------------------------

def test_fuzz_always_terminates_and_checkpoints_only_real_inserts():
    for seed in range(300):
        rng = random.Random(seed)
        random.seed(seed)
        rules = {
            "plan_ok": lambda goal, piece, w, rng=rng: rng.random() < 0.8,
            "grasp_holds": lambda piece, n, rng=rng: rng.random() < 0.7,
            "regrip_holds": lambda piece, n, rng=rng: rng.random() < 0.5,
            "aligned": lambda piece, n, rng=rng: rng.random() < 0.6,
            "drops_in": lambda piece, n, rng=rng: rng.random() < 0.6,
            "wedge_tilt": lambda piece, n, rng=rng: rng.choice([0.0, math.radians(40)]),
            "park_tilt": lambda piece, n, rng=rng: rng.choice([0.0, 0.0, math.radians(90)]),
            "keeps_hold": lambda piece, n, rng=rng: rng.random() < 0.9,
            "still": lambda piece, n, rng=rng: rng.random() < 0.7,
        }
        cfg = CuroboPolicyCfg(miss_notice_delay_max_steps=rng.choice([0, 30]))
        policy, world, env, _ = make(cfg=cfg, **rules)
        for piece in world.tilt:  # some start dropped on the lid: fell in, wedged, or level
            world.tilt[piece] = rng.choice([0.0, 0.0, math.radians(35), math.radians(70)])
            if world.tilt[piece] or rng.random() < 0.1:
                (world.on_box if world.tilt[piece] else world.in_box).add(piece)
        _, events = run(policy, env, max_steps=60_000)
        assert policy._phase is Phase.DONE, seed
        # A checkpoint is only ever a piece that really went in, one each.
        assert kinds(events).count("checkpoint") == len(world.in_box), seed


if __name__ == "__main__":
    for name, case in sorted(globals().items()):
        if name.startswith("test_"):
            case()
            print(f"ok {name}")
    print("all decide checks passed")
