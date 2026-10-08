"""Edge cases for the scripted CuroboPolicy, recorded for review on the Artefacts dashboard.

Each scenario boots Isaac Sim for minutes and records a viewport video, so it runs only
from the ``curobo_scenarios`` job in artefacts.yaml, never as a routine check (the fast
suite is ``arena_envs/tests``). By hand:

    python -m shape_sorting.scenarios 10_in_hole --out /tmp/scenario

The scenario runs ``shape_sorting.run_policy`` in a child process, moves what it wrote
(the video, per-episode results, Arena's report) out of its timestamped subfolder into
``--out``, and adds ``tests_junit.xml`` and ``metrics.json`` there, where Artefacts reads
them. Artefacts ignores exit codes, so without that JUnit every run would show as passed.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

from junit_xml import TestCase, TestSuite

from shape_sorting.shape_sorting_env import ShapeSortingEnvironmentCfg


@dataclass(frozen=True)
class Scenario:
    what: str
    """What the clip should show, in one line."""
    env: dict = field(default_factory=dict)
    """``ShapeSortingEnvironmentCfg`` overrides. Trained policies can reuse these."""
    curobo: dict = field(default_factory=dict)
    """``CuroboPolicyCfg`` overrides, which only CuroboPolicy understands."""
    expect: str = "success"
    """``success``: every episode sorts every piece, whatever recoveries it took on the way.
    ``completes``: every episode runs to its end and leaves a video, for scenarios built so
    that pieces are left unsorted."""
    seed: int = 0
    episodes: int = 1


SCENARIOS: dict[str, Scenario] = {
    "01_nominal": Scenario("Happy path, place-retreat after every insert. The reference clip."),
    # 10 and 11 use the 1 kg box the README measured these recoveries with: pulling a stuck
    # piece out drags the default 0.35 kg box far enough that later inserts miss.
    "10_in_hole": Scenario(
        "One piece starts fallen into, wedged in, or lying on its lid hole, and is taken out and sorted.",
        env={"drop_on_hole_prob": 1.0, "box_mass": 1.0, "episode_length_s": 60},
    ),
    "11_tipped": Scenario(
        "One piece starts on its side; the wrist pitches to stand it back up.",
        env={"tip_over_prob": 1.0, "box_mass": 1.0, "episode_length_s": 60},
    ),
    "12_tipped_triangle": Scenario(
        "A triangle starts on its side: no two faces are parallel, and lying at some angles"
        " none stays vertical, so the jaws take the least tipped.",
        # The cube is never tipped (on its side it is the same cube), so the triangle always is.
        env={"tip_over_prob": 1.0, "forms": ["cube", "triangle"], "episode_length_s": 60},
    ),
    "13_missed_grasp": Scenario(
        "Grasps aimed off-centre: an empty jaw is noticed, lifted clear and regrasped.",
        env={"episode_length_s": 60},
        curobo={"grasp_perturb_prob": 1.0},
    ),
    "14_misaligned_insert": Scenario(
        "Inserts aimed off in position and yaw are caught above the hole, then re-aimed or parked.",
        env={"episode_length_s": 60},
        curobo={"place_perturb_prob": 1.0},
    ),
    "15_release_regrip": Scenario(
        "Loose alignment checks let a misaligned piece go; it fails in the box and is regripped.",
        env={"episode_length_s": 60},
        # Wider than the ±12 mm / ±45° place perturbation, so nothing is caught before release.
        curobo={"place_perturb_prob": 1.0, "insert_align_xy_tol_m": 0.015, "insert_align_yaw_tol_rad": 0.8},
    ),
    "16_budgets_exhausted": Scenario(
        "Inserts are aimed off and a miss may not be retried: it is parked and deferred, until"
        " nothing workable is left and the policy stops at DONE instead of freezing.",
        # Short: reaching DONE does not end the episode, the arm just waits out the clock.
        env={"episode_length_s": 25},
        curobo={"place_perturb_prob": 1.0, "max_insert_retries": 0},
        expect="completes",
    ),
    "17_let_go_misaligned": Scenario(
        "Inserts aimed off are let go anyway, as a trained policy would; the arm climbs out with"
        " the jaw open and grasps the piece again where it settled.",
        env={"episode_length_s": 60},
        curobo={"place_perturb_prob": 1.0, "place_release_misaligned_prob": 1.0},
    ),
}

# Upper bound on a run: Kit boot + cold shader cache + cuRobo warm-up, then sim time
# with viewport rendering. Only a hang should ever reach it.
_BOOT_S = 600.0
_WALL_S_PER_SIM_S = 4.0


def flags(overrides: dict) -> list[str]:
    """``{"forms": ["star"], "debug_viz": True, "clearance": 0.001}`` -> ``--forms star --debug_viz --clearance 0.001``."""
    out = []
    for key, value in overrides.items():
        if value is True:
            out.append(f"--{key}")
        elif isinstance(value, list):
            out += [f"--{key}", *map(str, value)]
        else:
            out += [f"--{key}", str(value)]
    return out


def command(scenario: Scenario, out: Path) -> list[str]:
    return [
        sys.executable, "-m", "shape_sorting.run_policy",
        "--policy_type", "shape_sorting.curobo_policy.CuroboPolicy",
        "--num_episodes", str(scenario.episodes),
        "--seed", str(scenario.seed),
        # --seed alone leaves the piece layout to an unseeded RNG.
        "--placement_seed", str(scenario.seed),
        "--output_base_dir", str(out),
        "--record_viewport_video",
        # Headless Isaac Lab runs without Kit's renderer unless cameras are enabled, and
        # the viewport capture needs it (omni.replicator).
        "--enable_cameras",
        *flags(scenario.curobo),
        "--external_environment_class_path", "shape_sorting.shape_sorting_env:ShapeSortingEnvironment",
        "shape_sorting_test", "--embodiment", "so101_abs_joint",
        *flags(scenario.env),
    ]  # fmt: skip


def run(scenario: Scenario, out: Path, timeout_s: float) -> int | None:
    """Run the scenario; return its exit code, or ``None`` if it was killed on timeout."""
    # Exit right after a clean rollout instead of in Kit's app.close(), which can hang.
    env = {**os.environ, "ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE": "1"}
    # Own process group, so the kill below also reaches Kit's helper processes.
    proc = subprocess.Popen(command(scenario, out), env=env, start_new_session=True)
    try:
        return proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return None
    finally:
        # SIGKILL: Kit installs its own SIGTERM handler.
        with suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def judge(scenario: Scenario, runs: list[Path], returncode: int | None) -> tuple[str | None, dict]:
    """Return ``(failure message or None, numeric metrics)`` from the run folders it left."""
    episodes = [
        json.loads(line)
        for run_dir in runs
        for path in run_dir.glob("episode_results_rank*.jsonl")
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    successes = sum(bool(e["success"]) for e in episodes)
    metrics = {
        "episodes": len(episodes),
        "successes": successes,
        "success_rate": successes / len(episodes) if episodes else 0.0,
        "mean_episode_steps": sum(e["episode_length"] for e in episodes) / len(episodes) if episodes else 0.0,
        "returncode": -1 if returncode is None else returncode,
    }
    if returncode is None:
        failure = "timed out (hung)"
    elif returncode != 0:
        failure = f"run_policy exited with {returncode}"
    elif len(episodes) < scenario.episodes:
        failure = f"recorded {len(episodes)} of {scenario.episodes} episodes"
    elif not any(run_dir.rglob("*.mp4") for run_dir in runs):
        failure = "no video was written"
    elif scenario.expect == "success" and successes < len(episodes):
        failure = f"sorted every piece in {successes} of {len(episodes)} episodes"
    else:
        failure = None
    return failure, metrics


def flatten(runs: list[Path], out: Path) -> None:
    """Move run_policy's timestamped run folder up into ``out``, so the dashboard lists the video directly.

    Drops run_policy's own metrics.json (num_episodes, success_rate: ours has both) and
    tests_junit.xml (a 50%-success check that is not the scenario's verdict).
    """
    for run_dir in runs:
        for path in run_dir.iterdir():
            if path.name in ("metrics.json", "tests_junit.xml"):
                path.unlink()
            else:
                path.replace(out / path.name)
        run_dir.rmdir()


def run_and_judge(name: str, out: Path) -> tuple[str | None, dict]:
    if name not in SCENARIOS:
        return f"unknown scenario {name!r}, expected one of: {', '.join(SCENARIOS)}", {}
    scenario = SCENARIOS[name]
    sim_s = scenario.env.get("episode_length_s", ShapeSortingEnvironmentCfg.episode_length_s)
    # Judge only the folder this run creates: a reused --out still holds earlier results, and
    # run_policy appends to episode_results_rank*.jsonl rather than replacing it.
    before = set(out.iterdir())
    returncode = run(scenario, out, _BOOT_S + _WALL_S_PER_SIM_S * sim_s * scenario.episodes)
    runs = sorted(path for path in set(out.iterdir()) - before if path.is_dir())
    verdict = judge(scenario, runs, returncode)
    flatten(runs, out)
    return verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # No argparse `choices`: a bad name must still leave a failing JUnit behind.
    parser.add_argument("scenario", help=f"One of: {', '.join(SCENARIOS)}")
    parser.add_argument("--out", type=Path, required=True, help="Upload dir for videos, tests_junit.xml, metrics.json.")
    parser.add_argument(
        "--metrics",
        type=Path,
        help="Also write metrics.json here. artefacts.yaml's `metrics:` is a fixed path, but the upload dir is a new temp dir each run.",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.metrics:
        args.metrics.unlink(missing_ok=True)  # never report the previous scenario's numbers

    start = time.monotonic()
    failure, metrics = "scenarios.py crashed (see the traceback in the log)", {}
    try:
        failure, metrics = run_and_judge(args.scenario, args.out)
    finally:
        # Even when crashing: Artefacts shows a run without a JUnit verdict as passed.
        wall_s = time.monotonic() - start
        metrics["wall_s"] = round(wall_s, 1)
        what = SCENARIOS[args.scenario].what if args.scenario in SCENARIOS else ""
        if "episodes" in metrics:  # the outcome, also when the verdict does not depend on it
            what += f"\nSorted every piece in {metrics['successes']} of {metrics['episodes']} episode(s)."
        case = TestCase(args.scenario, classname="curobo_scenarios", elapsed_sec=wall_s, stdout=what)
        if failure:
            case.add_failure_info(failure)
        with (args.out / "tests_junit.xml").open("w", encoding="utf-8") as f:
            TestSuite.to_file(f, [TestSuite("curobo_scenarios", [case])], prettyprint=True)
        for path in filter(None, [args.out / "metrics.json", args.metrics]):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(f"[scenarios] {args.scenario}: {failure or 'passed'} {metrics}")
    return 1 if failure else 0


if __name__ == "__main__":
    sys.exit(main())
