# Copyright (c) 2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""Catches scenario typos without Isaac Sim. Run: python test_scenarios.py

A misspelt override only fails inside the scenario run, minutes into booting Isaac Sim.
"""

from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shape_sorting.curobo_policy import CuroboPolicyCfg
from shape_sorting.scenarios import SCENARIOS, flags, flatten
from shape_sorting.shape_forms import ShapeForm
from shape_sorting.shape_sorting_env import ShapeSortingEnvironmentCfg


def test_overrides_name_real_config_fields():
    env_fields = {f.name for f in fields(ShapeSortingEnvironmentCfg)}
    curobo_fields = {f.name for f in fields(CuroboPolicyCfg)}
    for name, scenario in SCENARIOS.items():
        assert not scenario.env.keys() - env_fields, (name, scenario.env.keys() - env_fields)
        assert not scenario.curobo.keys() - curobo_fields, (name, scenario.curobo.keys() - curobo_fields)
        assert scenario.expect in ("success", "completes"), name


def test_forms_are_real_forms():
    for scenario in SCENARIOS.values():
        for form in scenario.env.get("forms", []):
            ShapeForm(form)


def test_flags():
    assert flags({"forms": ["star", "cube"], "debug_viz": True, "clearance": 0.001}) == [
        "--forms", "star", "cube", "--debug_viz", "--clearance", "0.001",
    ]  # fmt: skip


def test_flatten_lifts_the_run_folder_and_drops_run_policys_verdict(tmp_path=None):
    import tempfile

    out = Path(tmp_path or tempfile.mkdtemp())
    (out / "test_process_log.txt").write_text("artefacts' log")  # already there, must survive
    (out / "rl-video-step-0.mp4").write_text("previous run")
    run_dir = out / "2026-09-24_14-00-34"
    run_dir.mkdir()
    for name in ("rl-video-step-0.mp4", "episode_results_rank0.jsonl", "index.html", "metrics.json", "tests_junit.xml"):
        (run_dir / name).write_text(name)

    flatten([run_dir], out)

    assert sorted(p.name for p in out.iterdir()) == [
        "episode_results_rank0.jsonl", "index.html", "rl-video-step-0.mp4", "test_process_log.txt",
    ]  # fmt: skip
    assert (out / "rl-video-step-0.mp4").read_text() == "rl-video-step-0.mp4"  # this run's, not the previous one


if __name__ == "__main__":
    for test in (
        test_overrides_name_real_config_fields,
        test_forms_are_real_forms,
        test_flags,
        test_flatten_lifts_the_run_folder_and_drops_run_policys_verdict,
    ):
        test()
    print("ok")
