"""Turn a lerobot-eval run into what Artefacts reads: numeric metrics and tests_junit.xml.

Artefacts ignores exit codes and reads ``metrics:`` from a fixed path, while lerobot-eval
writes ``eval_info.json`` into the per-run upload dir. Without this a crashed eval shows
as passed with no metrics.

    python envhub/scripts/artefacts_report.py "$ARTEFACTS_SCENARIO_UPLOAD_DIR" --metrics outputs/eval_metrics.json
"""

import argparse
import json
from pathlib import Path

from junit_xml import TestCase, TestSuite

parser = argparse.ArgumentParser()
parser.add_argument("out", type=Path, help="lerobot-eval --output_dir")
parser.add_argument("--metrics", type=Path, required=True)
args = parser.parse_args()

info_path = args.out / "eval_info.json"
case = TestCase("lerobot_eval", classname="eval")
metrics = {}
if info_path.exists():
    overall = json.loads(info_path.read_text())["overall"]
    metrics = {k: v for k, v in overall.items() if isinstance(v, (int, float))}
    case.stdout = json.dumps(metrics)
else:
    case.add_failure_info("lerobot-eval wrote no eval_info.json; it crashed (see test_process_log.txt)")

with (args.out / "tests_junit.xml").open("w", encoding="utf-8") as f:
    TestSuite.to_file(f, [TestSuite("eval", [case])], prettyprint=True)
args.metrics.parent.mkdir(parents=True, exist_ok=True)
args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
print(f"[artefacts_report] {'passed' if metrics else 'failed'} {metrics}")
