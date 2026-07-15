# arena_envs/src/arena_envs/run_policy.py
import json
from pathlib import Path

import isaaclab_arena.evaluation.policy_runner as policy_runner
from isaaclab_arena.metrics.metrics_logger import metrics_to_plain_python_types
from isaaclab_arena.visualization.report import build_report as _build_report
from junit_xml import TestCase, TestSuite

_last_metrics = None
_original_rollout_policy = policy_runner.rollout_policy
_SUCCESS_RATE_THRESHOLD = 0.5


def _rollout_policy_with_capture(env, policy, num_steps, num_episodes):
    global _last_metrics
    metrics = _original_rollout_policy(env, policy, num_steps, num_episodes)
    _last_metrics = metrics
    return metrics


def _write_junit_report(metrics: dict[str, int | float | list], output_path: Path) -> None:
    success_rate = metrics.get("success_rate")
    test_case = TestCase("success_rate_above_50_percent", classname="shape_sorting.run_policy")
    if success_rate is None:
        test_case.add_failure_info("success_rate metric is missing")
    elif success_rate <= _SUCCESS_RATE_THRESHOLD:
        test_case.add_failure_info(
            f"success_rate {success_rate:.4f} is not above {_SUCCESS_RATE_THRESHOLD:.0%}"
        )
    else:
        test_case.stdout = f"success_rate={success_rate:.4f} (threshold>{_SUCCESS_RATE_THRESHOLD:.0%})"

    test_suite = TestSuite("policy_evaluation", [test_case])
    with output_path.open("w", encoding="utf-8") as junit_file:
        TestSuite.to_file(junit_file, [test_suite], prettyprint=True)


def _build_report_with_metrics(output_dir, *args, **kwargs):
    report_path = _build_report(output_dir, *args, **kwargs)
    if _last_metrics is not None and policy_runner.get_local_rank() == 0:
        plain = metrics_to_plain_python_types(_last_metrics)
        output_path = Path(output_dir)
        metrics_path = output_path / "metrics.json"
        metrics_path.write_text(json.dumps(plain, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote aggregated metrics to {metrics_path}")

        junit_path = output_path / "tests_junit.xml"
        _write_junit_report(plain, junit_path)
        print(f"Wrote JUnit report to {junit_path}")
    return report_path


policy_runner.rollout_policy = _rollout_policy_with_capture
policy_runner.build_report = _build_report_with_metrics


def main():
    # import shape_sorting.interop  # noqa: F401 — register custom envs
    import shape_sorting.shape_sorting_env  # noqa: F401 — triggers @register_environment
    policy_runner.main()


if __name__ == "__main__":
    main()