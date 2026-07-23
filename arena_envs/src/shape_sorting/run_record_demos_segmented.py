"""Segmented (plan → execute) demo recording for the shape-sorting environment.

Registers the env, then runs ``record_demos_segmented.py``. Prefer this over Arena's
``record_demos.py`` when you want checkpointed plan/execute teleop.

Examples::

    # SE(3) keyboard / gamepad
    python -m shape_sorting.run_record_demos_segmented \\
      --viz kit --device cpu \\
      --dataset_file ./so101_shape_sorting.hdf5 \\
      --num_demos 10 --num_success_steps 2 \\
      shape_sorting_test \\
      --embodiment so101_ik \\
      --teleop_device keyboard

    # SO-101 leader (absolute joints)
    python -m shape_sorting.run_record_demos_segmented \\
      --viz kit --device cpu \\
      --dataset_file ./so101_shape_sorting.hdf5 \\
      --num_demos 10 --num_success_steps 2 \\
      shape_sorting_test \\
      --embodiment so101_abs_joint \\
      --teleop_device so101_leader \\
      --leader_port /dev/ttyACM0
"""

from __future__ import annotations

import runpy
from pathlib import Path


def _enable_isaaclab_env_ui_window() -> None:
    """Work around Isaac Lab 3.0 never setting ``/isaaclab/has_gui``."""
    from isaaclab.app import AppLauncher
    from isaaclab.app.settings_manager import get_settings_manager

    original_load_extensions = AppLauncher._load_extensions

    def _load_extensions_with_gui(self: AppLauncher) -> None:
        original_load_extensions(self)
        if not self._headless:
            get_settings_manager().set_bool("/isaaclab/has_gui", True)

    AppLauncher._load_extensions = _load_extensions_with_gui  # type: ignore[method-assign]


def main() -> None:
    import shape_sorting.shape_sorting_env  # noqa: F401 — @register_environment

    _enable_isaaclab_env_ui_window()

    script = Path(__file__).resolve().parent / "record_demos_segmented.py"
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
