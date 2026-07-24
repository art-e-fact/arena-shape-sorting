"""Teleop entry point for the shape-sorting environment.

Imports the env first so ``@register_environment`` runs, then hands off to Arena's
teleop script. That avoids the double-registration crash from combining
``@register_environment`` with ``--external_environment_class_path``.

Examples::

    # Absolute joint gamepad (recommended for SO-101)
    python -m shape_sorting.run_teleop \\
      --viz kit --num_envs 1 \\
      shape_sorting_test \\
      --embodiment so101_abs_joint \\
      --teleop_device gamepad

    # SE(3) — keyboard / spacemouse / gamepad
    python -m shape_sorting.run_teleop \\
      --viz kit --num_envs 1 \\
      shape_sorting_test \\
      --embodiment so101_ik \\
      --teleop_device keyboard

    # Physical SO-101 leader (absolute joints)
    python -m shape_sorting.run_teleop \\
      --viz kit --num_envs 1 \\
      shape_sorting_test \\
      --embodiment so101_abs_joint \\
      --teleop_device so101_leader \\
      --leader_port /dev/ttyACM0
"""

from __future__ import annotations

import runpy
from pathlib import Path


def _enable_isaaclab_env_ui_window() -> None:
    """Work around Isaac Lab 3.0 never setting ``/isaaclab/has_gui``.

    Without that flag, ``ManagerBasedEnv`` skips creating the IsaacLab debug
    window (Observations / Rewards / Terminations live plots) even when
    ``--viz kit`` is used. Patch AppLauncher after carb settings are
    initialized and before SimulationContext caches ``has_gui``.
    """
    from isaaclab.app import AppLauncher
    from isaaclab.app.settings_manager import get_settings_manager

    original_load_extensions = AppLauncher._load_extensions

    def _load_extensions_with_gui(self: AppLauncher) -> None:
        original_load_extensions(self)
        if not self._headless:
            get_settings_manager().set_bool("/isaaclab/has_gui", True)

    AppLauncher._load_extensions = _load_extensions_with_gui  # type: ignore[method-assign]


def main() -> None:
    # Register the env only — do NOT import shape_asset here. That module pulls in
    # ``isaaclab.sim`` / USD, which must not load before SimulationApp starts
    # (causes PhysxSchema.Tokens errors and a Kit segfault).
    import shape_sorting.shape_sorting_env  # noqa: F401 — triggers @register_environment

    import isaaclab_arena

    _enable_isaaclab_env_ui_window()

    teleop_script = (
        Path(isaaclab_arena.__file__).resolve().parent / "scripts" / "imitation_learning" / "teleop.py"
    )
    runpy.run_path(str(teleop_script), run_name="__main__")


if __name__ == "__main__":
    main()
