"""nebius/train_sigterm.py: SIGTERM makes the next step a save step, and the push then ends the run."""

from __future__ import annotations

import importlib.util
import os
import signal
import time
from pathlib import Path

import pytest

WRAPPER = Path(__file__).resolve().parents[2] / "nebius" / "train_sigterm.py"


def test_sigterm_saves_pushes_and_exits_143(monkeypatch):
    import lerobot.scripts.lerobot_train as train  # the real module: the names the wrapper patches must exist

    pushed = []
    monkeypatch.setattr(train, "push_checkpoint_to_hub", lambda *a, **k: pushed.append(a))
    original_should_save = train.should_save_checkpoint
    spec = importlib.util.spec_from_file_location("train_sigterm", WRAPPER)
    wrapper = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(wrapper)
        assert train.should_save_checkpoint is wrapper.should_save_checkpoint

        # Before the signal: the periodic schedule is untouched and a push does not exit.
        assert train.should_save_checkpoint(1, 2000, 32000) is False
        assert train.should_save_checkpoint(2000, 2000, 32000) is True
        train.push_checkpoint_to_hub("ckpt", "repo")
        assert pushed == [("ckpt", "repo")]

        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(0.01)  # the handler runs on the main thread, between bytecodes
        assert train.should_save_checkpoint(1, 2000, 32000) is True
        with pytest.raises(SystemExit) as exc:
            train.push_checkpoint_to_hub("ckpt", "repo")
        assert exc.value.code == 143
        assert pushed == [("ckpt", "repo")] * 2  # pushed first, exited after
    finally:
        train.should_save_checkpoint = original_should_save
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
