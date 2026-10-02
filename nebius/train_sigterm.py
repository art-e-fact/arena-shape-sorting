#!/usr/bin/env python
"""`lerobot-train`, plus a checkpoint on SIGTERM.

Nebius stops a preempted (or cancelled, or timed-out) job with SIGTERM, 60 s, SIGKILL.
`lerobot-train` has no handler, so it dies where it stands and loses everything since the
last `--save_freq`. This wrapper makes the step after the signal a save step; when the run
has `--save_checkpoint_to_hub=true` the push lands on the Hub and the process exits 143, so
the job ends FAILED and `nebius/train.sh --supervise` resumes it from there.

Run inside the job by nebius/train_entrypoint.sh: python train_sigterm.py <lerobot-train args>
"""

import logging
import signal
import sys

import lerobot.scripts.lerobot_train as train

_stop = False


def _on_sigterm(signum, frame):
    global _stop
    _stop = True
    logging.warning("SIGTERM: saving a checkpoint after this step, then exiting 143")


_should_save = train.should_save_checkpoint
_push = train.push_checkpoint_to_hub


def should_save_checkpoint(step, save_freq, total_steps):
    return _stop or _should_save(step, save_freq, total_steps)


def push_checkpoint_to_hub(*args, **kwargs):
    _push(*args, **kwargs)
    if _stop:
        # ponytail: without --save_checkpoint_to_hub this is never reached and the run trains on
        # until SIGKILL — the local checkpoint dies with the job disk anyway. A save that lands
        # mid-eval (--eval_steps pass > 60 s) is also lost; the Hub keeps the previous one.
        logging.warning("SIGTERM: checkpoint pushed, exiting")
        sys.exit(143)


# The train loop looks these up in its own module namespace at call time.
train.should_save_checkpoint = should_save_checkpoint
train.push_checkpoint_to_hub = push_checkpoint_to_hub
signal.signal(signal.SIGTERM, _on_sigterm)

if __name__ == "__main__":
    sys.argv[0] = "lerobot-train"
    train.main()
