#!/usr/bin/env bash
# Runs inside the Nebius job container; submitted by nebius/train.sh.
#
# The container starts from a bare python image, so lerobot is installed here at
# job start (~4 GB of wheels, a few minutes). That keeps the launcher free of any
# image-build step; if the wait becomes annoying, bake this into an image pushed
# to the project registry and drop the install block.
#
# Everything that varies per run arrives as arguments in /opt/train_args, one per
# line, written by the launcher — not as env vars, so quoting survives intact.
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN missing — check the --env-secret wiring in nebius/train.sh}"

echo "=== node ==="
nvidia-smi || echo "no GPU visible"
df -h / | tail -1
echo "============"

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -yqq --no-install-recommends git ffmpeg >/dev/null

pip install --root-user-action=ignore -q uv
uv pip install --system -q \
  "lerobot[training,groot,smolvla]${LEROBOT_VERSION:+==${LEROBOT_VERSION}}"

python - <<'PY'
import torch, lerobot
print(f"lerobot {lerobot.__version__} | torch {torch.__version__} | cuda {torch.cuda.is_available()}")
PY
# A GPU job that silently falls back to CPU would burn the whole timeout before
# anyone noticed, so fail here instead.
if nvidia-smi >/dev/null 2>&1; then
  python -c "import torch; assert torch.cuda.is_available(), 'GPU present but torch cannot see it'"
fi

mapfile -t TRAIN_ARGS < /opt/train_args

# Checkpoints on the Hub (--save_checkpoint_to_hub=true) make a run resumable, so the repo
# must hold only this run's. TRAIN_RESUME=1 (train.sh --resume, or a --supervise resubmit)
# continues from the latest one; a fresh start refuses to train over another run's, because
# a later resume would silently pick that run's higher step.
REPO="$(sed -n 's/^--policy\.repo_id=//p' /opt/train_args | tail -1)"
if grep -qx -- '--save_checkpoint_to_hub=true' /opt/train_args && [[ -n "${REPO}" ]]; then
  if [[ "${TRAIN_RESUME:-}" == 1 ]]; then
    CONFIG="$(python - "${REPO}" <<'PY'
import sys
from pathlib import Path
from lerobot.common.train_utils import resolve_resume_checkpoint
# Not --config_path=<repo>: on a reused repo that prefers the root train_config.json, which
# is the previous completed run's. The checkpoint's own config is the one to resume.
ckpt = resolve_resume_checkpoint(sys.argv[1], Path("outputs/train/resume"))
print(ckpt / "pretrained_model" / "train_config.json")
PY
)"
    echo "resuming from ${CONFIG}"
    TRAIN_ARGS=(--resume=true "--config_path=${CONFIG}")
  else
    python - "${REPO}" <<'PY'
import sys
from huggingface_hub.errors import RepositoryNotFoundError
from lerobot.utils.hub import find_latest_hub_checkpoint
try:
    latest = find_latest_hub_checkpoint(sys.argv[1])
except RepositoryNotFoundError:
    latest = None
if latest:
    sys.exit(
        f"{sys.argv[1]} already holds {latest} from an earlier run, and a resume would pick it up. "
        "Pass --resume to continue that run, or tag it and delete its checkpoints/ first "
        "(README: Hub versioning)."
    )
PY
  fi
fi
printf 'lerobot-train'; printf ' %q' "${TRAIN_ARGS[@]}"; printf '\n'
# Through the wrapper, not lerobot-train: it checkpoints on SIGTERM (preemption).
exec python /opt/train_sigterm.py "${TRAIN_ARGS[@]}"
