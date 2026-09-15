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
printf 'lerobot-train'; printf ' %q' "${TRAIN_ARGS[@]}"; printf '\n'
exec lerobot-train "${TRAIN_ARGS[@]}"
