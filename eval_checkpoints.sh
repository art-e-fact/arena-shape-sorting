#!/usr/bin/env bash
# Roll out a set of checkpoints in the sim env and report success rate per checkpoint.
#
#   ./eval_checkpoints.sh [--n-episodes N] [--batch-size N] [--viz kit] [--repo REPO] STEP...
#
# Held-out loss is a weak proxy for task success on a flow-matching policy, so the
# only way to pick a checkpoint is to roll it out. Checkpoints are downloaded from
# the Hub into ./ckpt (skipped if already there) and evaluated one at a time —
# Isaac Sim boots per checkpoint, which costs ~2 min each.
#
# A checkpoint that fails records FAILED and the sweep continues, so one bad entry
# does not lose the rest of the grid. Results land in outputs/eval/checkpoints/summary.tsv.
#
# `source ./setup.sh` first: lerobot-eval is a console script and does not put cwd
# on sys.path, so --env.discover_packages_path=envhub needs the repo root on PYTHONPATH.
set -euo pipefail

N_EPISODES=30
BATCH_SIZE=5
VIZ=""
REPO=Artefacts/smolvla-shape-sorting-30fps

while [[ $# -gt 0 ]]; do
  case "$1" in
    --n-episodes) N_EPISODES="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --viz) VIZ="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    -h|--help) sed -n '2,16p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) break ;;
  esac
done
[[ $# -gt 0 ]] || { echo "eval_checkpoints.sh: no checkpoint steps given (try --help)" >&2; exit 2; }

command -v lerobot-eval >/dev/null || { echo "eval_checkpoints.sh: run 'source ./setup.sh' first" >&2; exit 1; }

OUT=outputs/eval/checkpoints
mkdir -p "${OUT}"
SUMMARY="${OUT}/summary.tsv"
printf 'checkpoint\tn_episodes\tpc_success\tavg_max_reward\teval_s\n' > "${SUMMARY}"

RENAME='{"observation.images.camera_ego_rgb": "observation.images.ego_view", "observation.images.external_camera_rgb": "observation.images.exterior_image"}'

for STEP in "$@"; do
  DIR="ckpt/checkpoints/${STEP}/pretrained_model"
  if [[ ! -d "${DIR}" ]]; then
    echo "eval_checkpoints.sh: fetching ${STEP} from ${REPO} ..."
    python -c "
import sys
from huggingface_hub import snapshot_download
snapshot_download('${REPO}', allow_patterns=['checkpoints/${STEP}/pretrained_model/*'], local_dir='ckpt')
" || { printf '%s\t%s\tFAILED\t-\t-\n' "${STEP}" "${N_EPISODES}" >> "${SUMMARY}"; continue; }
  fi

  RUN_OUT="${OUT}/${STEP}"
  echo "eval_checkpoints.sh: evaluating ${STEP} (n=${N_EPISODES}, batch=${BATCH_SIZE}) ..."
  # shellcheck disable=SC2086
  if lerobot-eval \
      --policy.path="${DIR}" \
      --policy.device=cuda \
      --output_dir="${RUN_OUT}" \
      --env.discover_packages_path=envhub \
      --env.type=shape_sorting_arena \
      ${VIZ:+--env.visualizer=${VIZ}} \
      --rename_map="${RENAME}" \
      --eval.batch_size="${BATCH_SIZE}" \
      --eval.n_episodes="${N_EPISODES}" >"${OUT}/${STEP}.log" 2>&1; then
    python - "${STEP}" "${RUN_OUT}/eval_info.json" "${SUMMARY}" <<'PY'
import json, sys
step, path, summary = sys.argv[1], sys.argv[2], sys.argv[3]
o = json.load(open(path))["overall"]
with open(summary, "a") as f:
    f.write(f"{step}\t{o['n_episodes']}\t{o['pc_success']}\t{o['avg_max_reward']}\t{o['eval_s']:.0f}\n")
print(f"  {step}: pc_success={o['pc_success']}% over {o['n_episodes']} episodes")
PY
  else
    echo "  ${STEP}: FAILED (see ${OUT}/${STEP}.log)"
    printf '%s\t%s\tFAILED\t-\t-\n' "${STEP}" "${N_EPISODES}" >> "${SUMMARY}"
  fi
done

echo
echo "=== ${SUMMARY} ==="
cat "${SUMMARY}"
