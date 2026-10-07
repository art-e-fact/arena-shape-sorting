#!/usr/bin/env bash
# Submit a lerobot training run to Nebius as a one-off job — no VM to create,
# no notebook to paste, nothing to remember to shut down. The job provisions a
# GPU, runs nebius/train_entrypoint.sh, and releases the GPU when the command
# exits (or when --timeout is reached, whichever comes first).
#
#   nebius/train.sh [options] -- <lerobot-train args...>
#
# Options (all optional):
#   --name NAME        job name prefix (default: lerobot-train)
#   --platform P       default: gpu-h100-sxm      (also: gpu-h200-sxm, gpu-l40s-d)
#   --preset P         default: 1gpu-16vcpu-200gb
#   --timeout D        default: 12h, max 168h — the run is killed at this point
#   --disk-size S      default: 250Gi
#   --env-file PATH    read HF_TOKEN / WANDB_API_KEY from a dotenv file
#                      (default: .env in the repo root, when it exists)
#   --secret NAME      MysteryBox secret to fall back on for whichever of those
#                      two is not in the env file (default: creds)
#   --entrypoint PATH  script to run in the container instead of
#                      train_entrypoint.sh (for sweeps, eval, one-offs)
#   --preemptible      spot-priced GPU the platform may stop at any time (SIGTERM, 60 s,
#                      SIGKILL; train_sigterm.py checkpoints in that window). Add --supervise
#   --resume           continue the run already in --policy.repo_id from its latest Hub
#                      checkpoint instead of starting over
#   --supervise        stay attached: whenever the job ends FAILED (preempted, timed out,
#                      crashed) resubmit it with --resume, until it completes. Gives up after
#                      two attempts in a row that added no checkpoint. Needs
#                      --save_checkpoint_to_hub=true
#   --follow           stream logs until the job ends instead of returning the id
#   --dry-run          validate the request without creating anything (free)
#
# Each token is resolved independently: the env file first, then the surrounding
# shell, then the MysteryBox secret. A token taken from the first two is sent as
# a plain --env value, which is stored in the job spec and readable by anyone
# with project access; MysteryBox keeps it out of the spec. Convenient for a
# personal token, worth avoiding for a shared one. The chosen source for each is
# printed at submit time, so it is never a silent decision.
#
# Anything the checkpoints need to survive must be pushed to the Hub: the job's
# disk goes away with the job. Pass --policy.push_to_hub=true and
# --save_checkpoint_to_hub=true so intermediate checkpoints are recoverable. They are
# also what --resume and --supervise continue from, so a fresh run refuses to start
# over another run's checkpoints/ (README: Hub versioning).
#
# Example:
#   nebius/train.sh --name smolvla --timeout 6h -- \
#     --policy.path=lerobot/smolvla_base \
#     --dataset.repo_id=Artefacts/shape-sorting-so101-30fps \
#     --policy.repo_id=Artefacts/smolvla-shape-sorting-30fps \
#     --policy.push_to_hub=true --save_checkpoint_to_hub=true \
#     --batch_size=64 --steps=20000 --save_freq=2000 --wandb.enable=true
set -euo pipefail

NAME=lerobot-train
PLATFORM=gpu-h100-sxm
PRESET=1gpu-16vcpu-200gb
TIMEOUT=12h
DISK=250Gi
SECRET=creds
IMAGE=python:3.12-slim
EXTRA=()
PREEMPTIBLE=false RESUME=false SUPERVISE=false FOLLOW=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) NAME="$2"; shift 2 ;;
    --platform) PLATFORM="$2"; shift 2 ;;
    --preset) PRESET="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --disk-size) DISK="$2"; shift 2 ;;
    --secret) SECRET="$2"; shift 2 ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --image) IMAGE="$2"; shift 2 ;;
    --entrypoint) ENTRYPOINT="$2"; shift 2 ;;
    --preemptible) EXTRA+=(--preemptible); PREEMPTIBLE=true; shift ;;
    --resume) RESUME=true; shift ;;
    --supervise) SUPERVISE=true; shift ;;
    --dry-run) EXTRA+=(--dry-run); shift ;;
    --follow) FOLLOW=true; shift ;;
    --) shift; break ;;
    -h|--help) sed -n '2,38p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "train.sh: unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "train.sh: no lerobot-train arguments given (everything after '--')" >&2
  exit 2
fi
REPO="$(printf '%s\n' "$@" | sed -n 's/^--policy\.repo_id=//p' | tail -1)"
SAVE_TO_HUB=false
printf '%s\n' "$@" | grep -qx -- '--save_checkpoint_to_hub=true' && SAVE_TO_HUB=true
if [[ "${RESUME}" == true || "${SUPERVISE}" == true ]] && [[ "${SAVE_TO_HUB}" != true || -z "${REPO}" ]]; then
  echo "train.sh: --resume and --supervise need --save_checkpoint_to_hub=true and --policy.repo_id=<repo> after '--'" >&2
  exit 2
fi

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENTRYPOINT="${ENTRYPOINT:-${HERE}/train_entrypoint.sh}"
ENV_FILE="${ENV_FILE-$(dirname "${HERE}")/.env}"

# Read one KEY from a dotenv file. Deliberately not `source`: an env file holding
# secrets should never be able to execute anything. Accepts an optional `export`
# prefix and single or double quotes, ignores blanks and comments, last wins.
dotenv_get() {  # dotenv_get FILE KEY
  [[ -f "$1" ]] || return 0
  sed -n -E "s/^[[:space:]]*(export[[:space:]]+)?$2[[:space:]]*=[[:space:]]*(.*)$/\2/p" "$1" |
    sed -E -e 's/[[:space:]]+$//' -e 's/^"(.*)"$/\1/' -e "s/^'(.*)'\$/\1/" | tail -1
}

AUTH_ARGS=()
HF_AUTH=()  # for the Hub checks below; empty works for a public repo
for VAR in HF_TOKEN WANDB_API_KEY; do
  VAL="$(dotenv_get "${ENV_FILE}" "${VAR}")"
  SRC="${ENV_FILE}"
  if [[ -z "${VAL}" ]]; then VAL="${!VAR-}"; SRC="shell"; fi
  if [[ -n "${VAL}" ]]; then
    AUTH_ARGS+=(--env "${VAR}=${VAL}")
    [[ "${VAR}" != HF_TOKEN ]] || HF_AUTH=(-H "Authorization: Bearer ${VAL}")
    echo "train.sh: ${VAR} from ${SRC} (sent as a plain job env var)"
  else
    AUTH_ARGS+=(--env-secret "${VAR}=${SECRET}")
    echo "train.sh: ${VAR} from MysteryBox secret ${SECRET}"
  fi
done
export PATH="${HOME}/.nebius/bin:${PATH}"
command -v nebius >/dev/null || { echo "train.sh: nebius CLI not on PATH" >&2; exit 1; }

# From CLI 0.12.278 a preemptible job also picks a pricing mode. Follow the spot price: a
# max-price policy cannot beat PAYG (the spot ceiling is below it) and can only stall the job.
if [[ "${PREEMPTIBLE}" == true ]] && nebius ai job create --help 2>/dev/null | grep -q -- '--follows-spot-price'; then
  EXTRA+=(--follows-spot-price)
fi

# One subnet per project is the normal case; the 'managed-' one belongs to a
# managed service, so it is never the right target for a job.
SUBNET="${NEBIUS_SUBNET_ID:-}"
if [[ -z "${SUBNET}" ]]; then
  mapfile -t SUBNETS < <(nebius vpc subnet list --format json |
    jq -r '.items[] | select(.metadata.name | startswith("managed-") | not) | .metadata.id')
  if [[ ${#SUBNETS[@]} -ne 1 ]]; then
    echo "train.sh: expected exactly one non-managed subnet, found ${#SUBNETS[@]}." >&2
    echo "  Pick one and re-run with NEBIUS_SUBNET_ID=<id>:" >&2
    nebius vpc subnet list --format json | jq -r '.items[] | "  \(.metadata.id)  \(.metadata.name)"' >&2
    exit 1
  fi
  SUBNET="${SUBNETS[0]}"
fi

# --args takes a single string, so the training arguments travel as a file:
# one per line, preserving any that contain spaces or quotes. A newline inside an
# argument would split it in two on the far side, so reject that here rather than
# let the job start with silently mangled arguments.
for a in "$@"; do
  [[ "$a" != *$'\n'* ]] || { echo "train.sh: argument contains a newline: ${a@Q}" >&2; exit 2; }
done
ARGS_FILE="$(mktemp -t lerobot-train-args.XXXXXX)"
trap 'rm -f "${ARGS_FILE}"' EXIT
printf '%s\n' "$@" > "${ARGS_FILE}"

submit() {  # submit [more nebius flags...] — sets JOB_ID (empty if nothing was created) and STATUS
  JOB_NAME="${NAME}-$(date -u +%Y%m%d-%H%M%S)"
  echo "train.sh: submitting ${JOB_NAME} (${PLATFORM} ${PRESET}, timeout ${TIMEOUT})"
  OUT="$(nebius ai job create \
    --name "${JOB_NAME}" \
    --image "${IMAGE}" \
    --platform "${PLATFORM}" \
    --preset "${PRESET}" \
    --subnet-id "${SUBNET}" \
    --timeout "${TIMEOUT}" \
    --disk-size "${DISK}" \
    --restart-policy never \
    "${AUTH_ARGS[@]}" \
    --inject-file "${ENTRYPOINT}:/opt/train_entrypoint.sh" \
    --inject-file "${HERE}/train_sigterm.py:/opt/train_sigterm.py" \
    --inject-file "${ARGS_FILE}:/opt/train_args" \
    --container-command bash \
    --args /opt/train_entrypoint.sh \
    "${EXTRA[@]}" "$@" 2>&1)" && STATUS=0 || STATUS=$?
  printf '%s\n' "${OUT}"
  # The id is printed even when startup fails, and it is the only way to reach the
  # logs that say why — so recover it before honouring a non-zero exit.
  JOB_ID="$(printf '%s' "${OUT}" | grep -oE 'aijob-[a-z0-9]+' | head -1 || true)"
  [[ -z "${JOB_ID}" ]] || printf '\n  logs:   nebius ai job logs %s --follow\n  status: nebius ai job get %s --format json | jq .status\n  cancel: nebius ai job cancel %s\n' \
    "${JOB_ID}" "${JOB_ID}" "${JOB_ID}"
}

# Highest checkpoints/<step> in the Hub model repo, or "" (no repo, no checkpoints, no access).
latest_checkpoint() {
  curl -fsS "${HF_AUTH[@]}" "https://huggingface.co/api/models/${REPO}/tree/main/checkpoints" 2>/dev/null |
    jq -r '[.[].path | split("/")[-1] | select(test("^[0-9]+$"))] | max_by(tonumber) // ""' || true
}

RESUME_ENV=()
if [[ "${RESUME}" == true ]]; then
  RESUME_ENV=(--env TRAIN_RESUME=1)
elif [[ "${SAVE_TO_HUB}" == true && -n "${REPO}" ]] && LATEST="$(latest_checkpoint)" && [[ -n "${LATEST}" ]]; then
  # The entrypoint refuses this too, but only after five minutes of GPU time.
  echo "train.sh: ${REPO} already holds checkpoints/${LATEST} from an earlier run; pass --resume to" >&2
  echo "  continue it, or tag that run and delete its checkpoints/ first (README: Hub versioning)." >&2
  exit 1
fi

submit "${RESUME_ENV[@]}"
[[ -n "${JOB_ID}" ]] || exit "${STATUS}"   # --dry-run, or it failed before creating anything

if [[ "${SUPERVISE}" != true ]]; then
  if [[ "${FOLLOW}" == true ]]; then exec nebius ai job logs "${JOB_ID}" --follow; fi
  exit 0
fi

# --supervise: a FAILED or ERROR end — preempted, timed out and crashed all look the same — is
# resubmitted to resume from the Hub. Two attempts in a row that left no new checkpoint there
# is not preemption, so stop rather than pay for a crash loop (or wait out a capacity drought).
echo "train.sh: supervising; Ctrl-C stops this watcher only, the job keeps running"
BEFORE="$(latest_checkpoint)"
NO_PROGRESS=0
while :; do
  if [[ "${FOLLOW}" == true ]]; then nebius ai job logs "${JOB_ID}" --follow || true; fi
  while :; do
    STATE="$(nebius ai job get "${JOB_ID}" --format json 2>/dev/null | jq -r '.status.state' || true)"
    case "${STATE}" in COMPLETED|CANCELLED|FAILED|ERROR) break ;; esac
    sleep 60
  done
  echo "train.sh: ${JOB_ID} ended ${STATE} at $(date -u +%H:%M:%S) UTC"
  case "${STATE}" in
    COMPLETED) exit 0 ;;
    CANCELLED) echo "train.sh: cancelled, not resubmitting (re-run with --resume to continue)"; exit 1 ;;
  esac
  AFTER="$(latest_checkpoint)"
  if [[ -n "${AFTER}" && "${AFTER}" != "${BEFORE}" ]]; then NO_PROGRESS=0; else NO_PROGRESS=$((NO_PROGRESS + 1)); fi
  if (( NO_PROGRESS >= 2 )); then
    echo "train.sh: two attempts in a row without a new checkpoint — not preemption, giving up." >&2
    echo "  nebius ai job logs ${JOB_ID}" >&2
    exit 1
  fi
  BEFORE="${AFTER}"
  RESUME_ENV=()
  [[ -z "${AFTER}" ]] || RESUME_ENV=(--env TRAIN_RESUME=1)
  echo "train.sh: resubmitting${AFTER:+ to resume from checkpoint ${AFTER}}"
  submit "${RESUME_ENV[@]}"
  [[ -n "${JOB_ID}" ]] || exit "${STATUS}"
done
