#!/usr/bin/env bash
# Environment for arena-shape-sorting.
#
# Syncs IsaacLab-Arena's uv venv (Isaac Lab + Arena), installs cuRobo and our
# packages editable into it, then activates that venv in the current shell.
#
# Must be sourced so activation sticks:
#   source ./setup.sh
#   # or: . ./setup.sh
#
# After that, e.g.:
#   python -m shape_sorting.run_record_demos_segmented ...
#
# Options (pass after sourcing, e.g. `source ./setup.sh --force`):
#   --force   Re-run uv sync (after a submodule or pin bump)
#   -h/--help Show this help
#
# Optional env: ARENA_SO101_PATH — local isaaclab-so101 checkout (see DEVELOPMENT.md).

_setup_main() {
  local REPO_ROOT ARENA_DIR CUROBO_DIR VENV_DIR CUDA_EXTRA FORCE=false

  # Resolve repo root whether sourced or executed.
  if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
    REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  else
    REPO_ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
  fi
  ARENA_DIR="${REPO_ROOT}/submodules/IsaacLab-Arena"
  CUROBO_DIR="${REPO_ROOT}/submodules/curobo"
  VENV_DIR="${ARENA_DIR}/.venv"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --force) FORCE=true ;;
      -h|--help)
        sed -n '2,20p' "${BASH_SOURCE[0]:-$0}" | sed 's/^# \?//'
        return 0
        ;;
      *)
        echo "setup.sh: unknown option: $1 (try --help)" >&2
        return 2
        ;;
    esac
    shift
  done

  if [[ ! -f "${ARENA_DIR}/pyproject.toml" ]]; then
    echo "setup.sh: Arena submodule missing at ${ARENA_DIR}" >&2
    echo "  Run: git submodule update --init --recursive" >&2
    return 1
  fi

  if [[ ! -f "${CUROBO_DIR}/pyproject.toml" ]]; then
    echo "setup.sh: cuRobo submodule missing at ${CUROBO_DIR}" >&2
    echo "  Run: git submodule update --init --recursive" >&2
    return 1
  fi

  if ! command -v uv >/dev/null 2>&1; then
    echo "setup.sh: uv not found on PATH. Install: https://docs.astral.sh/uv/" >&2
    return 1
  fi

  # Sync on first use or when --force. uv sync removes what Arena's lock doesn't list, so
  # cuRobo and our packages go back in below after every sync.
  if [[ "${FORCE}" == true || ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "setup.sh: syncing Arena environment in ${ARENA_DIR} ..."
    (cd "${ARENA_DIR}" && uv sync) || return 1
  else
    echo "setup.sh: Arena venv already present (pass --force to re-sync)"
  fi

  # cuRobo: imported at runtime by shape_sorting.curobo_policy / curobo_motion.
  # Not in arena_envs' dependencies because it is vendored as a submodule.
  # No --no-build-isolation needed: this version JIT-compiles its kernels through
  # cuda.core, so the build needs setuptools only, not the installed torch.
  # The cuda-core extra must match the CUDA that the venv's torch was built against.
  CUDA_EXTRA="cu$("${VENV_DIR}/bin/python" -c \
    'import torch; print((torch.version.cuda or "12").split(".")[0])')" || return 1
  echo "setup.sh: installing cuRobo (${CUDA_EXTRA}) from ${CUROBO_DIR} ..."
  uv pip install --python "${VENV_DIR}/bin/python" \
    -e "${CUROBO_DIR}[${CUDA_EXTRA}]" || return 1

  # uv venvs do not ship pip; install into Arena's env with uv pip.
  # arena_envs pulls the pinned arena-so101 git dependency.
  echo "setup.sh: installing arena_envs (pulls pinned arena-so101) ..."
  uv pip install --python "${VENV_DIR}/bin/python" \
    -e "${REPO_ROOT}/arena_envs" || return 1

  # lerobot[smolvla] requires num2words, but Arena's lock does not list it and
  # `uv sync` removes it — so a --force re-sync silently breaks `lerobot-eval` on a
  # SmolVLA policy. transformers' SmolVLM processor raises ImportError at policy
  # load, and Isaac Sim's shutdown swallows it into **exit code 0**, so the run
  # looks like it passed. Pinned to lerobot's own range.
  echo "setup.sh: installing num2words (SmolVLM processor; dropped by uv sync) ..."
  uv pip install --python "${VENV_DIR}/bin/python" \
    "num2words>=0.5.14,<0.6.0" || return 1

  if [[ -n "${ARENA_SO101_PATH:-}" ]]; then
    if [[ ! -d "${ARENA_SO101_PATH}" ]]; then
      echo "setup.sh: ARENA_SO101_PATH is not a directory: ${ARENA_SO101_PATH}" >&2
      return 1
    fi
    echo "setup.sh: reinstalling arena-so101 editable from ${ARENA_SO101_PATH} ..."
    uv pip install --python "${VENV_DIR}/bin/python" \
      -e "${ARENA_SO101_PATH}[leader,lerobot]" || return 1
  fi

  export OMNI_KIT_ACCEPT_EULA=YES
  export ACCEPT_EULA=Y

  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate" || return 1

  export ARENA_SHAPE_SORTING_ROOT="${REPO_ROOT}"
  # Console scripts (lerobot-eval) do not put cwd on sys.path, so envhub is not
  # importable unless the repo root is on PYTHONPATH.
  case ":${PYTHONPATH:-}:" in
    *":${REPO_ROOT}:"*) ;;
    *) export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
  esac

  cd "${REPO_ROOT}" || return 1

  echo "setup.sh: ready — python=$(command -v python)"
}

# Refuse bare execution: activation must apply to the caller's shell.
if [[ "${BASH_SOURCE[0]:-}" == "${0}" ]]; then
  echo "setup.sh: source this script instead of executing it:" >&2
  echo "  source ./setup.sh" >&2
  exit 1
fi

_setup_main "$@"
unset -f _setup_main
