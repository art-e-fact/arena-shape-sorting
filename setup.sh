#!/usr/bin/env bash
# Host (non-Docker) environment for arena-shape-sorting.
#
# Syncs IsaacLab-Arena's uv venv (Isaac Lab + Arena), installs our packages
# editable into it, then activates that venv in the current shell.
#
# Must be sourced so activation sticks:
#   source ./setup.sh
#   # or: . ./setup.sh
#
# After that, use the same commands as in Docker, e.g.:
#   python -m shape_sorting.run_record_demos_segmented ...
#
# Options (pass after sourcing, e.g. `source ./setup.sh --wheel`):
#   --wheel   Use Arena's isaaclab-from-wheel group instead of from-source
#   --force   Re-run uv sync even if the venv already exists
#   -h/--help Show this help
#
# Optional env: ARENA_SO101_PATH — local isaaclab-so101 checkout (see DEVELOPMENT.md).

_setup_main() {
  local REPO_ROOT ARENA_DIR VENV_DIR FORCE=false WHEEL=false
  local UV_SYNC_ARGS=()

  # Resolve repo root whether sourced or executed.
  if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
    REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  else
    REPO_ROOT="$(cd -- "$(dirname -- "$0")" && pwd)"
  fi
  ARENA_DIR="${REPO_ROOT}/submodules/IsaacLab-Arena"
  VENV_DIR="${ARENA_DIR}/.venv"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --wheel) WHEEL=true ;;
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

  if ! command -v uv >/dev/null 2>&1; then
    echo "setup.sh: uv not found on PATH. Install: https://docs.astral.sh/uv/" >&2
    return 1
  fi

  if [[ "${WHEEL}" == true ]]; then
    UV_SYNC_ARGS=(--no-default-groups --group isaaclab-from-wheel)
  fi

  # Sync on first use, when --force, or when switching to the wheel flavor.
  if [[ "${FORCE}" == true || "${WHEEL}" == true || ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "setup.sh: syncing Arena environment in ${ARENA_DIR} ..."
    (cd "${ARENA_DIR}" && uv sync "${UV_SYNC_ARGS[@]}") || return 1
  else
    echo "setup.sh: Arena venv already present (pass --force to re-sync)"
  fi

  # uv venvs do not ship pip; install into Arena's env with uv pip.
  # arena_envs pulls the pinned arena-so101 git dependency.
  echo "setup.sh: installing arena_envs (pulls pinned arena-so101) ..."
  uv pip install --python "${VENV_DIR}/bin/python" \
    -e "${REPO_ROOT}/arena_envs" || return 1

  if [[ -n "${ARENA_SO101_PATH:-}" ]]; then
    if [[ ! -d "${ARENA_SO101_PATH}" ]]; then
      echo "setup.sh: ARENA_SO101_PATH is not a directory: ${ARENA_SO101_PATH}" >&2
      return 1
    fi
    echo "setup.sh: reinstalling arena-so101 editable from ${ARENA_SO101_PATH} ..."
    uv pip install --python "${VENV_DIR}/bin/python" \
      -e "${ARENA_SO101_PATH}[leader]" || return 1
  fi

  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate" || return 1

  export OMNI_KIT_ACCEPT_EULA=YES
  export ACCEPT_EULA=Y
  export ARENA_SHAPE_SORTING_ROOT="${REPO_ROOT}"
  # Console scripts (lerobot-eval) do not put cwd on sys.path, so envhub is not
  # importable unless the repo root is on PYTHONPATH.
  case ":${PYTHONPATH:-}:" in
    *":${REPO_ROOT}:"*) ;;
    *) export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" ;;
  esac

  cd "${REPO_ROOT}" || return 1

  echo "setup.sh: ready — python=$(command -v python)"
  echo "  Same demo commands as Docker work from here (python -m shape_sorting...)."
}

# Refuse bare execution: activation must apply to the caller's shell.
if [[ "${BASH_SOURCE[0]:-}" == "${0}" ]]; then
  echo "setup.sh: source this script instead of executing it:" >&2
  echo "  source ./setup.sh" >&2
  exit 1
fi

_setup_main "$@"
unset -f _setup_main
