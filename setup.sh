#!/usr/bin/env bash
# Host (non-Docker) environment for arena-shape-sorting.
#
# Syncs IsaacLab-Arena's uv venv (Isaac Lab + Arena), installs cuRobo and our
# packages editable into it, then activates that venv in the current shell.
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
#   --force   Re-run uv sync and regenerate the cuRobo SO-101 config
#   -h/--help Show this help
#
# Optional env: ARENA_SO101_PATH — local isaaclab-so101 checkout (see DEVELOPMENT.md).

_setup_main() {
  local REPO_ROOT ARENA_DIR CUROBO_DIR VENV_DIR CUDA_EXTRA CUROBO_YML FORCE=false WHEEL=false
  local UV_SYNC_ARGS=()

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

  if [[ ! -f "${CUROBO_DIR}/pyproject.toml" ]]; then
    echo "setup.sh: cuRobo submodule missing at ${CUROBO_DIR}" >&2
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

  if [[ -n "${ARENA_SO101_PATH:-}" ]]; then
    if [[ ! -d "${ARENA_SO101_PATH}" ]]; then
      echo "setup.sh: ARENA_SO101_PATH is not a directory: ${ARENA_SO101_PATH}" >&2
      return 1
    fi
    echo "setup.sh: reinstalling arena-so101 editable from ${ARENA_SO101_PATH} ..."
    uv pip install --python "${VENV_DIR}/bin/python" \
      -e "${ARENA_SO101_PATH}[leader]" || return 1
  fi

  export OMNI_KIT_ACCEPT_EULA=YES
  export ACCEPT_EULA=Y

  # so101.yml is generated, not shipped: it lands inside the installed arena_so101
  # package, so deleting the venv deletes it too. CuroboPolicy only loads it once a
  # rollout is already running, so a missing file surfaces as a crash minutes in —
  # generate it here instead. Needs headless Isaac Sim (USD→URDF) + CUDA (sphere fit).
  CUROBO_YML="$("${VENV_DIR}/bin/python" -c \
    'from shape_sorting.curobo_motion import _DEFAULT_ROBOT_YML as p; print(p)')" || return 1
  if [[ "${FORCE}" == true || ! -f "${CUROBO_YML}" ]]; then
    echo "setup.sh: generating cuRobo SO-101 config (a few minutes) -> ${CUROBO_YML}"
    "${VENV_DIR}/bin/python" -m arena_so101.generate_curobo_config --headless || return 1
  else
    echo "setup.sh: cuRobo SO-101 config present (pass --force to regenerate)"
  fi

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
