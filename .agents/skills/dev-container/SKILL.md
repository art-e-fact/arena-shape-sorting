---
name: dev-container
description: Sets up and manages the arena-shape-sorting Docker container — the environment used for running IsaacLab-Arena environments and scripts in this repo. Use when the user asks to set up the dev environment, bootstrap the project, get started on a fresh clone, build or rebuild the image, start or attach to the container, or run any command inside it. Also covers ./docker/run_docker.sh flag combinations (-r rebuild, -R rebuild without cache, -d/-m/-e custom dataset/model/eval mounts, -s container suffix, -S local isaaclab-so101 checkout), docker exec usage, and the /isaac-sim/python.sh aliasing.
allowed-tools: Bash(./docker/run_docker.sh *) Bash(docker exec *) Bash(docker images *) Bash(docker ps *)
---

# Dev Container

`arena-shape-sorting` runs inside a single Docker container built on top of Isaac Sim. The image bundles Isaac Lab, the vendored `IsaacLab-Arena` submodule, this repo's `arena_envs` package (installed editable), and the pinned `arena-so101` package from GitHub. The repo root is mounted into the container, so edits on the host take effect immediately.

Each clone of the repo on the host gets its own container, so separate clones can run in parallel. The image (`arena-shape-sorting:latest`) is shared; the container name is `arena-shape-sorting-latest-<suffix>`, where `run_docker.sh` derives `<suffix>` from the clone directory name automatically (override with `-s <suffix>`).

## Discover this clone's container (once per session)

Never hardcode the name. At the start of a session, resolve the container mounting this clone into `ARENA_CONTAINER` (empty result = none running, so start one below):

```bash
ARENA_CONTAINER=$(docker ps --filter "volume=$(git rev-parse --show-toplevel)" --format '{{.Names}}' | head -1)
```

Run the commands below in that same shell so `$ARENA_CONTAINER` stays set (or substitute the literal name).

## Start or attach

```bash
./docker/run_docker.sh
```

Idempotent: builds the (shared) image if it does not exist, starts this clone's container if it is not running, then attaches.

## Common flag combinations

| Flag | Purpose |
|---|---|
| `-r` | Force image rebuild |
| `-R` | Force image rebuild **without cache** |
| `-d <path>` | Mount a custom dataset directory (default `~/datasets` → `/datasets`) |
| `-m <path>` | Mount a custom model directory (default `~/models` → `/models`) |
| `-e <path>` | Mount a custom eval directory (default `~/eval` → `/eval`) |
| `-s <suffix>` | Override the container name suffix (run multiple containers) |
| `-S <path>` | Bind-mount a local isaaclab-so101 checkout and install it editable (see DEVELOPMENT.md) |

Example with custom mounts:

```bash
./docker/run_docker.sh -d ~/datasets -m ~/models -e ~/eval
```

## Run a command in the already-running container

```bash
docker exec "$ARENA_CONTAINER" su $(id -un) -c \
  "cd /workspaces/arena-shape-sorting && <command>"
```

The repo root is mounted at `/workspaces/arena-shape-sorting` inside the container. Run as the host user, not root.

## Python invocation

Inside the container, `python` is aliased to `/isaac-sim/python.sh`. Both forms work, but **prefer `/isaac-sim/python.sh` explicitly** in `docker exec` invocations from outside the container, where the alias is not active.

## Verify

A container is up and importable when:

```bash
docker exec "$ARENA_CONTAINER" su $(id -un) -c \
  "/isaac-sim/python.sh -c 'import isaaclab_arena, arena_envs, arena_so101; print(arena_envs.__file__); print(arena_so101.__file__)'"
```

`arena_envs` prints a path under `/workspaces/arena-shape-sorting/`. `arena_so101` prints Isaac Sim `site-packages` (git pin) or `/workspaces/arena-so101/` (when started with `-S`).
