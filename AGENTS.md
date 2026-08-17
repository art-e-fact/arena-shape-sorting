# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, OpenAI Codex, etc.) when working with code in this repository.

## Project

`arena-shape-sorting` is a project built **on top of** IsaacLab-Arena, following the "Arena in your repository" pattern: Arena is vendored unmodified as a git submodule (`submodules/IsaacLab-Arena`) and extended purely through its registration API. Our own code lives in the `arena_envs` package, which defines custom environments and registers them with Arena.

We consume Arena and the SO-101 embodiment as dependencies — we do **not** develop them here. Treat everything under `submodules/IsaacLab-Arena` as read-only third-party code. SO-101 (`arena_so101`) comes from [art-e-fact/isaaclab-so101](https://github.com/art-e-fact/isaaclab-so101), pinned in `arena_envs/pyproject.toml`. Local editable overrides are documented in [DEVELOPMENT.md](DEVELOPMENT.md).

## Skill library

Multi-step workflows are captured as Agent Skills under `.agents/skills/`. When a task matches a skill, prefer invoking it over re-deriving the procedure from this file. Currently:

- `dev-container` — build, start, attach to, or exec into this repo's Docker container.

## Docker environment

Anything that touches Isaac Sim or Arena (running environments, scripts, evaluation) runs inside this repo's Docker container. The image bundles Isaac Sim, Isaac Lab, the Arena submodule, our `arena_envs` package (installed editable), and the pinned `arena-so101` package from GitHub. The repo root is mounted at `/workspaces/arena-shape-sorting`, so edits on the host take effect immediately.

Each clone gets its own container (shared image `arena-shape-sorting:latest`, per-clone container name). **Don't hardcode the container name** — use the `dev-container` skill to build, start, attach to, discover, or exec into it.

Start or attach (idempotent — builds the image if missing):

```bash
./docker/run_docker.sh
```

Run a command in the already-running container as the host user (not root):

```bash
docker exec "$ARENA_CONTAINER" su $(id -un) -c \
  "cd /workspaces/arena-shape-sorting && <command>"
```

Inside the container, `python` is aliased to `/isaac-sim/python.sh` — prefer the explicit path in `docker exec` invocations from outside, where the alias is not active.

A non-Docker host install is `source ./setup.sh`. Contributor workflows (SO-101 pin, local checkout, Docker `-S`) are in [DEVELOPMENT.md](DEVELOPMENT.md).

## Repository layout

- `arena_envs/` — our package: custom environments that subclass Arena base classes and register via `@register_environment` (`src/arena_envs/` / `src/shape_sorting/`)
- `docker/` — container build (`Dockerfile`) and run (`run_docker.sh`) scripts
- `submodules/IsaacLab-Arena/` — vendored Arena submodule (read-only), which itself vendors Isaac Lab under `submodules/IsaacLab`
- `DEVELOPMENT.md` — pin bumps, local `arena-so101` checkout, Docker `-S`

## Defining and running environments

Custom environments subclass an Arena base class (e.g. `ExampleEnvironmentBase`), set a unique `name`, and implement `get_env()` / `add_cli_args()`. They register themselves on import via `@register_environment`. See `arena_envs/src/arena_envs/environment.py` for the reference example.

Run an external environment through Arena's policy runner, pointing `--external_environment_class_path` at the fully qualified `module:Class` and passing the environment `name` as the first positional argument:

```bash
/isaac-sim/python.sh submodules/IsaacLab-Arena/isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action \
  --num_steps 50 \
  --external_environment_class_path arena_envs.environment:ExternalFrankaTableEnvironment \
  franka_push_coffee_machine_button
```

For the full external-integration reference, see the Arena docs under `submodules/IsaacLab-Arena/docs/pages/arena_in_your_repo/`.

## Boundaries

- **Never edit the Arena submodule** to add features — extend it from `arena_envs` through the registration API. If Arena itself needs a change, that belongs upstream, not here.
- **Never vendor `arena_so101/` back into this repo** — change [isaaclab-so101](https://github.com/art-e-fact/isaaclab-so101) and bump the SHA (see [DEVELOPMENT.md](DEVELOPMENT.md)).
- **Never commit models, datasets, or secrets.** Keep them on the host and mount them via `./docker/run_docker.sh -d <datasets> -m <models> -e <eval>`.
- **Ask first** before changing `docker/` or bumping `submodules/IsaacLab-Arena` — these affect every contributor.
