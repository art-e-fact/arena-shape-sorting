# AGENTS.md

This file provides guidance to AI coding agents (Claude Code, OpenAI Codex, etc.) when working with code in this repository.

## Project

`arena-shape-sorting` is a project built **on top of** IsaacLab-Arena, following the "Arena in your repository" pattern: Arena is vendored unmodified as a git submodule (`submodules/IsaacLab-Arena`) and extended purely through its registration API. Our own code lives in the `arena_envs` package, which defines custom environments and registers them with Arena.

We consume Arena and the SO-101 embodiment as dependencies — we do **not** develop them here. Treat everything under `submodules/IsaacLab-Arena` as read-only third-party code. SO-101 (`arena_so101`) comes from [art-e-fact/isaaclab-so101](https://github.com/art-e-fact/isaaclab-so101), pinned in `arena_envs/pyproject.toml`. Local editable overrides are documented in [DEVELOPMENT.md](DEVELOPMENT.md).

## Skill library

Multi-step workflows are captured as Agent Skills under `.agents/skills/`. When a task matches a skill, prefer invoking it over re-deriving the procedure from this file. Currently none.

## Environment

Everything runs natively in Arena's uv venv (`submodules/IsaacLab-Arena/.venv`): Isaac Sim, Isaac Lab, the Arena submodule, cuRobo (`submodules/curobo`), our `arena_envs` package (editable) and the pinned `arena-so101` package from GitHub. `source ./setup.sh` creates or updates the venv and activates it; `source ./setup.sh --force` re-syncs after a pin or submodule bump. Contributor workflows (SO-101 pin, local checkout) are in [DEVELOPMENT.md](DEVELOPMENT.md).

## Repository layout

- `arena_envs/` — our package: custom environments that subclass Arena base classes and register via `@register_environment` (`src/arena_envs/` / `src/shape_sorting/`)
- `submodules/IsaacLab-Arena/` — vendored Arena submodule (read-only), which itself vendors Isaac Lab under `submodules/IsaacLab`
- `DEVELOPMENT.md` — pin bumps, local `arena-so101` checkout

## Defining and running environments

Custom environments subclass an Arena base class (e.g. `ExampleEnvironmentBase`), set a unique `name`, and implement `get_env()` / `add_cli_args()`. They register themselves on import via `@register_environment`. See `arena_envs/src/arena_envs/environment.py` for the reference example.

Run an external environment through Arena's policy runner, pointing `--external_environment_class_path` at the fully qualified `module:Class` and passing the environment `name` as the first positional argument:

```bash
python submodules/IsaacLab-Arena/isaaclab_arena/evaluation/policy_runner.py \
  --policy_type zero_action \
  --num_steps 50 \
  --external_environment_class_path arena_envs.environment:ExternalFrankaTableEnvironment \
  franka_push_coffee_machine_button
```

For the full external-integration reference, see the Arena docs under `submodules/IsaacLab-Arena/docs/pages/arena_in_your_repo/`.

## Boundaries

- **Never edit the Arena submodule** to add features — extend it from `arena_envs` through the registration API. If Arena itself needs a change, that belongs upstream, not here.
- **Never vendor `arena_so101/` back into this repo** — change [isaaclab-so101](https://github.com/art-e-fact/isaaclab-so101) and bump the SHA (see [DEVELOPMENT.md](DEVELOPMENT.md)).
- **Never commit models, datasets, or secrets.** Keep them outside the tracked tree (`datasets/`, `outputs/` are gitignored).
- **Ask first** before bumping `submodules/IsaacLab-Arena` — it affects every contributor.
- **Don't run `shape_sorting.scenarios` or the `curobo_scenarios` Artefacts job to check your work.** They boot Isaac Sim for minutes per scenario and exist only to put videos on the Artefacts dashboard. The fast check is `python -m pytest arena_envs/tests`.
