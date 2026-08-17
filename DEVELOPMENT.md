# Development

Contributor notes for this repo. End-user setup and demo commands live in [README.md](README.md).

## SO-101 package (`arena-so101`)

The SO-101 embodiment, teleop devices, and LeRobot recorder live in
[art-e-fact/isaaclab-so101](https://github.com/art-e-fact/isaaclab-so101)
(import name `arena_so101`). This repo does **not** vendor that tree.

`arena_envs` pins a git SHA in [`arena_envs/pyproject.toml`](arena_envs/pyproject.toml):

```toml
"arena-so101[leader] @ git+https://github.com/art-e-fact/isaaclab-so101.git@<sha>"
```

Installing `arena_envs` (Docker image build and `source ./setup.sh`) pulls that pin.
The package lands in Isaac Sim / the Arena venv `site-packages`, not under this repo
root — bind-mounting the pollenating-demo tree does not overlay it.

### Bump the pin

1. Merge / push the change in isaaclab-so101.
2. Replace the SHA in `arena_envs/pyproject.toml`.
3. Rebuild the Docker image (`./docker/run_docker.sh -r`) or re-run `source ./setup.sh --force`.

### Local editable checkout

Use this when you are changing SO-101 code and want this repo to pick up edits immediately.

Host (uv) install:

```bash
export ARENA_SO101_PATH=/path/to/isaaclab-so101   # e.g. ../arena-so101
source ./setup.sh --force
```

Docker — pass `-S` (or export `ARENA_SO101_PATH` before `run_docker.sh`). That bind-mounts
the checkout at `/workspaces/arena-so101` and the entrypoint runs
`pip install -e "${ARENA_SO101_PATH}[leader]"` so it wins over the image's git install:

```bash
./docker/run_docker.sh -S /path/to/isaaclab-so101
```

Restart the container after changing `-S`; attaching to an already-running container
does not remount.

Confirm which copy is loaded:

```bash
/isaac-sim/python.sh -c "import arena_so101; print(arena_so101.__file__)"
```

Git install → a path under Isaac Sim `site-packages`. Local override →
`/workspaces/arena-so101/...` (Docker) or `$ARENA_SO101_PATH/...` (host).

Do not copy `arena_so101/` back into this repo; it will shadow the installed package.

## Docker

See [`.agents/skills/dev-container/SKILL.md`](.agents/skills/dev-container/SKILL.md) for
container discovery, `run_docker.sh` flags, and `docker exec` usage.

Ask before changing `docker/` or bumping `submodules/IsaacLab-Arena` — those affect
every contributor.
