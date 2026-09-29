# Development

Contributor notes for this repo. End-user setup and demo commands live in [README.md](README.md).

## SO-101 package (`arena-so101`)

The SO-101 embodiment, teleop devices, and LeRobot recorder live in
[art-e-fact/isaaclab-so101](https://github.com/art-e-fact/isaaclab-so101)
(import name `arena_so101`). This repo does **not** vendor that tree.

`arena_envs` pins a git SHA in [`arena_envs/pyproject.toml`](arena_envs/pyproject.toml):

```toml
"arena-so101[leader,lerobot] @ git+https://github.com/art-e-fact/isaaclab-so101.git@<sha>"
```

Installing `arena_envs` (`source ./setup.sh`) pulls that pin. The package lands in the
Arena venv's `site-packages`, not under this repo root.

### Bump the pin

1. Merge / push the change in isaaclab-so101.
2. Replace the SHA in `arena_envs/pyproject.toml`.
3. Re-run `source ./setup.sh --force`.

### Local editable checkout

Use this when you are changing SO-101 code and want this repo to pick up edits immediately.

```bash
export ARENA_SO101_PATH=/path/to/isaaclab-so101   # e.g. ../arena-so101
source ./setup.sh --force
```

Confirm which copy is loaded:

```bash
python -c "import arena_so101; print(arena_so101.__file__)"
```

Git install → a path under the venv's `site-packages`. Local override → `$ARENA_SO101_PATH/...`.

Do not copy `arena_so101/` back into this repo; it will shadow the installed package.

## Arena submodule

Ask before bumping `submodules/IsaacLab-Arena` — it affects every contributor. After a bump:
`git submodule update --init --recursive` and `source ./setup.sh --force`.
