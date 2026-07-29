"""Generate cuRobo URDF + robot YAML for the SO-101 workshop USD.

Converts ``embodiments/data/SO-ARM101-USD.usd`` to URDF (Isaac Lab), then runs
cuRobo ``RobotBuilder`` to fit collision spheres and write a planning config.
The YAML is patched for this repo's joint names, home pose, and locked Jaw.

Requires Isaac Sim (USD→URDF) and cuRobo v0.8+ (sphere fitting). CUDA is
needed for the build step.

Examples::

    # Full pipeline (headless Isaac Sim + cuRobo)
    python -m arena_so101.generate_curobo_config --headless

    # Rebuild YAML from an existing URDF (no Isaac Sim)
    python -m arena_so101.generate_curobo_config \\
      --skip-usd-convert \\
      --urdf /path/to/SO-ARM101-USD.urdf \\
      --asset-path /path/to/meshes

    # Inspect fitted spheres in Viser
    python -m arena_so101.generate_curobo_config --headless --visualize
"""

from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths & SO-101 constants (workshop USD / arena_so101.embodiments.so101)
# ---------------------------------------------------------------------------

_PKG_DIR = Path(__file__).resolve().parent
_DEFAULT_USD = _PKG_DIR / "embodiments" / "data" / "SO-ARM101-USD.usd"
_DEFAULT_OUTPUT_DIR = _PKG_DIR / "embodiments" / "data" / "curobo"

# Must match ArticulationCfg init_state in embodiments/so101.py.
_HOME_JOINT_POS: dict[str, float] = {
    "Rotation": -0.2736,
    "Pitch": -0.6109,
    "Elbow": -0.0745,
    "Wrist_Pitch": 1.5148,
    "Wrist_Roll": -1.6034,
    "Jaw": -0.1465,
}

_JAW_OPEN_RAD = math.radians(100.0)
_JAW_CLOSE_RAD = math.radians(-10.0)

# Matches ImplicitActuatorCfg.effort_limit_sim on the SO-101 embodiment.
_DEFAULT_JOINT_EFFORT = 30.0
# Sensible placeholder when the exporter left velocity at 0 / missing.
_DEFAULT_JOINT_VELOCITY = 10.0

# Preferred names from the workshop USD; aliases cover LeRobot-style URDFs.
_EE_LINK_CANDIDATES = ("gripper", "gripper_frame_link", "jaw", "Moving Jaw")
_JAW_JOINT_CANDIDATES = ("Jaw", "gripper", "jaw")
_BASE_LINK_CANDIDATES = ("base", "Base", "base_link", "root")


def _parse_urdf_names(urdf_path: Path) -> tuple[str, list[str], list[str]]:
    """Return (base_link, link_names, revolute_joint_names) from a URDF."""
    root = ET.parse(urdf_path).getroot()
    link_names = [e.get("name") for e in root.findall("link") if e.get("name")]
    joint_names: list[str] = []
    child_links: set[str] = set()
    for joint in root.findall("joint"):
        jtype = joint.get("type", "")
        name = joint.get("name")
        if name and jtype in {"revolute", "continuous", "prismatic"}:
            joint_names.append(name)
        child = joint.find("child")
        if child is not None and child.get("link"):
            child_links.add(child.get("link"))
    base = next((n for n in link_names if n not in child_links), link_names[0] if link_names else "base")
    return base, link_names, joint_names


def _pick_name(candidates: tuple[str, ...], available: list[str], *, kind: str) -> str:
    available_l = {n.lower(): n for n in available}
    for cand in candidates:
        if cand in available:
            return cand
        if cand.lower() in available_l:
            return available_l[cand.lower()]
    raise RuntimeError(
        f"Could not find {kind} among {list(candidates)}; URDF has: {available}"
    )


def sanitize_urdf_for_curobo(
    urdf_path: Path,
    *,
    effort: float = _DEFAULT_JOINT_EFFORT,
    velocity: float = _DEFAULT_JOINT_VELOCITY,
) -> int:
    """Fix joint limits that break cuRobo's ``JointLimits`` validation.

    Isaac Lab's USD→URDF path runs ``_sanitize_urdf_for_pinocchio``, which sets
    missing/infinite ``effort`` to ``0.``. cuRobo then builds effort bounds as
    ``[-0, +0]`` and raises ``lower effort limits must be less than upper``.
    """
    tree = ET.parse(urdf_path)
    changed = 0
    for limit in tree.iter("limit"):
        raw_effort = limit.get("effort")
        if raw_effort is None or float(raw_effort) <= 0.0:
            limit.set("effort", f"{effort:g}")
            changed += 1
        raw_velocity = limit.get("velocity")
        if raw_velocity is None or float(raw_velocity) <= 0.0:
            limit.set("velocity", f"{velocity:g}")
            changed += 1
    if changed:
        tree.write(urdf_path, encoding="unicode")
    return changed


def convert_usd_to_urdf(usd_path: Path, output_dir: Path) -> tuple[Path, Path]:
    """Export workshop USD to URDF + meshes under ``output_dir``."""
    from isaaclab.controllers.utils import convert_usd_to_urdf as _convert

    output_dir.mkdir(parents=True, exist_ok=True)
    urdf_path_str, mesh_dir_str = _convert(
        str(usd_path.resolve()),
        str(output_dir.resolve()),
        force_conversion=True,
    )
    urdf_path = Path(urdf_path_str)
    mesh_dir = Path(mesh_dir_str)
    if not urdf_path.is_file():
        raise FileNotFoundError(f"USD→URDF did not produce a URDF at {urdf_path}")
    n = sanitize_urdf_for_curobo(urdf_path)
    if n:
        print(f"  sanitized {n} joint limit field(s) for cuRobo (effort/velocity)")
    return urdf_path, mesh_dir


def build_curobo_yaml(
    urdf_path: Path,
    asset_path: Path,
    output_yml: Path,
    *,
    tool_frame: str,
    base_link: str,
    sphere_density: float = 1.0,
    num_collision_samples: int = 1000,
    compute_metrics: bool = True,
    visualize: bool = False,
    viz_port: int = 8080,
    seed: int = 42,
) -> Path:
    """Fit spheres / collision matrix and write a cuRobo robot YAML."""
    import numpy as np
    import torch
    from curobo.logging import setup_logger
    from curobo.robot_builder import RobotBuilder

    setup_logger("warning")
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"Building cuRobo model from URDF: {urdf_path}")
    print(f"  meshes: {asset_path}")
    print(f"  tool frame: {tool_frame}  base: {base_link}")

    builder = RobotBuilder(
        urdf_path=str(urdf_path.resolve()),
        asset_path=str(asset_path.resolve()),
        tool_frames=[tool_frame],
    )

    # Base is fixed to the world; the arm cannot collide with it, so skip spheres.
    if base_link in builder._mesh_link_names:
        builder._mesh_link_names = [n for n in builder._mesh_link_names if n != base_link]
        print(f"Skipping collision spheres for fixed base link: {base_link}")

    print("\nFitting collision spheres...")
    builder.fit_collision_spheres(
        sphere_density=sphere_density,
        compute_metrics=compute_metrics,
        protrusion_weight=2000.0,
    )
    print(f"Fitted {builder.num_spheres} spheres across {len(builder.collision_link_names)} links")

    if compute_metrics and builder.link_metrics:
        print(f"  {'link':<25s} {'n_sph':>5s} {'cover%':>7s} {'protr%':>7s}")
        print(f"  {'-' * 50}")
        for link_name, m in builder.link_metrics.items():
            print(
                f"  {link_name:<25s} {m.num_spheres:5d} "
                f"{m.coverage * 100:6.1f}% {m.protrusion * 100:6.1f}%"
            )

    print("\nComputing collision matrix...")
    builder.compute_collision_matrix(prune_collisions=True, num_samples=num_collision_samples)
    print(f"Created ignore matrix with {len(builder.collision_matrix)} entries")

    config = builder.build()
    output_yml.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp name first; patch wraps under robot_cfg and SO-101 fields.
    raw_yml = output_yml.with_suffix(".raw.yml")
    builder.save(config, str(raw_yml))

    _, _, revolute_joints = _parse_urdf_names(urdf_path)
    patched = patch_so101_robot_yaml(
        raw_yml,
        output_yml,
        urdf_path=urdf_path,
        asset_path=asset_path,
        tool_frame=tool_frame,
        jaw_joint=_pick_name(_JAW_JOINT_CANDIDATES, revolute_joints, kind="jaw joint"),
    )
    raw_yml.unlink(missing_ok=True)

    print(f"\nWrote {patched}")

    if visualize:
        print(f"Starting Viser at http://localhost:{viz_port} (Ctrl+C to stop)")
        builder.visualize(config, port=viz_port)
        try:
            import time

            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping visualization")

    return patched


def _ensure_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    """Return ``parent[key]`` as a dict, replacing missing/``None`` values."""
    value = parent.get(key)
    if not isinstance(value, dict):
        parent[key] = {}
    return parent[key]


def _ensure_list(parent: dict[str, Any], key: str) -> list[Any]:
    """Return ``parent[key]`` as a list, replacing missing/``None`` values."""
    value = parent.get(key)
    if not isinstance(value, list):
        parent[key] = []
    return parent[key]


def patch_so101_robot_yaml(
    raw_yml: Path,
    output_yml: Path,
    *,
    urdf_path: Path,
    asset_path: Path,
    tool_frame: str,
    jaw_joint: str,
) -> Path:
    """Wrap builder output under ``robot_cfg`` and apply SO-101 planning defaults."""
    import yaml

    with raw_yml.open() as f:
        data: dict[str, Any] = yaml.safe_load(f)

    # RobotBuilder.save writes ``kinematics:``; shipped configs use ``robot_cfg:``.
    if "robot_cfg" in data:
        kin = data["robot_cfg"]["kinematics"]
    elif "kinematics" in data:
        kin = data["kinematics"]
        data = {"robot_cfg": {"kinematics": kin}}
    else:
        raise RuntimeError(f"Unexpected cuRobo YAML structure in {raw_yml}: keys={list(data)}")

    kin = data["robot_cfg"]["kinematics"]
    kin["urdf_path"] = str(urdf_path.resolve())
    kin["asset_root_path"] = str(asset_path.resolve())
    kin["tool_frames"] = [tool_frame]
    kin["lock_joints"] = {jaw_joint: _JAW_OPEN_RAD}

    # Grasp attach frame (same pattern as franka.yml) — parent is the EE link.
    # Builder may serialize these fields as explicit nulls; setdefault won't replace None.
    extra_spheres = _ensure_mapping(kin, "extra_collision_spheres")
    extra_spheres["attached_object"] = 4
    extra_links = _ensure_mapping(kin, "extra_links")
    extra_links["attached_object"] = {
        "fixed_transform": [0, 0, 0, 1, 0, 0, 0],
        "joint_name": "attach_joint",
        "joint_type": "FIXED",
        "link_name": "attached_object",
        "parent_link_name": tool_frame,
    }
    collision_links = _ensure_list(kin, "collision_link_names")
    if "attached_object" not in collision_links:
        collision_links.append("attached_object")
    self_buf = _ensure_mapping(kin, "self_collision_buffer")
    self_buf.setdefault("attached_object", 0.0)
    self_ignore = _ensure_mapping(kin, "self_collision_ignore")
    ee_ignore = self_ignore.get(tool_frame)
    if not isinstance(ee_ignore, list):
        ee_ignore = []
        self_ignore[tool_frame] = ee_ignore
    if "attached_object" not in ee_ignore:
        ee_ignore.append("attached_object")

    cspace = kin.get("cspace")
    if not isinstance(cspace, dict):
        cspace = {}
        kin["cspace"] = cspace
    joint_names: list[str] = list(cspace.get("joint_names") or [])
    if joint_names:
        defaults = []
        for name in joint_names:
            if name in _HOME_JOINT_POS:
                defaults.append(float(_HOME_JOINT_POS[name]))
            elif name == jaw_joint:
                defaults.append(float(_JAW_OPEN_RAD))
            else:
                # Keep builder mid-range if we don't know this joint.
                existing = cspace.get("default_joint_position")
                idx = len(defaults)
                if isinstance(existing, list) and idx < len(existing):
                    defaults.append(float(existing[idx]))
                else:
                    defaults.append(0.0)
        cspace["default_joint_position"] = defaults

    # Sibling metadata for scripted policies / Arena CuroboEmbodimentCfg (not consumed by cuRobo).
    data["arena_so101"] = {
        "ee_link_name": tool_frame,
        "gripper_joint_names": [jaw_joint],
        "gripper_open_joint_pos": {jaw_joint: _JAW_OPEN_RAD},
        "gripper_closed_joint_pos": {jaw_joint: _JAW_CLOSE_RAD},
        "hand_link_names": [tool_frame],
        "home_joint_pos": dict(_HOME_JOINT_POS),
    }

    output_yml.parent.mkdir(parents=True, exist_ok=True)
    with output_yml.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
    return output_yml


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate SO-101 URDF + cuRobo YAML for motion planning.",
    )
    parser.add_argument(
        "--usd",
        type=Path,
        default=_DEFAULT_USD,
        help=f"Workshop USD path (default: {_DEFAULT_USD})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help=f"Directory for URDF, meshes, and YAML (default: {_DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--yaml-name",
        type=str,
        default="so101.yml",
        help="Output robot YAML filename (default: so101.yml)",
    )
    parser.add_argument(
        "--skip-usd-convert",
        action="store_true",
        help="Do not run USD→URDF; use --urdf / --asset-path (or files already in --output-dir).",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=None,
        help="Existing URDF (implies --skip-usd-convert when set).",
    )
    parser.add_argument(
        "--asset-path",
        type=Path,
        default=None,
        help="Mesh directory for the URDF (required with --urdf unless --output-dir already has meshes/).",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Only convert USD→URDF; skip cuRobo sphere fitting.",
    )
    parser.add_argument(
        "--sphere-density",
        type=float,
        default=2.0,
        help="cuRobo sphere-density multiplier (default: 1.0).",
    )
    parser.add_argument(
        "--num-collision-samples",
        type=int,
        default=1000,
        help="Samples for self-collision ignore pruning (default: 1000).",
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Skip per-link sphere-fit quality metrics.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Open Viser after building to inspect collision spheres.",
    )
    parser.add_argument(
        "--viz-port",
        type=int,
        default=8080,
        help="Viser port (default: 8080).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for sphere fitting / collision sampling.",
    )
    parser.add_argument(
        "--tool-frame",
        type=str,
        default=None,
        help="Override EE / tool link name (default: auto-detect gripper).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    # AppLauncher args (--headless, --device, …) when USD conversion is needed.
    try:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
        has_app_launcher = True
    except ImportError:
        has_app_launcher = False

    args = parser.parse_args(argv)

    if args.urdf is not None:
        args.skip_usd_convert = True

    need_sim = not args.skip_usd_convert
    simulation_app = None
    if need_sim:
        if not has_app_launcher:
            print(
                "Isaac Lab AppLauncher is required for USD→URDF. "
                "Run inside the Isaac Sim / Arena environment, or pass --urdf.",
                file=sys.stderr,
            )
            return 1
        # Default to headless for this asset-generation script.
        if not getattr(args, "headless", False) and "--headless" not in (argv or sys.argv[1:]):
            args.headless = True
        app_launcher = AppLauncher(vars(args))
        simulation_app = app_launcher.app

    try:
        output_dir: Path = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        if need_sim:
            if not args.usd.is_file():
                print(f"USD not found: {args.usd}", file=sys.stderr)
                return 1
            print(f"Converting USD → URDF\n  usd: {args.usd}\n  out: {output_dir}")
            urdf_path, mesh_dir = convert_usd_to_urdf(args.usd, output_dir)
            print(f"  urdf: {urdf_path}\n  meshes: {mesh_dir}")
        else:
            urdf_path = args.urdf
            if urdf_path is None:
                # Prefer previously generated file under output-dir.
                candidate = output_dir / "urdf" / f"{_DEFAULT_USD.stem}.urdf"
                if candidate.is_file():
                    urdf_path = candidate
                else:
                    print(
                        "Pass --urdf or generate once without --skip-usd-convert.",
                        file=sys.stderr,
                    )
                    return 1
            urdf_path = urdf_path.resolve()
            mesh_dir = args.asset_path
            if mesh_dir is None:
                mesh_dir = output_dir / "meshes"
            mesh_dir = mesh_dir.resolve()
            if not urdf_path.is_file():
                print(f"URDF not found: {urdf_path}", file=sys.stderr)
                return 1
            if not mesh_dir.is_dir():
                print(f"Mesh directory not found: {mesh_dir}", file=sys.stderr)
                return 1
            n = sanitize_urdf_for_curobo(urdf_path)
            if n:
                print(f"Sanitized {n} joint limit field(s) for cuRobo (effort/velocity)")

        base_link, link_names, joint_names = _parse_urdf_names(urdf_path)
        # Prefer workshop base name when present.
        try:
            base_link = _pick_name(_BASE_LINK_CANDIDATES, link_names, kind="base link")
        except RuntimeError:
            print(f"Using URDF root link as base: {base_link}")

        tool_frame = args.tool_frame or _pick_name(
            _EE_LINK_CANDIDATES, link_names, kind="end-effector link"
        )
        print(f"URDF joints: {joint_names}")
        print(f"Resolved base={base_link}  tool_frame={tool_frame}")

        if args.skip_build:
            print("Skipping cuRobo build (--skip-build).")
            return 0

        output_yml = output_dir / args.yaml_name
        build_curobo_yaml(
            urdf_path,
            mesh_dir,
            output_yml,
            tool_frame=tool_frame,
            base_link=base_link,
            sphere_density=args.sphere_density,
            num_collision_samples=args.num_collision_samples,
            compute_metrics=not args.no_metrics,
            visualize=args.visualize,
            viz_port=args.viz_port,
            seed=args.seed,
        )
        print("\nDone. Point MotionPlanner / CuroboEmbodimentCfg at:")
        print(f"  robot YAML: {output_yml}")
        print(f"  URDF:       {urdf_path}")
        return 0
    except Exception as exc:
        # Isaac Sim's process teardown can obscure tracebacks; print ours first.
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
    finally:
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
