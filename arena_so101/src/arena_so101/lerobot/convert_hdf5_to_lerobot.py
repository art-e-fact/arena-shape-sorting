"""Convert MimicGen HDF5 → GR00T LeRobot, with a torchvision video polyfill.

Arena's ``convert_hdf5_to_lerobot`` calls ``torchvision.io.write_video``, which
was removed in torchvision ≥ 0.24. This wrapper restores a compatible
``write_video`` (via imageio + ffmpeg) before running the Arena converter.

It also supports a multi-camera ``cameras`` map in the YAML (sim ``camera_obs``
key → LeRobot ``observation.images.*`` key). Arena's converter only writes the
primary ``pov_cam_name_sim`` / ``video_name_lerobot`` pair; extra streams are
written afterward and merged into ``meta/info.json``.
"""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
from typing import Any


def _install_write_video_polyfill() -> None:
    """Provide ``torchvision.io.write_video`` when the installed torchvision lacks it."""
    import numpy as np
    import torch
    import torchvision

    if hasattr(torchvision.io, "write_video"):
        return

    try:
        import imageio.v2 as imageio
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "torchvision.io.write_video is missing and imageio is not installed. "
            "Install imageio (and ffmpeg) or use torchvision < 0.24."
        ) from exc

    def write_video(
        filename,
        video_array,
        fps: float,
        video_codec: str = "h264",
        options: dict | None = None,
        **_kwargs,
    ) -> None:
        if isinstance(video_array, torch.Tensor):
            frames = video_array.detach().cpu().numpy()
        else:
            frames = np.asarray(video_array)
        if frames.ndim != 4 or frames.shape[-1] not in (1, 3, 4):
            raise ValueError(f"Expected video array shaped (T, H, W, C), got {frames.shape}")
        if frames.dtype != np.uint8:
            frames = np.clip(frames, 0, 255).astype(np.uint8)

        codec = "libx264" if video_codec in ("h264", "libx264") else video_codec
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer_kwargs: dict = {
            "fps": fps,
            "codec": codec,
            "macro_block_size": 1,
            "ffmpeg_params": ["-pix_fmt", "yuv420p"],
        }
        if options:
            # torchvision passed ffmpeg options as a dict; forward common keys.
            for key, value in options.items():
                writer_kwargs.setdefault("ffmpeg_params", []).extend([f"-{key}", str(value)])

        with imageio.get_writer(str(path), **writer_kwargs) as writer:
            for frame in frames:
                writer.append_data(frame)

    torchvision.io.write_video = write_video  # type: ignore[attr-defined]
    print(
        f"Polyfilled torchvision.io.write_video "
        f"(torchvision {torchvision.__version__} has no built-in writer)."
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}, got {type(data).__name__}")
    return data


def _camera_pairs(raw: dict[str, Any]) -> list[tuple[str, str]]:
    """Return ordered (sim_camera_obs_key, lerobot_video_key) pairs."""
    cameras = raw.get("cameras")
    if isinstance(cameras, dict) and cameras:
        return [(str(sim), str(lerobot)) for sim, lerobot in cameras.items()]

    pov = raw.get("pov_cam_name_sim")
    video = raw.get("video_name_lerobot")
    if pov is None or video is None:
        raise ValueError("YAML must define ``cameras`` or both ``pov_cam_name_sim`` and ``video_name_lerobot``.")
    if isinstance(pov, list) or isinstance(video, list):
        pov_list = list(pov) if isinstance(pov, list) else [pov]
        video_list = list(video) if isinstance(video, list) else [video]
        if len(pov_list) != len(video_list):
            raise ValueError(
                f"pov_cam_name_sim ({len(pov_list)}) and video_name_lerobot ({len(video_list)}) length mismatch"
            )
        return [(str(s), str(v)) for s, v in zip(pov_list, video_list, strict=True)]
    return [(str(pov), str(video))]


def _write_extra_camera_videos(
    config,
    camera_pairs: list[tuple[str, str]],
) -> None:
    """Write non-primary camera streams and merge their metadata into info.json."""
    import h5py
    import numpy as np
    import torchvision
    from isaaclab_arena_gr00t.lerobot.convert_hdf5_to_lerobot import (
        get_video_metadata,
        resize_frames_with_padding,
    )
    from isaaclab_arena_gr00t.utils.io_utils import dump_json, load_json
    from tqdm import tqdm

    primary_sim, primary_lerobot = camera_pairs[0]
    extras = [(s, v) for s, v in camera_pairs[1:] if not (s == primary_sim and v == primary_lerobot)]
    if not extras:
        return

    print(f"Writing {len(extras)} extra camera stream(s): {[v for _, v in extras]}")

    hdf5_handler = h5py.File(config.hdf5_file_path, "r")
    hdf5_data = hdf5_handler["data"]
    trajectory_ids = list(hdf5_data.keys())

    # Representative paths for info.json features (first episode of each stream).
    video_paths: dict[str, Path] = {}

    for episode_index, trajectory_id in enumerate(tqdm(trajectory_ids, desc="extra cameras")):
        trajectory = hdf5_data[trajectory_id]
        episode_chunk = episode_index // config.chunks_size
        camera_obs = trajectory["camera_obs"]

        for sim_key, lerobot_key in extras:
            assert sim_key in camera_obs, f"{sim_key} missing in {trajectory_id}/camera_obs"
            frames = np.array(camera_obs[sim_key])[:-1]  # Lab reports one extra frame
            video_relpath = config.video_path.format(
                episode_chunk=episode_chunk, video_key=lerobot_key, episode_index=episode_index
            )
            video_path = config.lerobot_data_dir / video_relpath
            video_path.parent.mkdir(parents=True, exist_ok=True)

            assert frames.shape[1:] == tuple(
                config.original_image_size
            ), f"{sim_key}: frames.shape[1:]={frames.shape[1:]} != {config.original_image_size}"
            if config.target_image_size != config.original_image_size:
                frames = resize_frames_with_padding(
                    frames, target_image_size=config.target_image_size, bgr_conversion=False, pad_img=True
                )
            torchvision.io.write_video(str(video_path), frames, config.fps, video_codec="h264")
            if lerobot_key not in video_paths:
                video_paths[lerobot_key] = video_path

    hdf5_handler.close()

    info_path = config.lerobot_data_dir / "meta" / "info.json"
    info = load_json(info_path)
    features = info.setdefault("features", {})
    for lerobot_key, video_path in video_paths.items():
        features[lerobot_key] = get_video_metadata(video_path)
    dump_json(info, info_path, indent=4)
    print(f"Updated {info_path} with extra camera features: {list(video_paths)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert MimicGen HDF5 to GR00T LeRobot (SO-101-friendly wrapper)."
    )
    parser.add_argument("--yaml_file", required=True, help="Path to Gr00tDatasetConfig YAML")
    args = parser.parse_args(argv)

    _install_write_video_polyfill()

    from isaaclab_arena_gr00t.lerobot.config.dataset_config import Gr00tDatasetConfig
    from isaaclab_arena_gr00t.lerobot.convert_hdf5_to_lerobot import convert_hdf5_to_lerobot
    from isaaclab_arena_gr00t.utils.io_utils import create_config_from_yaml

    yaml_path = Path(args.yaml_file)
    raw = _load_yaml(yaml_path)
    camera_pairs = _camera_pairs(raw)

    # Arena ignores unknown YAML keys (e.g. ``cameras``); primary pair fields remain.
    primary_sim, primary_lerobot = camera_pairs[0]
    if raw.get("pov_cam_name_sim") not in (None, primary_sim) or raw.get("video_name_lerobot") not in (
        None,
        primary_lerobot,
    ):
        print(
            f"Note: using primary camera pair from cameras map: "
            f"{primary_sim} → {primary_lerobot}"
        )

    config = create_config_from_yaml(yaml_path, Gr00tDatasetConfig)
    # Ensure primary pair matches the first cameras entry even if YAML only had ``cameras``.
    config.pov_cam_name_sim = primary_sim
    config.video_name_lerobot = primary_lerobot
    config.lerobot_keys["video"] = primary_lerobot

    print("\n" + "=" * 50)
    print("GR00T LEROBOT DATASET CONFIGURATION:")
    print("=" * 50)
    for field in fields(Gr00tDatasetConfig):
        if field.init:
            print(f"  {field.name}: {getattr(config, field.name)}")
    print(f"  cameras (wrapper): {camera_pairs}")
    print("=" * 50 + "\n")

    convert_hdf5_to_lerobot(config)
    _write_extra_camera_videos(config, camera_pairs)


if __name__ == "__main__":
    main()
