"""Generate ARX X5 PCB assembly episodes from the supplied simulation bundle."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import BadZipFile, ZipFile


BUNDLE = "arx_x5_pcb_assembly"
EPISODES = ("episode_01", "episode_02")
CAMERAS = ("overhead", "left_wrist", "right_wrist")
REVIEW_VIEWS = ("table_front", "mounted_top_camera", "left_wrist", "right_wrist")


def source_directory(source, scratch):
    if source.is_dir():
        return source

    with ZipFile(source) as archive:
        extracted = set()
        for member in archive.infolist():
            if member.is_dir():
                continue
            parts = member.filename.split("/")
            if parts[0] != BUNDLE or len(parts) < 2:
                continue
            if parts[1] not in {"code", "scene"} and parts[1:] != ["replay.py"]:
                continue
            if any(part in {"", ".", ".."} or "\\" in part or ":" in part for part in parts):
                raise ValueError(f"Invalid bundle path: {member.filename}")
            key = member.filename.casefold()
            if key in extracted:
                raise ValueError(f"Duplicate bundle path: {member.filename}")
            extracted.add(key)
            archive.extract(member, scratch)
    return scratch / BUNDLE


def video_info(path, fps, frames):
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate,nb_read_frames,width,height",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    streams = json.loads(probe.stdout)["streams"]
    if (len(streams) != 1 or Fraction(streams[0]["avg_frame_rate"]) != fps
            or streams[0].get("nb_read_frames") != str(frames)):
        raise ValueError(f"Expected {frames} frames at {fps} fps in {path.name}")
    return {"file": path.name, "decoded_frames": frames, "fps": fps,
            "width": streams[0]["width"], "height": streams[0]["height"],
            "duration_s": frames / fps}


def index_outputs(dataset, states_only):
    manifest = dataset / "dataset.json"
    data = json.loads(manifest.read_text())
    geometry = json.loads((dataset / "geometry_validation.json").read_text())
    data["fixture_pickup"] = json.loads((dataset / "scene" / "fixture_bores.json").read_text())
    data["source_workcell"] = "scene/scene_pcb_arx_x5_fixed_dual.xml"
    fps = data["fps"]
    if fps != 15 or tuple(episode["episode"] for episode in data["episodes"]) != EPISODES:
        raise ValueError("Expected two ARX X5 trajectories at 15 Hz")
    for episode in data["episodes"]:
        name = episode["episode"]
        folder = dataset / name
        metadata_path = folder / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        frames = episode["frames"]
        if (not (folder / "trajectory.npz").is_file() or
                (metadata["fps"], metadata["frames"]) != (fps, frames)):
            raise ValueError(f"Inconsistent trajectory files for {name}")
        episode["geometry_validation"] = geometry[name]
        if states_only:
            continue
        cameras = (*CAMERAS, "mounted_top_camera") if name == EPISODES[0] else CAMERAS
        episode["videos"] = {camera: video_info(folder / f"{camera}.mp4", fps, frames)
                             for camera in cameras}
        metadata["cameras"] = {camera: info["file"] for camera, info in episode["videos"].items()}
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    if not states_only:
        review_frames = 2 * data["episodes"][0]["frames"]
        front = video_info(dataset / "episode_01_table_front.mp4", 2 * fps, review_frames)
        data["front_review"] = {**front, "episode": EPISODES[0], "camera": "assembly_front",
                                "data_row_frame_stride": 2}
        combined = video_info(dataset / "episode_01_synchronized_mounted_camera_views.mp4",
                              2 * fps, review_frames)
        data["synchronized_review"] = {**combined, "episode": EPISODES[0], "views": REVIEW_VIEWS,
                                       "data_row_frame_stride": 2}
    manifest.write_text(json.dumps(data, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="ARX X5 ZIP or its extracted directory")
    parser.add_argument("--out", required=True, type=Path, help="new dataset directory")
    parser.add_argument("--states-only", action="store_true", help="skip camera rendering")
    args = parser.parse_args()
    output = args.out.absolute()
    if output.exists() or output.is_symlink():
        parser.error(f"Output already exists: {output}")
    if not args.states_only and any(shutil.which(tool) is None for tool in ("ffmpeg", "ffprobe")):
        parser.error("Video export requires ffmpeg and ffprobe in PATH")

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".arx-x5-", dir=output.parent) as work:
            work = Path(work)
            source = source_directory(args.source.resolve(), work)
            rebuild = source / "code" / "rebuild.py"
            base_scene = source / "scene" / "scene_pcb_arx_x5_fixed_dual.xml"
            if not rebuild.is_file() or not base_scene.is_file():
                raise ValueError("Source must contain code/rebuild.py and the ARX X5 workcell")
            dataset = work / "dataset"
            command = [sys.executable, str(rebuild), str(dataset)]
            if args.states_only:
                command.append("--states-only")
            subprocess.run(command, cwd=source, check=True)
            index_outputs(dataset, args.states_only)
            replay = source / "replay.py"
            if replay.is_file():
                shutil.copy2(replay, dataset / replay.name)
            if output.exists() or output.is_symlink():
                raise FileExistsError(f"Output already exists: {output}")
            dataset.rename(output)
    except (OSError, ValueError, BadZipFile, subprocess.CalledProcessError) as exc:
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            print(exc.stderr.strip(), file=sys.stderr)
        parser.exit(1, f"PCB generation failed: {exc}\n")
    print(output)


if __name__ == "__main__":
    main()
