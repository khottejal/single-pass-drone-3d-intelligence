"""Prepare single-pass drone footage and optionally run a COLMAP reconstruction."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import sqlite3
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "OpenCV and NumPy are required. Install dependencies with: "
        "python -m pip install -r requirements.txt"
    ) from exc


VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}
TIME_ALIASES = {
    "timestamp", "time", "datetime", "utctime", "gpstime", "gpstimestamp",
    "videotimes", "videotime",
}
LAT_ALIASES = {"lat", "latitude", "latitudedegrees", "dronelat"}
LON_ALIASES = {"lon", "long", "longitude", "longitudedegrees", "dronelon"}
ALT_ALIASES = {
    "alt", "altitude", "altitudem", "gpsaltitude", "height", "dronealtitudem",
}
OPTIONAL_ALIASES = {
    "roll": {"roll", "rollangle"},
    "pitch": {"pitch", "pitchangle"},
    "yaw": {"yaw", "heading", "course"},
}


class PipelineError(Exception):
    """An actionable input or processing error."""


@dataclass
class Telemetry:
    seconds: float
    timestamp: str
    latitude: float
    longitude: float
    altitude: float
    roll: float | None = None
    pitch: float | None = None
    yaw: float | None = None


def normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def parse_time_value(value: str) -> tuple[float, str]:
    value = value.strip()
    if not value:
        raise PipelineError("GPS CSV contains an empty timestamp.")
    try:
        number = float(value)
        # Epoch milliseconds are common in telemetry exports.
        if abs(number) >= 100_000_000_000:
            number /= 1000.0
        return number, value
    except ValueError:
        pass

    iso_value = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
    try:
        parsed = dt.datetime.fromisoformat(iso_value)
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(dt.timezone.utc).replace(tzinfo=None)
        return parsed.timestamp(), value
    except ValueError:
        pass

    for fmt in ("%H:%M:%S.%f", "%H:%M:%S", "%M:%S.%f", "%M:%S"):
        try:
            parsed_time = dt.datetime.strptime(value, fmt)
            return (
                parsed_time.hour * 3600
                + parsed_time.minute * 60
                + parsed_time.second
                + parsed_time.microsecond / 1_000_000,
                value,
            )
        except ValueError:
            continue
    raise PipelineError(f"Unsupported GPS timestamp value: {value!r}")


def _find_column(headers: list[str], aliases: set[str], required: bool = True) -> str | None:
    for header in headers:
        if normalize_header(header) in aliases:
            return header
    if required:
        raise PipelineError(
            f"GPS CSV is missing a required column. Accepted names: {', '.join(sorted(aliases))}"
        )
    return None


def load_telemetry(path: Path) -> list[Telemetry]:
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise PipelineError(f"GPS CSV has no header row: {path}")
        headers = list(reader.fieldnames)
        time_col = _find_column(headers, TIME_ALIASES)
        lat_col = _find_column(headers, LAT_ALIASES)
        lon_col = _find_column(headers, LON_ALIASES)
        alt_col = _find_column(headers, ALT_ALIASES, required=False)
        optional_cols = {
            name: _find_column(headers, aliases, required=False)
            for name, aliases in OPTIONAL_ALIASES.items()
        }
        video_time_col = _find_column(
            headers, {"videotimes", "videotime"}, required=False
        )
        records: list[tuple[float, str, float, float, float, float | None, float | None, float | None]] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                timestamp_value = (row.get(time_col) or "").strip()
                if not timestamp_value and video_time_col:
                    timestamp_value = (row.get(video_time_col) or "").strip()
                timestamp_seconds, timestamp = parse_time_value(timestamp_value)
                latitude = float(row[lat_col] or "")
                longitude = float(row[lon_col] or "")
                altitude = float(row[alt_col] or "0") if alt_col else 0.0
                angles = [
                    float(row[column]) if column and row.get(column, "").strip() else None
                    for column in optional_cols.values()
                ]
            except (TypeError, ValueError, KeyError, PipelineError) as exc:
                raise PipelineError(f"Invalid GPS CSV row {row_number}: {exc}") from exc
            if not (math.isfinite(latitude) and -90 <= latitude <= 90):
                raise PipelineError(f"Invalid latitude on GPS CSV row {row_number}.")
            if not (math.isfinite(longitude) and -180 <= longitude <= 180):
                raise PipelineError(f"Invalid longitude on GPS CSV row {row_number}.")
            if not math.isfinite(altitude) or not math.isfinite(timestamp_seconds):
                raise PipelineError(f"Non-finite GPS value on CSV row {row_number}.")
            records.append((timestamp_seconds, timestamp, latitude, longitude, altitude, *angles))

    if len(records) < 2:
        raise PipelineError("GPS CSV must contain at least two valid telemetry samples.")
    records.sort(key=lambda item: item[0])
    origin = records[0][0]
    telemetry = [
        Telemetry(seconds=item[0] - origin, timestamp=item[1], latitude=item[2],
                  longitude=item[3], altitude=item[4], roll=item[5], pitch=item[6], yaw=item[7])
        for item in records
    ]
    if any(b.seconds <= a.seconds for a, b in zip(telemetry, telemetry[1:])):
        raise PipelineError("GPS timestamps must be unique and strictly increasing.")
    return telemetry


def interpolate_telemetry(samples: list[Telemetry], seconds: float) -> Telemetry | None:
    if seconds < samples[0].seconds or seconds > samples[-1].seconds:
        return None
    for left, right in zip(samples, samples[1:]):
        if left.seconds <= seconds <= right.seconds:
            fraction = (seconds - left.seconds) / (right.seconds - left.seconds)

            def linear(a: float | None, b: float | None) -> float | None:
                if a is None or b is None:
                    return a if fraction < 0.5 else b
                return a + (b - a) * fraction

            def angle(a: float | None, b: float | None) -> float | None:
                if a is None or b is None:
                    return a if fraction < 0.5 else b
                delta = (b - a + 180) % 360 - 180
                return (a + fraction * delta + 180) % 360 - 180

            return Telemetry(
                seconds=seconds,
                timestamp=left.timestamp,
                latitude=linear(left.latitude, right.latitude),
                longitude=linear(left.longitude, right.longitude),
                altitude=linear(left.altitude, right.altitude),
                roll=angle(left.roll, right.roll),
                pitch=angle(left.pitch, right.pitch),
                yaw=angle(left.yaw, right.yaw),
            )
    return samples[-1] if math.isclose(seconds, samples[-1].seconds) else None


def parse_source_urls(path: Path) -> dict[str, str]:
    sources: dict[str, str] = {}
    if not path.exists():
        return sources
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r"\s*(Video|GPS|Calibration)\s*:\s*(https?://\S+)", line, re.I)
        if match:
            sources[match.group(1).lower()] = match.group(2)
    return sources


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "drone3d-pipeline/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output)
    except Exception as exc:
        destination.unlink(missing_ok=True)
        raise PipelineError(f"Could not download {url}: {exc}") from exc


def _download_name(url: str, fallback: str) -> str:
    name = Path(urllib.parse.unquote(urllib.parse.urlparse(url).path)).name
    return name or fallback


def resolve_inputs(input_dir: Path, allow_download: bool) -> dict[str, Path | None]:
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise PipelineError(f"Input folder does not exist: {input_dir}")
    files = [p for p in input_dir.iterdir() if p.is_file()]
    video = next((p for p in files if p.suffix.lower() in VIDEO_SUFFIXES), None)
    gps = next((p for p in files if p.suffix.lower() == ".csv" and "gps" in p.stem.lower()), None)
    if gps is None:
        csv_files = [p for p in files if p.suffix.lower() == ".csv"]
        gps = csv_files[0] if len(csv_files) == 1 else None
    metadata = next(
        (
            p for p in files
            if p.suffix.lower() == ".json"
            and "flightmetadata" in normalize_header(p.stem)
            and "template" not in p.stem.lower()
        ),
        None,
    )
    if metadata is None:
        metadata = next(
            (p for p in files if p.suffix.lower() == ".json" and "template" not in p.stem.lower()),
            None,
        )
    calibration = next(
        (
            p for p in files
            if (("intrinsic" in p.stem.lower()) or p.suffix.lower() in {".yaml", ".yml"})
            and "template" not in p.stem.lower()
        ),
        None,
    )
    sources = parse_source_urls(input_dir / "SOURCE_URLS.txt")
    if allow_download:
        for kind, existing, fallback in (
            ("video", video, "drone_video.mp4"),
            ("gps", gps, "gps.csv"),
            ("calibration", calibration, "camera_calibration.yaml"),
        ):
            if existing is None and kind in sources:
                destination = input_dir / _download_name(sources[kind], fallback)
                download(sources[kind], destination)
                if kind == "video":
                    video = destination
                elif kind == "gps":
                    gps = destination
                else:
                    calibration = destination
        files = [p for p in input_dir.iterdir() if p.is_file()]
        if gps is None:
            gps = next((p for p in files if p.suffix.lower() == ".csv"), None)
    if video is None:
        raise PipelineError("No video found. Add a video file or rerun with --download-missing.")
    if gps is None:
        raise PipelineError("No GPS CSV found. Add a GPS CSV or rerun with --download-missing.")
    return {"video": video, "gps": gps, "metadata": metadata, "calibration": calibration}


def extract_frames(
    video_path: Path,
    telemetry: list[Telemetry],
    output_dir: Path,
    interval: float,
    min_blur_score: float,
    duplicate_threshold: float,
    timestamp_offset: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise PipelineError(f"OpenCV could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
        capture.release()
        raise PipelineError("Video does not expose valid FPS and frame-count metadata.")

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    counts = {"sampled": 0, "blurred": 0, "duplicate": 0, "unsynchronized": 0, "saved": 0}
    previous_small: np.ndarray | None = None
    frame_index = 0
    stride = max(1, round(fps * interval))
    started = time.monotonic()
    last_progress = started

    def save_checkpoint() -> None:
        if not rows:
            return
        with (output_dir / "frame_telemetry.csv").open(
            "w", newline="", encoding="utf-8"
        ) as target:
            writer = csv.DictWriter(target, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    try:
        while frame_index < frame_count:
            selected = frame_index % stride == 0
            ok = capture.grab()
            if not ok:
                break
            if selected:
                ok, frame = capture.retrieve()
                if not ok:
                    raise PipelineError(f"Could not decode video frame {frame_index}.")
                counts["sampled"] += 1
                elapsed = frame_index / fps
                gps = interpolate_telemetry(telemetry, elapsed + timestamp_offset)
                if gps is None:
                    counts["unsynchronized"] += 1
                else:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
                    if blur_score < min_blur_score:
                        counts["blurred"] += 1
                    elif previous_small is not None and float(
                        cv2.absdiff(small, previous_small).mean()
                    ) < duplicate_threshold:
                        counts["duplicate"] += 1
                    else:
                        filename = f"frame_{frame_index:08d}_{elapsed:010.3f}.jpg"
                        destination = frames_dir / filename
                        if not cv2.imwrite(str(destination), frame):
                            raise PipelineError(f"Could not write extracted frame: {destination}")
                        rows.append({
                            "frame": filename,
                            "frame_index": frame_index,
                            "video_seconds": round(elapsed, 6),
                            "gps_seconds_from_start": round(gps.seconds, 6),
                            "gps_timestamp": gps.timestamp,
                            "latitude": gps.latitude,
                            "longitude": gps.longitude,
                            "altitude": gps.altitude,
                            "roll": gps.roll,
                            "pitch": gps.pitch,
                            "yaw": gps.yaw,
                            "blur_score": round(blur_score, 3),
                        })
                        counts["saved"] += 1
                        previous_small = small

                now = time.monotonic()
                if now - last_progress >= 10:
                    percent = min(100, frame_index * 100 / frame_count)
                    print(
                        f"Frame extraction: {percent:.1f}% "
                        f"({frame_index:,}/{frame_count:,} frames scanned; "
                        f"{counts['saved']:,} retained)",
                        flush=True,
                    )
                    last_progress = now
                if counts["saved"] and counts["saved"] % 50 == 0:
                    save_checkpoint()
            frame_index += 1
    except KeyboardInterrupt:
        save_checkpoint()
        raise PipelineError(
            f"Frame extraction interrupted after scanning {frame_index:,} of "
            f"{frame_count:,} frames. Saved {counts['saved']:,} frames so far"
            + (
                f"; partial frame/GPS data is in {output_dir / 'frame_telemetry.csv'}."
                if rows else ". No synchronized frames had been saved yet."
            )
        ) from None
    finally:
        capture.release()
    if not rows:
        raise PipelineError(
            "No synchronized frames were retained. Check GPS timestamps and --timestamp-offset."
        )
    return rows, {
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": frame_count / fps,
        "processing_seconds": round(time.monotonic() - started, 2),
        "width": width,
        "height": height,
        **counts,
    }


def write_colmap_database(
    rows: list[dict[str, Any]],
    frames_dir: Path,
    output_dir: Path,
    command_prefix: list[str],
    environment: dict[str, str],
    match_overlap: int,
    resume_from_mapper: bool,
    resume_from_patch_match: bool,
    gpu_index: int,
    patch_match_max_image_size: int,
) -> None:
    database = output_dir / "colmap" / "database.db"
    database.parent.mkdir(parents=True, exist_ok=True)
    # The extracted frames directory contains only synchronized, quality-filtered
    # images, so COLMAP can read them directly without duplicating hundreds of MB.
    colmap_images = frames_dir
    for row in rows:
        if not (colmap_images / row["frame"]).is_file():
            raise PipelineError(f"Prepared frame is missing: {colmap_images / row['frame']}")
    dense_dir = output_dir / "colmap" / "dense"
    if resume_from_patch_match:
        validate_dense_workspace(dense_dir, output_dir / "colmap" / "sparse" / "0", rows)
        print(
            "Reusing sparse model and undistorted images; resuming at GPU PatchMatch.",
            flush=True,
        )
    else:
        if resume_from_mapper:
            validate_matching_database(database, len(rows))
            print("Reusing COLMAP features and matches; resuming at sparse mapping.", flush=True)
        else:
            commands = [
                [
                    "feature_extractor", "--database_path", str(database),
                    "--image_path", str(colmap_images), "--ImageReader.single_camera", "1",
                ],
                [
                    "sequential_matcher", "--database_path", str(database),
                    "--SequentialMatching.overlap", str(match_overlap),
                ],
            ]
            for args in commands:
                run_colmap(command_prefix, args, environment)

        sparse_output = output_dir / "colmap" / "sparse"
        sparse_output.mkdir(parents=True, exist_ok=True)
        run_colmap(
            command_prefix,
            [
                "mapper", "--database_path", str(database), "--image_path", str(colmap_images),
                "--output_path", str(sparse_output),
            ],
            environment,
        )
        sparse_model = sparse_output / "0"
        if not sparse_model.is_dir():
            raise PipelineError("COLMAP mapper completed without producing sparse/0.")
        run_colmap(
            command_prefix,
            ["image_undistorter", "--image_path", str(colmap_images),
             "--input_path", str(sparse_model), "--output_path", str(dense_dir),
             "--output_type", "COLMAP"],
            environment,
        )
    run_colmap(
        command_prefix,
        ["patch_match_stereo", "--workspace_path", str(dense_dir),
         "--workspace_format", "COLMAP", "--PatchMatchStereo.geom_consistency", "true",
         "--PatchMatchStereo.gpu_index", str(gpu_index),
         "--PatchMatchStereo.max_image_size", str(patch_match_max_image_size),
         "--PatchMatchStereo.cache_size", "4"],
        environment,
    )
    run_colmap(
        command_prefix,
        ["stereo_fusion", "--workspace_path", str(dense_dir),
         "--workspace_format", "COLMAP", "--input_type", "geometric",
         "--output_path", str(dense_dir / "fused.ply")],
        environment,
    )


def validate_dense_workspace(
    dense_dir: Path, sparse_model: Path, rows: list[dict[str, Any]]
) -> None:
    if not sparse_model.is_dir():
        raise PipelineError(
            f"Cannot resume at PatchMatch: sparse model is missing: {sparse_model}. "
            "Resume from mapper or rerun reconstruction."
        )
    dense_images = dense_dir / "images"
    if not dense_images.is_dir():
        raise PipelineError(
            f"Cannot resume at PatchMatch: undistorted image folder is missing: {dense_images}. "
            "Resume from mapper or rerun reconstruction."
        )
    missing = [
        row["frame"] for row in rows
        if not (dense_images / row["frame"]).is_file()
    ]
    if missing:
        raise PipelineError(
            f"Cannot resume at PatchMatch: {len(missing)} of {len(rows)} undistorted "
            f"images are missing (first: {missing[0]}). Resume from mapper or rerun reconstruction."
        )
    patch_config = dense_dir / "stereo" / "patch-match.cfg"
    if not patch_config.is_file():
        raise PipelineError(
            f"Cannot resume at PatchMatch: COLMAP workspace config is missing: {patch_config}. "
            "Resume from mapper or rerun reconstruction."
        )


def validate_matching_database(database: Path, expected_images: int) -> None:
    if not database.is_file():
        raise PipelineError(f"COLMAP database is missing: {database}")
    try:
        connection = sqlite3.connect(f"file:{database.resolve().as_posix()}?mode=ro", uri=True)
        try:
            image_count = connection.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            feature_count = connection.execute(
                "SELECT COUNT(DISTINCT image_id) FROM keypoints"
            ).fetchone()[0]
            match_count = connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
            geometry_count = connection.execute(
                "SELECT COUNT(*) FROM two_view_geometries"
            ).fetchone()[0]
        finally:
            connection.close()
    except sqlite3.Error as exc:
        raise PipelineError(f"Could not inspect COLMAP matching database: {exc}") from exc
    if image_count != expected_images or feature_count != expected_images:
        raise PipelineError(
            f"Cannot resume at mapper: database has {image_count} images and "
            f"features for {feature_count}, but {expected_images} prepared frames are expected. "
            "Run without --resume-from-mapper to rebuild features and matches."
        )
    if match_count == 0 or geometry_count == 0:
        raise PipelineError(
            "Cannot resume at mapper: COLMAP database has no completed matches/geometries. "
            "Run without --resume-from-mapper to rebuild features and matches."
        )


def resolve_colmap_launcher(executable: str) -> tuple[list[str], dict[str, str]]:
    resolved = shutil.which(executable)
    if resolved is None and Path(executable).is_file():
        resolved = str(Path(executable).resolve())
    if resolved is None:
        raise PipelineError(f"COLMAP executable not found: {executable}")

    executable_path = Path(resolved)
    if executable_path.suffix.lower() == ".bat":
        launcher = executable_path
        binary_dir = launcher.parent / "bin"
    else:
        launcher = executable_path.parent.parent / "COLMAP.bat"
        binary_dir = executable_path.parent

    environment = os.environ.copy()
    environment["PATH"] = str(binary_dir) + os.pathsep + environment.get("PATH", "")
    plugin_dir = launcher.parent / "plugins"
    if plugin_dir.is_dir():
        existing_plugins = environment.get("QT_PLUGIN_PATH")
        environment["QT_PLUGIN_PATH"] = (
            str(plugin_dir) + os.pathsep + existing_plugins
            if existing_plugins else str(plugin_dir)
        )

    if launcher.is_file():
        prefix = ["cmd.exe", "/d", "/c", "call", str(launcher)]
    else:
        prefix = [resolved]
    return prefix, environment


def run_colmap(
    command_prefix: list[str], arguments: list[str], environment: dict[str, str]
) -> None:
    try:
        subprocess.run(
            [*command_prefix, *arguments],
            check=True,
            env=environment,
        )
    except subprocess.CalledProcessError as exc:
        status = exc.returncode & 0xFFFFFFFF
        message = (
            f"COLMAP command '{arguments[0]}' failed with exit code "
            f"{exc.returncode} (0x{status:08X})."
        )
        if status == 0xC0000135:
            message += (
                " Windows could not load a required DLL. Check that the COLMAP "
                "archive is fully extracted and its bundled bin and plugins "
                "folders are beside COLMAP.bat."
            )
        elif status == 0xC0000409 and arguments[0] == "patch_match_stereo":
            message += (
                " PatchMatch needs GPU support; make sure --colmap points to the "
                "CUDA-enabled COLMAP build, not the no-CUDA build."
            )
        raise PipelineError(message) from exc


def load_prepared_output(
    output_dir: Path, expected_inputs: dict[str, Path | None]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = output_dir / "manifest.json"
    telemetry_path = output_dir / "frame_telemetry.csv"
    if not manifest_path.is_file() or not telemetry_path.is_file():
        raise PipelineError(
            "Cannot reuse prepared output: manifest.json or frame_telemetry.csv is missing. "
            "Run preprocessing first without --reconstruct."
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with telemetry_path.open("r", newline="", encoding="utf-8-sig") as source:
            rows = list(csv.DictReader(source))
    except (OSError, json.JSONDecodeError, csv.Error) as exc:
        raise PipelineError(f"Could not read prepared output: {exc}") from exc
    if manifest.get("status") not in {"prepared", "reconstructed"}:
        raise PipelineError(
            f"Prepared output has unexpected status {manifest.get('status')!r}; "
            "run preprocessing again before reconstruction."
        )
    manifest_inputs = manifest.get("inputs", {})
    for name in ("video", "gps"):
        expected = expected_inputs[name]
        recorded = manifest_inputs.get(name)
        if expected is None or recorded != str(expected):
            raise PipelineError(
                f"Prepared output was generated from a different {name} input. "
                "Run preprocessing again before reconstruction."
            )
    if not rows:
        raise PipelineError("Prepared frame/GPS table contains no frames.")
    frames_dir = output_dir / "frames"
    for row in rows:
        image_name = row.get("frame", "")
        if not image_name or Path(image_name).name != image_name:
            raise PipelineError(f"Invalid frame filename in prepared CSV: {image_name!r}")
        if not (frames_dir / image_name).is_file():
            raise PipelineError(f"Prepared frame is missing: {frames_dir / image_name}")
    video_info = manifest.get("video")
    if not isinstance(video_info, dict):
        raise PipelineError("Prepared manifest is missing video information.")
    if video_info.get("saved") != len(rows):
        raise PipelineError(
            "Prepared frame/GPS CSV does not match the manifest's retained-frame count; "
            "run preprocessing again before reconstruction."
        )
    print(f"Reusing {len(rows):,} prepared frames from {output_dir}", flush=True)
    return rows, manifest


def run(args: argparse.Namespace) -> Path:
    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    inputs = resolve_inputs(input_dir, args.download_missing)
    metadata: Any = None
    if inputs["metadata"] is not None:
        try:
            metadata = json.loads(inputs["metadata"].read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise PipelineError(f"Invalid flight metadata JSON: {exc}") from exc
    telemetry = load_telemetry(inputs["gps"])
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_manifest = output_dir / "manifest.json"
    if args.reconstruct and prepared_manifest.is_file():
        rows, manifest = load_prepared_output(output_dir, inputs)
    else:
        rows, video_info = extract_frames(
            inputs["video"], telemetry, output_dir, args.interval, args.min_blur_score,
            args.duplicate_threshold, args.timestamp_offset,
        )
        telemetry_csv = output_dir / "frame_telemetry.csv"
        with telemetry_csv.open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        manifest = {
            "status": "prepared",
            "input_directory": str(input_dir),
            "inputs": {name: str(path) if path else None for name, path in inputs.items()},
            "video": video_info,
            "gps_samples": len(telemetry),
            "synchronization": {
                "assumption": "The first GPS sample aligns with video time zero, adjusted by timestamp_offset_seconds.",
                "timestamp_offset_seconds": args.timestamp_offset,
            },
            "preprocessing": {
                "sample_interval_seconds": args.interval,
                "minimum_laplacian_variance": args.min_blur_score,
                "duplicate_mean_absolute_difference_threshold": args.duplicate_threshold,
                "frame_counts": {key: video_info[key] for key in (
                    "sampled", "blurred", "duplicate", "unsynchronized", "saved"
                )},
            },
            "metadata": metadata,
        }
        prepared_manifest.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    if args.reconstruct:
        command_prefix, colmap_environment = resolve_colmap_launcher(args.colmap)
        if not 2 <= args.match_overlap < len(rows):
            raise PipelineError(
                f"--match-overlap must be between 2 and {len(rows) - 1} for "
                f"{len(rows)} prepared frames."
            )
        write_colmap_database(
            rows, output_dir / "frames", output_dir,
            command_prefix, colmap_environment, args.match_overlap,
            args.resume_from_mapper, args.resume_from_patch_match,
            args.gpu_index, args.patch_match_max_image_size,
        )
        manifest["status"] = "reconstructed"
        manifest["reconstruction"] = {
            "engine": "COLMAP",
            "matching": "sequential",
            "sequential_overlap": args.match_overlap,
            "patch_match_gpu_index": args.gpu_index,
            "patch_match_max_image_size": args.patch_match_max_image_size,
            "coordinate_system": "COLMAP local reconstruction coordinates; not GPS georeferenced",
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Synchronize one drone video with GPS, extract useful frames, and optionally reconstruct with COLMAP."
    )
    parser.add_argument("--input", required=True, help="Test-case folder (e.g. Test Set/airlock_test_sets/TEST_01).")
    parser.add_argument("--output", help="Output folder; defaults to <input>/output.")
    parser.add_argument("--download-missing", action="store_true", help="Download assets listed in SOURCE_URLS.txt.")
    parser.add_argument("--interval", type=float, default=1.0, help="Frame sampling interval in seconds (default: 1).")
    parser.add_argument("--min-blur-score", type=float, default=40.0, help="Minimum Laplacian variance (default: 40).")
    parser.add_argument("--duplicate-threshold", type=float, default=2.0, help="Duplicate-image difference threshold (default: 2).")
    parser.add_argument("--timestamp-offset", type=float, default=0.0, help="GPS time offset from video start in seconds.")
    parser.add_argument(
        "--reconstruct", action="store_true",
        help="Run COLMAP; reuse prepared frames/CSV when they already exist.",
    )
    parser.add_argument(
        "--match-overlap", type=int, default=10,
        help="Number of neighboring frames for sequential matching (default: 10).",
    )
    parser.add_argument(
        "--resume-from-mapper", action="store_true",
        help="Reuse existing COLMAP features/matches and continue at sparse mapping.",
    )
    parser.add_argument(
        "--resume-from-patch-match", action="store_true",
        help="Reuse the sparse model and undistorted images; start at GPU PatchMatch.",
    )
    parser.add_argument(
        "--gpu-index", default="0",
        help="GPU index for COLMAP PatchMatch (default: 0; use -1 for automatic selection).",
    )
    parser.add_argument(
        "--patch-match-max-image-size", type=int, default=1200,
        help="Maximum image dimension for PatchMatch; lower values reduce GPU memory use (default: 1200).",
    )
    parser.add_argument("--colmap", default="colmap", help="COLMAP executable name or path.")
    args = parser.parse_args()
    args.output = args.output or str(Path(args.input) / "output")
    if (args.resume_from_mapper or args.resume_from_patch_match) and not args.reconstruct:
        parser.error("Resume options require --reconstruct.")
    if args.resume_from_mapper and args.resume_from_patch_match:
        parser.error("Choose only one resume stage.")
    if args.patch_match_max_image_size == 0 or args.patch_match_max_image_size < -1:
        parser.error("--patch-match-max-image-size must be -1 or a positive integer.")
    try:
        output_dir = run(args)
    except (PipelineError, OSError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Drone flight pipeline completed: {output_dir}")
    print(f"Manifest: {output_dir / 'manifest.json'}")
    print(f"Frame/GPS table: {output_dir / 'frame_telemetry.csv'}")
    if args.reconstruct:
        print(f"COLMAP reconstruction: {output_dir / 'colmap' / 'dense' / 'fused.ply'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
