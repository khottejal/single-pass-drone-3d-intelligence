import csv
import sqlite3
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from drone_pipeline import (
    PipelineError,
    Telemetry,
    interpolate_telemetry,
    load_prepared_output,
    load_telemetry,
    parse_time_value,
    resolve_colmap_launcher,
    run_colmap,
    validate_dense_workspace,
    validate_matching_database,
)


class TimestampTests(unittest.TestCase):
    def test_clock_timestamp(self):
        seconds, original = parse_time_value("00:00:02.500")
        self.assertEqual(seconds, 2.5)
        self.assertEqual(original, "00:00:02.500")

    def test_epoch_milliseconds(self):
        seconds, _ = parse_time_value("1700000000000")
        self.assertEqual(seconds, 1700000000)


class TelemetryTests(unittest.TestCase):
    def test_loads_and_normalizes_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gps.csv"
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.writer(output)
                writer.writerow(["timestamp", "latitude", "longitude", "altitude", "yaw"])
                writer.writerow(["00:00:01", "19.1", "73.1", "80", "179"])
                writer.writerow(["00:00:03", "19.3", "73.3", "84", "-179"])
            samples = load_telemetry(path)
        self.assertEqual([sample.seconds for sample in samples], [0, 2])
        midpoint = interpolate_telemetry(samples, 1)
        self.assertIsNotNone(midpoint)
        self.assertAlmostEqual(midpoint.latitude, 19.2)
        self.assertAlmostEqual(midpoint.altitude, 82)
        self.assertAlmostEqual(abs(midpoint.yaw), 180)

    def test_does_not_extrapolate(self):
        samples = [
            Telemetry(0, "t0", 19, 73, 80),
            Telemetry(1, "t1", 20, 74, 81),
        ]
        self.assertIsNone(interpolate_telemetry(samples, -0.01))
        self.assertIsNone(interpolate_telemetry(samples, 1.01))

    def test_loads_airlock_video_time_and_drone_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gps.csv"
            with path.open("w", newline="", encoding="utf-8") as output:
                writer = csv.writer(output)
                writer.writerow([
                    "frame_index", "video_time_s", "timestamp", "drone_lat",
                    "drone_lon", "drone_altitude_m",
                ])
                writer.writerow([0, "0.000", "", "30.276501", "-97.764242", "445.9"])
                writer.writerow([1, "0.016", "", "30.276502", "-97.764243", "446.0"])
            samples = load_telemetry(path)
        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].seconds, 0)
        self.assertEqual(samples[1].seconds, 0.016)
        self.assertAlmostEqual(samples[0].latitude, 30.276501)
        self.assertAlmostEqual(samples[0].longitude, -97.764242)
        self.assertAlmostEqual(samples[0].altitude, 445.9)


class PreparedOutputTests(unittest.TestCase):
    def test_loads_valid_prepared_frames_without_processing_video(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "frames").mkdir()
            (output / "frames" / "frame_000.jpg").write_bytes(b"frame")
            (output / "manifest.json").write_text(
                '{"status":"prepared","inputs":{"video":"video.mp4","gps":"gps.csv"},'
                '"video":{"saved":1}}',
                encoding="utf-8",
            )
            (output / "frame_telemetry.csv").write_text(
                "frame,video_seconds\nframe_000.jpg,0\n", encoding="utf-8"
            )
            rows, manifest = load_prepared_output(
                output, {"video": Path("video.mp4"), "gps": Path("gps.csv")}
            )
        self.assertEqual(rows[0]["frame"], "frame_000.jpg")
        self.assertEqual(manifest["video"]["saved"], 1)

    def test_rejects_incomplete_prepared_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "frames").mkdir()
            (output / "frames" / "frame_000.jpg").write_bytes(b"frame")
            (output / "manifest.json").write_text(
                '{"status":"prepared","inputs":{"video":"video.mp4","gps":"gps.csv"},'
                '"video":{"saved":2}}',
                encoding="utf-8",
            )
            (output / "frame_telemetry.csv").write_text(
                "frame\nframe_000.jpg\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(PipelineError, "does not match"):
                load_prepared_output(
                    output, {"video": Path("video.mp4"), "gps": Path("gps.csv")}
                )


class ColmapLauncherTests(unittest.TestCase):
    def test_uses_bundled_batch_launcher_and_runtime_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary_dir = root / "bin"
            binary_dir.mkdir()
            (binary_dir / "colmap.exe").touch()
            (root / "COLMAP.bat").touch()
            (root / "plugins").mkdir()

            prefix, environment = resolve_colmap_launcher(str(binary_dir / "colmap.exe"))

        self.assertEqual(prefix[:4], ["cmd.exe", "/d", "/c", "call"])
        self.assertTrue(prefix[4].endswith("COLMAP.bat"))
        self.assertIn(str(binary_dir), environment["PATH"])
        self.assertIn(str(root / "plugins"), environment["QT_PLUGIN_PATH"])

    def test_reports_missing_windows_dll(self):
        with patch(
            "drone_pipeline.subprocess.run",
            side_effect=subprocess.CalledProcessError(
                0xC0000135, ["colmap.exe", "feature_extractor"]
            ),
        ):
            with self.assertRaisesRegex(PipelineError, "required DLL"):
                run_colmap(["colmap.exe"], ["feature_extractor"], {})

    def test_validates_database_before_resuming_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "database.db"
            connection = sqlite3.connect(database)
            connection.executescript(
                "CREATE TABLE images (image_id INTEGER);"
                "CREATE TABLE keypoints (image_id INTEGER);"
                "CREATE TABLE matches (pair_id INTEGER);"
                "CREATE TABLE two_view_geometries (pair_id INTEGER);"
                "INSERT INTO images VALUES (1), (2);"
                "INSERT INTO keypoints VALUES (1), (2);"
                "INSERT INTO matches VALUES (1);"
                "INSERT INTO two_view_geometries VALUES (1);"
            )
            connection.close()
            validate_matching_database(database, expected_images=2)

    def test_rejects_missing_match_data_when_resuming(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "database.db"
            connection = sqlite3.connect(database)
            connection.executescript(
                "CREATE TABLE images (image_id INTEGER);"
                "CREATE TABLE keypoints (image_id INTEGER);"
                "CREATE TABLE matches (pair_id INTEGER);"
                "CREATE TABLE two_view_geometries (pair_id INTEGER);"
                "INSERT INTO images VALUES (1);"
                "INSERT INTO keypoints VALUES (1);"
            )
            connection.close()
            with self.assertRaisesRegex(PipelineError, "no completed matches"):
                validate_matching_database(database, expected_images=1)

    def test_validates_undistorted_workspace_for_patch_match_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sparse_model = root / "sparse" / "0"
            sparse_model.mkdir(parents=True)
            dense = root / "dense"
            (dense / "images").mkdir(parents=True)
            (dense / "images" / "frame_000.jpg").write_bytes(b"image")
            (dense / "stereo").mkdir()
            (dense / "stereo" / "patch-match.cfg").write_text("", encoding="utf-8")
            validate_dense_workspace(
                dense, sparse_model, [{"frame": "frame_000.jpg"}]
            )

    def test_rejects_missing_undistorted_images_for_patch_match_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sparse_model = root / "sparse" / "0"
            sparse_model.mkdir(parents=True)
            dense = root / "dense"
            (dense / "images").mkdir(parents=True)
            with self.assertRaisesRegex(PipelineError, "undistorted images are missing"):
                validate_dense_workspace(
                    dense, sparse_model, [{"frame": "frame_000.jpg"}]
                )

    def test_explains_cuda_requirement_for_patch_match_crash(self):
        with patch(
            "drone_pipeline.subprocess.run",
            side_effect=subprocess.CalledProcessError(
                0xC0000409, ["colmap.exe", "patch_match_stereo"]
            ),
        ):
            with self.assertRaisesRegex(PipelineError, "CUDA-enabled COLMAP"):
                run_colmap(["colmap.exe"], ["patch_match_stereo"], {})


if __name__ == "__main__":
    unittest.main()
