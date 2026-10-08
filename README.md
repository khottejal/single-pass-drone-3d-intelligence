# Single-pass drone preprocessing and 3D reconstruction

This first implementation covers data ingestion, video/GPS synchronization,
quality-filtered frame extraction, and an optional COLMAP reconstruction. It
does not run Nemotron or perform scene understanding.

## Inputs

Each test-case directory can contain:

- One video (`.mp4`, `.mov`, `.m4v`, `.avi`, or `.mkv`)
- A GPS/telemetry CSV with a timestamp, latitude, longitude, and optionally
  altitude, roll, pitch, and yaw/heading columns
- An optional flight metadata JSON file
- An optional camera intrinsics file (recorded in the manifest; calibration
  ingestion is not yet applied to COLMAP)
- An optional `SOURCE_URLS.txt` containing `Video: <URL>`, `GPS: <URL>`, and
  `Calibration: <URL>` entries

Timestamp headers accepted include `timestamp`, `time`, `datetime`, `utc_time`,
`gps_time`, and `video_time_s` (used when a timestamp column is empty);
coordinate headers accept common latitude/longitude/altitude variants,
including the AirLock dataset's `drone_lat`, `drone_lon`, and
`drone_altitude_m`. CSV timestamps may be numeric seconds, epoch milliseconds,
ISO datetimes, or clock values such as `00:00:01.250`.

## Setup

Use Python 3.10 or newer:

```powershell
python -m pip install -r requirements.txt
```

COLMAP is required only when using `--reconstruct`; install it separately and
make the executable available on `PATH`. The pipeline does not install COLMAP
automatically. For the Windows release ZIP, pass its `bin\colmap.exe` path;
the pipeline uses the sibling `COLMAP.bat` launcher so the bundled DLL and Qt
plugin paths are configured.

## Prepare one test case

```powershell
python .\drone_pipeline.py --input ".\tests\Test_Set\airlock_test_sets\TEST_01"
```

Downloads are opt-in. Without `--download-missing`, place the video and GPS CSV
in the test-case folder first. Outputs default to that folder's `output`
subdirectory; pass `--output` to choose a different location. The example path
matches the project layout shown in the VS Code Explorer; adjust it if you put
`Test_Set` elsewhere.

Useful controls:

```powershell
python .\drone_pipeline.py --input ".\TEST_01" --interval 0.5 --min-blur-score 40
python .\drone_pipeline.py --input ".\TEST_01" --timestamp-offset 1.25
python .\drone_pipeline.py --input ".\tests\Test_Set\airlock_test_sets\TEST_01" --reconstruct
```

When `--reconstruct` is used and a valid `output/manifest.json`,
`output/frame_telemetry.csv`, and all referenced frames already exist, the
pipeline reuses them instead of decoding the video again. It verifies the
recorded video/GPS paths and retained frame count first. COLMAP sequential
matching checks nearby frames (10-frame overlap by default), rather than
comparing every frame against every other frame. Change the overlap with
`--match-overlap`; matching fewer neighbors is faster but can miss useful
feature matches.

If COLMAP stops after matching has completed but before sparse mapping starts,
resume from the saved database without repeating feature extraction or matching:

```powershell
python .\drone_pipeline.py --input ".\tests\Test_Set\airlock_test_sets\TEST_01" --reconstruct --resume-from-mapper --colmap "D:\path\to\colmap-x64-windows-nocuda\bin\colmap.exe"
```

The resume option checks that the database contains features for every prepared
frame and has saved match data before continuing.

If sparse mapping and undistortion finish but dense stereo fails, resume from
the existing undistorted workspace instead of rerunning those stages:

```powershell
python .\drone_pipeline.py --input ".\tests\Test_Set\airlock_test_sets\TEST_01" --reconstruct --resume-from-patch-match --gpu-index 0 --colmap "D:\path\to\colmap-x64-windows-cuda\bin\colmap.exe"
```

PatchMatch requires a CUDA-enabled COLMAP build. The default maximum image size
is 1200 pixels to reduce GPU memory usage; set `--patch-match-max-image-size -1`
to keep the full image size if the GPU has enough memory. The selected GPU and
size are included in the reconstruction manifest.

The default synchronization assumption is that the first GPS sample aligns
with video time zero. Set `--timestamp-offset` if the telemetry recording
started earlier or later. Frames without a GPS sample bracketing their adjusted
timestamp are omitted and counted in the manifest; the pipeline does not
silently extrapolate positions.

During extraction, OpenCV decodes frames sequentially but only retrieves and
processes sampled frames, and the CLI prints scan progress. If interrupted,
already written frames and a partial `frame_telemetry.csv` checkpoint are kept;
a new run currently starts extraction again from the beginning.

## Outputs

```text
output/
  manifest.json
  frame_telemetry.csv
  frames/
  colmap/                 # only when --reconstruct is used
    database.db
    sparse/
    dense/fused.ply
```

COLMAP reads the existing extracted `frames/` directory directly; it does not
create a second copy of the images.

The point cloud is in COLMAP's reconstruction coordinate system. This initial
version records GPS positions for the retained frames but does **not** yet align
the reconstruction to a real-world CRS or claim metric/georeferenced accuracy.
Camera calibration file contents are likewise not yet wired into COLMAP. Those
steps require confirming the exact calibration and flight metadata formats in
the supplied test set.

## Tests

```powershell
python -m unittest discover -s tests -v
```

## GitHub and input data

Keep downloaded drone flights, GPS telemetry, extracted frames, COLMAP databases,
and reconstruction outputs out of Git. These files can be very large and may
contain sensitive location information. `.gitignore` excludes the test dataset
folders and common media, telemetry, and reconstruction file types. Share code
and setup instructions in the repository; obtain permission and review dataset
license terms before distributing any input data.
