## Video cutting tools

Cut videos (and optional paired GPX tracks) at manually specified timestamps.

Run all scripts from inside `video-gpx-cutting/`. Requires Python 3.10+ and `ffmpeg`/`ffprobe` in `PATH`.


### Input and output format

Cutting input format: One video per line — filename stem followed by cut points in `ss`, `mm:ss` or `hh:mm:ss`:
```text
VID_20241210_172830_00_003  08:45
VID_20241210_105715_00_015  00:13  02:06  03:30
```

Each video is split at the given timestamps into N+1 segments. Segment starts are automatically snapped to the nearest previous keyframe for clean cuts. If a `.gpx` file with the same stem exists alongside the video, it is cut as well.

Directory structure:
- **Input**: Videos can be organized in a flat or nested directory structure.
- **Output**: The input folder hierarchy is preserved in the output:
  - `input/vid.mp4` $\rightarrow$ `output/vid_1.mp4`, `output/vid_2.mp4`, ...
  - `input/sub/vid.mp4` $\rightarrow$ `output/sub/vid_1.mp4`, ...


### CLI

```bash
python cut_manual.py \
  --input-dir /path/to/videos \
  --output-dir /path/to/output \
  --list-file cuts.txt \
  --dry-run
```

Omit `--list-file` to read from stdin. Remove `--dry-run` to cut.


### GUI

```bash
python cut_manual_gui.py
```

1. Select **Input** and **Output** directories.
2. Paste the cut table into the text box.
3. **Preview** to check parsed stems and cut points.
4. **Run** to start the processing.
   - Check **Clear output directory before processing** to wipe the output folder first (with confirmation).
