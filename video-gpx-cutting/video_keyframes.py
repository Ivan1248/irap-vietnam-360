import bisect
import json
import subprocess
from typing import List


def get_keyframe_timestamps(video_path: str) -> List[float]:
    """
    Extract keyframe timestamps (I-frames) from a video file using ffprobe.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v",
        "-skip_frame",
        "nokey",
        "-show_entries",
        "frame=pkt_pts_time",
        "-of",
        "json",
        video_path,
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {proc.stderr.strip()}")

    data = json.loads(proc.stdout or "{}")
    frames = data.get("frames", [])
    timestamps = [float(f["pkt_pts_time"]) for f in frames if "pkt_pts_time" in f]
    return sorted(timestamps)


def snap_to_previous_keyframe(
    requested_time_s: float, keyframe_timestamps: List[float]
) -> float:
    """
    Find the timestamp of the nearest keyframe at or before the requested time.
    """
    if not keyframe_timestamps:
        return requested_time_s

    # Find the largest k such that k <= requested_time_s
    idx = bisect.bisect_right(keyframe_timestamps, requested_time_s) - 1
    return keyframe_timestamps[idx] if idx >= 0 else requested_time_s
