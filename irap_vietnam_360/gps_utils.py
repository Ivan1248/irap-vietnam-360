from datetime import datetime
from typing import List
import xml.etree.ElementTree as ET
from pathlib import Path

from .data import GPSTrack, TrackPoint


# Parsing


def parse_timestamp(timestamp_str: str) -> float:
    """Parse ISO timestamp and return seconds since epoch"""
    dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
    return dt.timestamp()


def _parse_gpx_root(root) -> List[dict]:
    """Parses GPX root element and extract track points. Returns list of track points with lat, lon,
    and timestamp.
    """
    ns = {"gpx": "http://www.topografix.com/GPX/1/1"}  # GPX 1.1 namespace

    # Find all track points
    return [
        TrackPoint(
            lat=float(trkpt.get("lat")),
            lon=float(trkpt.get("lon")),
            time=parse_timestamp(trkpt.find("gpx:time", ns).text),
            altitude=float(trkpt.find("gpx:ele", ns).text),
        )
        for trkpt in root.findall(".//gpx:trkpt", ns)
    ]


def parse_gpx_from_string(gpx_content: str) -> List[TrackPoint]:
    """Parses a GPX string and extracts coordinates with timestamps. Returns list of track points."""
    root = ET.fromstring(gpx_content)
    return _parse_gpx_root(root)


def parse_gpx_track_from_string(gpx_content: str) -> GPSTrack:
    """Parses a GPX string and extracts coordinates with timestamps. Returns list of track points."""
    root = ET.fromstring(gpx_content)
    return GPSTrack.from_track_points(_parse_gpx_root(root))


def parse_gpx_file(gpx_path: str | Path) -> GPSTrack:
    """Parse GPX file and extract track points with timing.

    This function reads a GPX file and returns a GPXTrack object with relative timestamps.
    It uses the existing parsing infrastructure to avoid code duplication.
    """
    p = Path(gpx_path)
    with open(p, "r", encoding="utf-8") as f:
        gpx_content = f.read()
    track_points = parse_gpx_from_string(gpx_content)
    track = GPSTrack.from_track_points(track_points)
    return track
