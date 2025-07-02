from datetime import datetime
import math
from typing import List
import xml.etree.ElementTree as ET

import numpy as np


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth in meters.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    # Radius of earth in meters
    r = 6371000
    return c * r


def _parse_gpx_root(root) -> List[dict]:
    """
    Parse GPX root element and extract track points.
    Returns list of track points with lat, lon, and timestamp.
    """
    # Define namespace
    ns = {"gpx": "http://www.topografix.com/GPX/1/1"}

    # Find all track points
    track_points = []
    for trkpt in root.findall(".//gpx:trkpt", ns):
        lat = float(trkpt.get("lat"))
        lon = float(trkpt.get("lon"))
        time_elem = trkpt.find("gpx:time", ns)
        timestamp = time_elem.text if time_elem is not None else None
        track_points.append(dict(lat=lat, lon=lon, timestamp=timestamp))
    return track_points


def parse_gpx_from_string(gpx_content: str) -> List[dict]:
    """
    Parses a GPX content from string and extracts coordinates with timestamps.
    Returns list of track points.
    """
    root = ET.fromstring(gpx_content)
    return _parse_gpx_root(root)


def calculate_distances_from_start(track_points: List[dict]) -> np.array:
    """Calculates cumulative path distances from the starting point for each track point."""
    distances = [0.0]
    prev = track_points[0]
    for curr in track_points[1:]:
        distances.append(
            distances[-1] + haversine_distance(prev["lat"], prev["lon"], curr["lat"], curr["lon"])
        )
        prev = curr
    return np.array(distances)


def calculate_times_from_start(track_points) -> np.array:
    """Calculates relative times in seconds from the start."""
    start_time = parse_timestamp(track_points[0]["timestamp"])
    return np.array([parse_timestamp(p["timestamp"]) - start_time for p in track_points])


class GPXTrack:
    """
    A class to process GPX track data and convert between distances, times, and frame numbers.
    """

    def __init__(self, track_points: List[dict] = None, gpx_content: str = None):
        """
        Initialize with GPX data from various sources.

        Args:
            track_points: Pre-parsed track points
            gpx_content: GPX content as string
            gpx_file_path: Path to GPX file
        """
        # Load track points from provided source
        if track_points is not None:
            self.track_points = track_points
        elif gpx_content is not None:
            self.track_points = parse_gpx_from_string(gpx_content)
        else:
            raise ValueError("Must provide either track_points, gpx_content, or gpx_file_path")

        if not self.track_points:
            raise ValueError("No track points found")

        # Pre-calculate distances and times
        self.distances = calculate_distances_from_start(self.track_points)
        self.times = calculate_times_from_start(self.track_points)

    def distance_to_time(self, distance):
        """Convert distance from start to time in seconds"""
        return np.interp(distance, self.distances, self.times)

    def time_to_distance(self, time):
        """Convert time in seconds to distance from start"""
        return np.interp(time, self.times, self.distances)

    def distance_to_frame_number(self, distance, fps: float = 30.0):
        """Convert distance from start to frame number"""
        return time_to_frame_number(self.distance_to_time(distance), fps)

    def frame_number_to_distance(self, frame_number, fps: float = 30.0):
        """Convert frame number to distance from start"""
        return self.time_to_distance(frame_number_to_time(frame_number, fps))


def time_to_frame_number(time, fps: float = 30.0) -> int:
    return np.array(time * fps + 0.5, dtype=int)


def frame_number_to_time(frame_number, fps: float = 30.0) -> float:
    return frame_number / fps


def parse_timestamp(timestamp_str: str) -> float:
    """Parse ISO timestamp and return seconds since epoch"""
    dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
    return dt.timestamp()


# Example usage with your GPX data
if __name__ == "__main__":
    """
    Example of how to use the function with your GPX data
    """

    # Load the actual GPX file
    with open("activity_17792202410.gpx", "r", encoding="utf-8") as file:
        gpx_content = file.read()

    track = GPXTrack(gpx_content=gpx_content)

    target_distances = np.array([x * 10 for x in range(100)])

    frame_numbers = track.distance_to_frame_number(target_distances, fps=30.0)

    print("Distance (m) -> Frame Number")
    print("-" * 30)
    for dist, frame in zip(target_distances[:20], frame_numbers[:20]):  # Show first 20
        print(f"{dist:8.1f} -> {frame:6d}")
    print("...")

    # Show some statistics about the track
    max_distance = max(track.distances)
    max_time = max(track.times)

    print(f"\nTrack Statistics:")
    print(f"Total track points: {len(track.track_points)}")
    print(f"Maximum distance from start: {max_distance:.1f} meters")
    print(f"Total time: {max_time:.1f} seconds ({max_time/60:.1f} minutes)")

    # Demonstrate new functions
    print(f"\nFunction Examples:")
    print(
        f"Distance 300m -> Time: {track.distance_to_time(300):.1f}s -> Frame: {track.distance_to_frame_number(300)}"
    )
    print(
        f"Frame 1501 -> Time: {frame_number_to_time(1501):.1f}s -> Distance: {track.frame_number_to_distance(1501):.1f}m"
    )
    print(
        f"Time 50s -> Distance: {track.time_to_distance(50):.1f}m -> Frame: {time_to_frame_number(50)}"
    )
