from datetime import datetime
import math
from typing import List
import xml.etree.ElementTree as ET
import dataclasses as dc

import numpy as np


def haversine_distance(
    lat1: float, lon1: float, lat2: float, lon2: float, sphere_radius: float = 6371000
) -> float:
    """
    Calculate the great circle distance between two points on the Earth.

    Args:
        lat1 (float): Latitude of the first point in decimal degrees.
        lon1 (float): Longitude of the first point in decimal degrees.
        lat2 (float): Latitude of the second point in decimal degrees.
        lon2 (float): Longitude of the second point in decimal degrees.

    Returns:
        float: Distance between the two points in meters.

    References:
        - https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return c * sphere_radius


@dc.dataclass
class TrackPoint:
    lat: float
    lon: float
    timestamp: str


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
            timestamp=trkpt.find("gpx:time", ns).text,
        )
        for trkpt in root.findall(".//gpx:trkpt", ns)
    ]


def parse_gpx_from_string(gpx_content: str) -> List[TrackPoint]:
    """Parses a GPX string and extracts coordinates with timestamps. Returns list of track points."""
    root = ET.fromstring(gpx_content)
    return _parse_gpx_root(root)


def calculate_distances_from_start(track_points: List[dict]) -> np.array:
    """Calculates cumulative path distances from the starting point for each track point."""
    distances = [0.0]
    prev = track_points[0]
    for curr in track_points[1:]:
        distances.append(distances[-1] + haversine_distance(prev.lat, prev.lon, curr.lat, curr.lon))
        prev = curr
    return np.array(distances)


def calculate_times_from_start(track_points) -> np.array:
    """Calculates relative times in seconds from the start."""
    start_time = parse_timestamp(track_points[0].timestamp)
    return np.array([parse_timestamp(p.timestamp) - start_time for p in track_points])


class GPXTrack:
    """A class for calculations based on GPX track data."""

    def __init__(self, gpx_data):
        """
        Initialize with either a list of TrackPoint or a GPX content string.

        Args:
            gpx_data: List[TrackPoint] or GPX content as string.
        """
        if isinstance(gpx_data, list):
            self.track_points = gpx_data
        elif isinstance(gpx_data, str):
            self.track_points = parse_gpx_from_string(gpx_data)
        else:
            raise ValueError("Argument must be a list of TrackPoint or a GPX content string.")

        self.distances = calculate_distances_from_start(self.track_points)
        self.times = calculate_times_from_start(self.track_points)

    def distance_to_time(self, distance):
        """Converts distance from start to time in seconds"""
        return np.interp(distance, self.distances, self.times)

    def time_to_distance(self, time):
        """Converts time in seconds to distance from start"""
        return np.interp(time, self.times, self.distances)

    def distance_to_frame_number(self, distance, fps: float = 30.0):
        """Converts distance from start to frame number"""
        return time_to_frame_number(self.distance_to_time(distance), fps)

    def frame_number_to_distance(self, frame_number, fps: float = 30.0):
        """Converts frame number to distance from start"""
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

    track = GPXTrack(gpx_content)

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
