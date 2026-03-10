from typing import Optional, List
import dataclasses as dc

import numpy as np


# GPS

@dc.dataclass
class TrackPoint:
    lat: float
    lon: float
    time: str
    altitude: float


@dc.dataclass
class GPSTrack:
    time: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    altitude: np.ndarray
    speed: Optional[np.ndarray] = None
    track: Optional[np.ndarray] = None

    def from_track_points(track_points: List[TrackPoint]) -> "GPSTrack":
        return GPSTrack(
            time=np.array([p.time for p in track_points]),
            lat=np.array([p.lat for p in track_points]),
            lon=np.array([p.lon for p in track_points]),
            altitude=np.array([p.altitude for p in track_points]),
        )

    def with_zero_start_time(self) -> "GPSTrack":
        return GPSTrack(
            time=self.time - self.time[0],
            lat=self.lat,
            lon=self.lon,
            altitude=self.altitude,
        )

    def __getitem__(self, index: int) -> TrackPoint:
        return TrackPoint(
            lat=self.lat[index],
            lon=self.lon[index],
            time=self.time[index],
            altitude=self.altitude[index],
        )

    def __len__(self) -> int:
        return len(self.time)

# IMU

@dc.dataclass
class IMUData:
    """IMU samples in the body frame following ROS REP-103 conventions.

    Body frame convention (ROS REP-103):
    - x: forward, y: left, z: up (right-handed)
    - Fixed axis rotations: roll about X, pitch about Y, yaw about Z
    - Gravity vector: [0, 0, -9.81] (points down in body frame)
    """

    time: np.ndarray
    angular_velocity: np.ndarray  # Shape (N, 3): [gyro_x, gyro_y, gyro_z] in rad/s
    accelerometer: np.ndarray  # Shape (N, 3): [accel_x, accel_y, accel_z] in m/s²

    AXIS_NAMES = {0: "x", 1: "y", 2: "z"}

    # Axis indices (ROS REP-103)
    ROLL_AXIS = 0  # gyro_x (rotation about X-axis)
    PITCH_AXIS = 1  # gyro_y (rotation about Y-axis)
    YAW_AXIS = 2  # gyro_z (rotation about Z-axis)

    # Coordinate axis indices (ROS REP-103 body frame)
    FORWARD_AXIS = 0  # X-axis (forward direction)
    LEFT_AXIS = 1  # Y-axis (left direction)
    UP_AXIS = 2  # Z-axis (up direction)

    # Physical constants
    STANDARD_GRAVITY_ACCELERATION = np.array([0.0, 0.0, +9.81])  # Gravity vector in body frame

    @property
    def yaw_rate(self) -> np.ndarray:
        """Yaw rate in rad/s (positive = counter-clockwise about Z-axis)."""
        return self.angular_velocity[:, self.YAW_AXIS]

    @property
    def roll_rate(self) -> np.ndarray:
        """Roll rate in rad/s (positive = right wing down about X-axis)."""
        return self.angular_velocity[:, self.ROLL_AXIS]

    @property
    def pitch_rate(self) -> np.ndarray:
        """Pitch rate in rad/s (positive = nose up about Y-axis)."""
        return self.angular_velocity[:, self.PITCH_AXIS]

# Camera

@dc.dataclass
class FrameData:
    time: np.ndarray
    exposure_time: np.ndarray


@dc.dataclass
class FrameOrientationData:
    time: np.ndarray
    yaw_pitch_roll: np.ndarray
