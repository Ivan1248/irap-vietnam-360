from pathlib import Path
import argparse
import shutil
import subprocess
import re
import dataclasses as dc
from typing import Any, List, Dict

import numpy as np
from tqdm import tqdm


def parse_metadata_value(value: str) -> Any:
    """Parses a metadata value string into its appropriate type.
    Handles integers, floats, and lists of numbers.
    """

    def parse_float_list(value: str) -> List[float]:
        return [float(x) for x in value.strip().split() if x]

    parsers = [float, parse_float_list, str.strip]
    for parser in parsers:
        try:
            return parser(value)
        except ValueError:
            pass


def extract_metadata(filepath: str) -> Dict[str, Any]:
    """Extracts metadata from the INSV file using exiftool.

    Returns a dictionary with general metadata, IMU records ("Time code",
    "Accelerometer", "Angular velocity"), and per-frame records ("Time code",
    "Exposure time").
    """
    exiftool_path = shutil.which("exiftool")
    if not exiftool_path:
        print("Error: exiftool not found in PATH.")
        return {}
    try:
        out = subprocess.run(
            [exiftool_path, "-api", "largefilesupport=1", "-ee", "-G3", "-m", filepath],
            stdout=subprocess.PIPE,
            text=True,
            check=True,
        ).stdout
    except Exception as e:
        print(f"Error running exiftool: {e}")
        return {}
    metadata = {}
    for m in tqdm(re.finditer(r"\[(.*?)\]\s+([^\:]+?)\s*:\s*(.*)", out)):
        cat, key, val = m.groups()
        metadata.setdefault(cat, []).append((key, parse_metadata_value(val)))
    main = dict(metadata.pop("Main"))
    timecoded = [metadata[f"Doc{i}"] for i in range(1, len(metadata) + 1)]
    imu_records = []
    frame_records = []
    for x in timecoded:
        data = dict(x)
        if "Accelerometer" in data:
            imu_records.append(data)
        elif "Exposure Time" in data:
            frame_records.append(data)
        else:
            assert False, f"Unexpected entry: {data}."
    return dict(main=main, imu_records=imu_records, frame_records=frame_records)


@dc.dataclass
class IMUData:
    time: np.ndarray
    angular_velocity: np.ndarray
    accelerometer: np.ndarray


@dc.dataclass
class FrameData:
    time: np.ndarray
    exposure_time: np.ndarray


@dc.dataclass
class FrameOrientationData:
    time: np.ndarray
    yaw_pitch_roll: np.ndarray


def compute_sampling_rate(times: np.ndarray) -> float:
    """Computes the average sampling rate from a list of timestamps.

    Assumes that each timestamp corresponds to the beginning (or end) of one frame.
    """
    return (len(times) - 1) / (times[-1] - times[0])


def get_motion_data(metadata: Dict[str, Any]) -> Dict[str, Any]:
    timecode_unit = 1 / 1000.0
    frame_rate = metadata["main"]["Video Frame Rate"]

    imu_records = metadata["imu_records"]
    imu_data = IMUData(
        time=np.array([rec["Time Code"] for rec in imu_records]) * timecode_unit,
        angular_velocity=np.array([rec["Angular Velocity"] for rec in imu_records]),
        accelerometer=np.array([rec["Accelerometer"] for rec in imu_records]),
    )
    num_imu_records = len(imu_data.time)
    measured_imu_sampling_rate = compute_sampling_rate(imu_data.time)
    duration_imu = num_imu_records / measured_imu_sampling_rate

    frame_records = metadata["frame_records"]
    frame_data = FrameData(
        time=np.array([rec["Time Code"] for rec in frame_records]) * timecode_unit,
        exposure_time=np.array([rec["Exposure Time"] for rec in frame_records]),
    )
    num_frames = len(frame_data.time)
    measured_frame_rate = compute_sampling_rate(frame_data.time)
    duration_frames = num_frames / measured_frame_rate

    frame_rate_factor = frame_rate / measured_frame_rate

    print(
        f"Frame rate from metadata: {frame_rate:.5f} Hz, "
        + f"\nFrame rate from records: {measured_frame_rate:.5f} Hz, "
        + f"\nIMU sampling rate: {measured_imu_sampling_rate:.5f} Hz"
        + f"\nIMU sampling rate (corrected): {measured_imu_sampling_rate*frame_rate_factor:.5f} Hz, "
        + f"\nDuration from metadata: {metadata['main']['Duration']} "
        + f"\nDuration (frames): {duration_frames:.3f} s"
        + f"\nDuration (24 Hz frame rate): {num_frames/frame_rate:.3f} s"
        + f"\nDuration (IMU): {duration_imu:.3f} s, "
        + f"\nDuration (IMU, corrected): {duration_imu/frame_rate_factor:.3f} s, "
        + f"\nFrame - IMU start delta: {frame_data.time[0]-imu_data.time[0]}, "
        + f"\nFrame - IMU end delta: {frame_data.time[-1]-imu_data.time[-1]}, "
    )

    assert np.isclose(measured_frame_rate, frame_rate, atol=1e-3, rtol=1e-4)

    return dict(
        frame_rate=measured_frame_rate,
        imu_data=imu_data,
        frames=frame_data,
        imu_sampling_rate=measured_imu_sampling_rate,
        duration_frames=duration_frames,
        duration_imu=duration_imu,
        declared_frame_rate=frame_rate,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract gyroscope data from INSV files")
    parser.add_argument("input_file", help="Input INSV file path")
    args = parser.parse_args()

    if not Path(args.input_file).exists():
        print(f"Error: Input file '{args.input_file}' not found.")
        exit(1)

    print("\nExtracting metadata with exiftool:")
    metadata = extract_metadata(args.input_file)
    for k, v in metadata["main"].items():
        print(f"{k}: {v}")

    motion_data = get_motion_data(metadata)
