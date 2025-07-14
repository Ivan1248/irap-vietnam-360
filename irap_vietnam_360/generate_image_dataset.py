from functools import partial
from pathlib import Path
import cv2
import numpy as np
import warnings

from tqdm import tqdm
import av

from .gpx_utils import GPXTrack
from .fisheye_utils import (
    get_fisheye_to_perspective_converter,
    iterate_video_frames,
    PyAVVideoReader,
)
from .stabilization import estimate_orientations
from scipy.spatial.transform import Rotation as R


def extract_frames_by_distance(
    root_dir: str,
    output_dir: str,
    video_ext: str = ".insv",
    gpx_ext: str = ".gpx",
    dist_step: float = 10.0,
    perspective_fov_h: float = 127.0,
    output_size: tuple = (384, 256),
    fisheye_radius_factor: float = 0.94,
    yaw_pitch_roll: tuple = (0.0, 0.0, 0.0),
    stabilize: bool = False,
):
    """
    For each subfolder in root_dir, extract frames from the fisheye video at every `distance_step` meters
    using the corresponding .gpx file, and save them as images in output_dir.
    """
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    yaw_pitch_roll = np.array(yaw_pitch_roll, dtype=np.float32)

    for input_seq_dir in filter(lambda d: d.is_dir(), root_dir.iterdir()):
        video_files = list(input_seq_dir.glob(f"*{video_ext}"))
        gpx_files = list(input_seq_dir.glob(f"*{gpx_ext}"))
        video_path, gpx_path = video_files[0], gpx_files[0]

        assert (
            len(video_files) == 1
        ), f"Expected exactly one {video_ext} video file in {input_seq_dir}, found {len(video_files)}."

        reader = PyAVVideoReader(av.open(video_path), 0)

        if stabilize:
            orientations = estimate_orientations(video_path, False).yaw_pitch_roll
            if len(orientations) != reader.num_frames:
                message = f"The number of estimated orientations ({len(orientations)}) does not match the number of frames ({reader.num_frames})."
                if len(orientations) > reader.num_frames:
                    warnings.warn(message)
                else:
                    raise ValueError(message)

        with open(gpx_path, "r", encoding="utf-8") as f:
            track = GPXTrack(f.read())

        print(
            f"Warning: GPS path duration: {track.times[-1]}, video duration: {reader.num_frames / reader.fps}. Rescaling GPS path duration."
        )
        # Hack to make the video and path duration match, TODO
        track.times = track.times * (reader.num_frames / reader.fps / track.times[-1])

        frame_indices = np.unique(
            track.distance_to_frame_number(
                np.arange(0, track.distances[-1], dist_step), fps=reader.fps
            )
        )

        def get_frame_converter(yaw_pitch_roll):
            return get_fisheye_to_perspective_converter(
                input_size=reader.frame_size,
                fov_h=perspective_fov_h,
                output_size=output_size,
                fisheye_radius_factor=fisheye_radius_factor,
                yaw_pitch_roll=yaw_pitch_roll,
            )

        frame_converter = get_frame_converter(yaw_pitch_roll)
        out_seq_dir = output_dir / input_seq_dir.name
        out_seq_dir.mkdir(parents=True, exist_ok=True)
        for frame_idx, frame in iterate_video_frames(
            reader,
            frames=frame_indices,
            pbar_f=partial(tqdm, desc=f"Processing {input_seq_dir.name}"),
        ):
            if stabilize:
                # Convert to rotation matrices
                orientation_rot = R.from_euler("zyx", orientations[frame_idx], degrees=True)
                yaw_pitch_roll_rot = R.from_euler("zyx", yaw_pitch_roll, degrees=True)
                combined_rot = yaw_pitch_roll_rot * orientation_rot.inv()
                frame_converter = get_frame_converter(combined_rot.as_euler("zyx", degrees=True))
            out_frame = frame_converter(frame)
            # draw orientation text on frame
            if stabilize:
                orientation_text = f"Yaw: {orientations[frame_idx][0]:.2f}, Pitch: {orientations[frame_idx][1]:.2f}, Roll: {orientations[frame_idx][2]:.2f}"
                cv2.putText(
                    out_frame,
                    orientation_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )
            cv2.imwrite(out_seq_dir / f"{frame_idx:07d}.png", out_frame)

        print(
            f"Processed {input_seq_dir.name}: {len(frame_indices)} frames saved in {out_seq_dir}."
        )
        reader.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract frames from fisheye videos every N meters using GPX."
    )
    parser.add_argument(
        "input_dir",
        type=str,
        default="input",
        help="Root directory containing subfolders with .insv and .gpx files.",
    )
    parser.add_argument(
        "output_dir", type=str, default="output", help="Directory to save extracted images."
    )
    parser.add_argument(
        "--distance_step", type=float, default=10.0, help="Distance step in meters."
    )
    parser.add_argument(
        "--fov_h", type=float, default=127.0, help="Horizontal FOV for perspective output."
    )
    parser.add_argument("--output_width", type=int, default=384, help="Output image width.")
    parser.add_argument("--output_height", type=int, default=288, help="Output image height.")
    parser.add_argument(
        "--fisheye_radius_factor",
        type=float,
        default=0.94,
        help="Fisheye radius factor (fraction of image radius).",
    )
    parser.add_argument(
        "--yaw", type=float, default=0.0, help="Horizontal rotation angle (degrees)."
    )
    parser.add_argument(
        "--pitch", type=float, default=0.0, help="Vertical rotation angle (degrees)."
    )
    parser.add_argument(
        "--roll", type=float, default=0.0, help="Optical axis rotation angle (degrees)."
    )
    parser.add_argument(
        "--stabilize",
        action="store_true",
        help="Apply stabilization to the fisheye frames based on motion sensor data.",
    )
    args = parser.parse_args()

    extract_frames_by_distance(
        args.input_dir,
        args.output_dir,
        dist_step=args.distance_step,
        perspective_fov_h=args.fov_h,
        output_size=(args.output_width, args.output_height),
        fisheye_radius_factor=args.fisheye_radius_factor,
        yaw_pitch_roll=(args.yaw, args.pitch, args.roll),
        stabilize=args.stabilize,
    )
