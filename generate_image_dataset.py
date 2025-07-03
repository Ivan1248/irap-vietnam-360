from functools import partial
from pathlib import Path
import cv2
import numpy as np

from tqdm import tqdm

from gpx_utils import GPXTrack
from fisheye_utils import get_fisheye_to_perspective_converter, iterate_video_frames


def extract_frames_by_distance(
    root_dir: str,
    output_dir: str,
    video_ext: str = ".insv",
    gpx_ext: str = ".gpx",
    dist_step: float = 10.0,
    perspective_fov_h: float = 127.0,
    output_size: tuple = (384, 256),
    fisheye_radius_factor: float = 0.94,
    fov_center: tuple = (0, 0),
):
    """
    For each subfolder in root_dir, extract frames from the fisheye video at every `distance_step` meters
    using the corresponding .gpx file, and save them as images in output_dir.
    """
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_seq_dir in filter(lambda d: d.is_dir(), root_dir.iterdir()):
        video_files = list(input_seq_dir.glob(f"*{video_ext}"))
        gpx_files = list(input_seq_dir.glob(f"*{gpx_ext}"))
        video_path, gpx_path = video_files[0], gpx_files[0]

        cap = cv2.VideoCapture(str(video_path))
        input_size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with open(gpx_path, "r", encoding="utf-8") as f:
            track = GPXTrack(gpx_content=f.read())

        print(
            f"Warning: GPS path duration: {track.times[-1]}, video duration: {total_frames / fps}. Rescaling GPS path duration."
        )
        # Hack to make the video and path duration match, TODO
        track.times = track.times * (total_frames / fps / track.times[-1])

        frame_numbers = np.unique(
            track.distance_to_frame_number(np.arange(0, track.distances[-1], dist_step), fps=fps)
        )

        frame_converter = get_fisheye_to_perspective_converter(
            input_size=input_size,
            fov_h=perspective_fov_h,
            output_size=output_size,
            fisheye_radius_factor=fisheye_radius_factor,
            fov_center=fov_center,
        )

        out_seq_dir = output_dir / input_seq_dir.name
        out_seq_dir.mkdir(parents=True, exist_ok=True)
        for frame_idx, frame in iterate_video_frames(
            cap, frames=frame_numbers, pbar_f=partial(tqdm, desc=f"Processing {input_seq_dir.name}")
        ):
            out_frame = frame_converter(frame)
            cv2.imwrite(out_seq_dir / f"{frame_idx:07d}.png", out_frame)

        print(f"Processed {input_seq_dir.name}: {len(frame_numbers)} frames saved in {out_seq_dir}.")
        cap.release()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract frames from fisheye videos every N meters using GPX."
    )
    parser.add_argument(
        "root_dir",
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
    parser.add_argument("--fisheye_radius_factor", type=float, default=0.94, help="Fisheye radius factor (fraction of image radius).")
    parser.add_argument("--tilt", type=float, default=0.0, help="Vertical tilt angle (degrees) for FOV center.")
    args = parser.parse_args()

    extract_frames_by_distance(
        args.root_dir,
        args.output_dir,
        dist_step=args.distance_step,
        perspective_fov_h=args.fov_h,
        output_size=(args.output_width, args.output_height),
        fisheye_radius_factor=args.fisheye_radius_factor,
        fov_center=(0, -args.tilt),
    )
