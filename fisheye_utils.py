import cv2
import numpy as np
import math
from tqdm import tqdm
import typing as T
from pathlib import Path
from functools import partial

DEFAULT_FISHEYE_RADIUS_FACTOR = 0.95


def get_aspect_ratio_from_fov(
    fov, projection: T.Literal["perspective", "equirectangular"] = "perspective"
):
    if projection == "perspective":
        return math.tan(math.radians(fov[0] / 2)) / math.tan(math.radians(fov[1] / 2))
    elif projection == "equirectangular":
        return 2.0  # todo


def get_perspective_output_size(
    input_size,
    fov,
    fisheye_radius_factor,
    projection: T.Literal["perspective", "equirectangular"] = "perspective",
):
    if projection == "perspective":
        # Fisheye effective focal length in pixels (radius of fisheye circle)
        fisheye_radius = min(input_size[0] / 2, input_size[1] / 2) * fisheye_radius_factor
        # Perspective output size so that center scale matches fisheye center
        output_width = 2 * fisheye_radius * math.tan(math.radians(fov[0] / 2))
        output_height = 2 * fisheye_radius * math.tan(math.radians(fov[1] / 2))
        return int(round(output_width)), int(round(output_height))
    elif projection == "equirectangular":
        return input_size[1] * 2, input_size[1]  # Standard equirectangular aspect ratio


def get_fov_v(aspect_ratio, fov_h):
    return 2 * math.degrees(math.atan(math.tan(math.radians(fov_h / 2)) / aspect_ratio))


def create_fisheye_to_perspective_map(
    input_size,
    output_size=None,
    fov=(120, 120),
    fisheye_center=None,
    fisheye_radius_factor=DEFAULT_FISHEYE_RADIUS_FACTOR,
):
    """
    Create mapping tables for fisheye to perspective projection

    Args:
        input_width, input_height: Input image dimensions
        fov_h, fov_v: Horizontal and vertical field of view in degrees
        fisheye_center_x, fisheye_center_y: Fisheye circle center (None = auto-detect as image center)
        fisheye_radius_factor: Fisheye radius as fraction of min(center_x, center_y)
    """
    input_width, input_height = input_size
    fov_h, fov_v = fov

    if output_size is None:
        output_size = get_perspective_output_size(input_size, fov, fisheye_radius_factor)
    output_width, output_height = output_size

    if fisheye_center is None:
        fisheye_center = (input_width / 2, input_height / 2)
    fisheye_cx, fisheye_cy = fisheye_center

    map_x = np.zeros((output_height, output_width), dtype=np.float32)
    map_y = np.zeros((output_height, output_width), dtype=np.float32)

    # Fisheye parameters
    fisheye_radius = min(fisheye_cx, fisheye_cy) * fisheye_radius_factor

    # Convert FOV to focal length equivalent for perspective projection
    focal_length_x = output_width / (2 * math.tan(math.radians(fov_h / 2)))
    focal_length_y = output_height / (2 * math.tan(math.radians(fov_v / 2)))

    # Perspective projection parameters
    perspective_cx = output_width / 2
    perspective_cy = output_height / 2

    for y in range(output_height):
        for x in range(output_width):
            # Perspective projection: convert pixel coordinates to normalized image coordinates
            x_norm = (x - perspective_cx) / focal_length_x
            y_norm = (y - perspective_cy) / focal_length_y

            # Convert to 3D ray direction (perspective projection)
            ray_x = x_norm
            ray_y = y_norm
            ray_z = 1.0  # Looking down positive Z axis

            # Normalize the ray
            ray_length = math.sqrt(ray_x**2 + ray_y**2 + ray_z**2)
            ray_x /= ray_length
            ray_y /= ray_length
            ray_z /= ray_length

            # Convert 3D ray to fisheye coordinates
            # Calculate angle from optical axis (Z-axis)
            theta = math.acos(max(-1, min(1, ray_z)))  # Clamp to avoid numerical errors

            if theta > 0 and theta < math.pi / 2:  # Only map rays in front hemisphere
                # For equidistant fisheye projection: r = f * theta
                rho = fisheye_radius * theta / (math.pi / 2)

                # Calculate azimuth angle
                phi = math.atan2(ray_y, ray_x)

                # Convert to fisheye image coordinates
                fisheye_x = fisheye_cx + rho * math.cos(phi)
                fisheye_y = fisheye_cy + rho * math.sin(phi)

                # Check bounds
                if 0 <= fisheye_x < input_width and 0 <= fisheye_y < input_height:
                    map_x[y, x] = fisheye_x
                    map_y[y, x] = fisheye_y

    return (map_x, map_y), output_size


def create_dual_fisheye_to_equirectangular_map(
    input_width,
    input_height,
    output_aspect_ratio=2.0,
    fisheye_radius_factor=DEFAULT_FISHEYE_RADIUS_FACTOR,
    front_fisheye_position="left",
):
    """
    Create mapping tables for dual fisheye to equirectangular projection

    Args:
        input_width, input_height: Input image dimensions
        output_aspect_ratio: Output aspect ratio (2.0 = standard equirectangular)
        fisheye_radius_factor: Fisheye radius as fraction of available space
        front_fisheye_position: 'right'/'left' for side_by_side, 'top'/'bottom' for top_bottom
    """
    # Auto-compute output dimensions
    output_width = input_width
    output_height = int(input_width / output_aspect_ratio)
    fisheye_width = input_width // 2
    fisheye_height = input_height

    map_x = np.zeros((output_height, output_width), dtype=np.float32)
    map_y = np.zeros((output_height, output_width), dtype=np.float32)

    # Fisheye parameters
    center_x = fisheye_width / 2
    center_y = fisheye_height / 2
    radius = min(center_x, center_y) * fisheye_radius_factor

    for y in range(output_height):
        for x in range(output_width):
            # Convert to spherical coordinates
            lon = (x / output_width) * 2 * math.pi - math.pi
            lat = (y / output_height) * math.pi - math.pi / 2

            # Convert to 3D coordinates
            x3d = math.cos(lat) * math.cos(lon)
            y3d = math.sin(lat)
            z3d = math.cos(lat) * math.sin(lon)

            # Determine which fisheye to use and calculate offset
            use_front = z3d >= 0

            if (use_front and front_fisheye_position == "right") or (
                not use_front and front_fisheye_position == "left"
            ):
                offset_x, offset_y = fisheye_width, 0
            else:
                offset_x, offset_y = 0, 0

            # Flip coordinates for back camera
            if not use_front:
                x3d = -x3d
                z3d = -z3d

            # Project to fisheye
            r3d = math.sqrt(x3d * x3d + y3d * y3d)
            if r3d > 0 and abs(z3d) > 1e-6:
                theta = math.atan2(r3d, abs(z3d))
                rho = radius * theta / (math.pi / 2)

                fisheye_x = center_x + rho * (x3d / r3d)
                fisheye_y = center_y + rho * (y3d / r3d)
                final_x = fisheye_x + offset_x
                final_y = fisheye_y + offset_y

                if 0 <= final_x < input_width and 0 <= final_y < input_height:
                    map_x[y, x] = final_x
                    map_y[y, x] = final_y

    return map_x, map_y, (output_width, output_height)


def remap_frame(frame, map, out_size, interpolation=cv2.INTER_LINEAR, antialias="auto"):
    assert antialias in ["auto", "none", "gaussian", 2, 4, 8, 16]

    # Apply smoothing before downsampling if output size is smaller than input
    if antialias in ["auto", "gaussian"] and (
        out_size[0] < frame.shape[1] or out_size[1] < frame.shape[0]
    ):
        # Use Gaussian blur with kernel size proportional to downsampling factor
        scale_x = frame.shape[1] / out_size[0]
        scale_y = frame.shape[0] / out_size[1]
        ksize_x = max(1, int(2 * round(scale_x) + 1))
        ksize_y = max(1, int(2 * round(scale_y) + 1))
        # Ensure kernel size is odd and at least 3
        ksize_x = ksize_x if ksize_x % 2 == 1 else ksize_x + 1
        ksize_y = ksize_y if ksize_y % 2 == 1 else ksize_y + 1
        smoothed_frame = cv2.GaussianBlur(frame, (ksize_x, ksize_y), 0)
    elif isinstance(antialias, int) and (
        out_size[0] < frame.shape[1] or out_size[1] < frame.shape[0]
    ):
        # Upscale the map and output size by smoothing factor if specified
        upscale = antialias
        up_out_size = (out_size[0] * upscale, out_size[1] * upscale)
        up_map_x = cv2.resize(map[0], up_out_size, interpolation=cv2.INTER_LINEAR)
        up_map_y = cv2.resize(map[1], up_out_size, interpolation=cv2.INTER_LINEAR)

        # Remap at higher resolution
        upsampled = cv2.remap(
            frame,
            up_map_x,
            up_map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        # Downsample to target size
        return cv2.resize(upsampled, out_size, interpolation=interpolation)
    else:
        smoothed_frame = frame
    return cv2.remap(
        smoothed_frame, *map, interpolation=interpolation, borderMode=cv2.BORDER_CONSTANT
    )


def iterate_video_frames(
    cap, *, start_time=0, end_time=None, frames=None, pbar_f=partial(tqdm, desc="Processing frames")
):
    """
    Generator that yields (frame_index, frame) from a cv2.VideoCapture object.
    Handles seeking, start/end time, and non-contiguous frame indices.
    Optionally wraps the frame index iterator with a progress bar or other factory.
    Supports frames as None, list/array, or a slice object.
    """
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(start_time * fps)
    stop_frame = int(end_time * fps) if end_time is not None else total_frames
    if stop_frame > total_frames:
        stop_frame = total_frames

    if start_frame >= stop_frame:
        return

    if frames is None:
        frames_iter = range(start_frame, stop_frame)
    elif isinstance(frames, slice):
        start, stop, step = frames.indices(stop_frame - start_frame)
        frames_iter = range(start + start_frame, stop + start_frame, step)
    else:
        frames = sorted(frames)
        if frames[-1] >= stop_frame:
            raise ValueError(f"Frame index {frames[-1]} exceeds final frame ({stop_frame-1}).")
        frames_iter = frames

    if pbar_f is not None:
        frames_iter = pbar_f(frames_iter)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    prev_frame_index = start_frame - 1
    for frame_index in frames_iter:
        if frame_index != prev_frame_index + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        prev_frame_index = frame_index
        ret, frame = cap.read()
        if not ret:
            break
        yield frame_index, frame


def get_fisheye_to_perspective_converter(
    input_size,
    *,
    fov_h=120,
    fov_v=None,
    output_size=None,
    output_aspect_ratio=None,
    fisheye_radius_factor=0.95,
    mode="perspective",
):
    """
    Process Insta360 fisheye video to perspective or equirectangular.

    Args:
        input_path: Path to input video
        output_path: Path to output video
        fov_h, fov_v: Horizontal and vertical field of view in degrees (perspective mode)
        output_aspect_ratio: Output aspect ratio (None = 16:9 for single, 2:1 for dual)
        fisheye_center_x, fisheye_center_y: Fisheye center (None = image center)
        fisheye_radius_factor: Fisheye radius factor
        start_time: Start time in seconds (None = from beginning)
        end_time: End time in seconds (None = till end)
        mode: 'perspective' (default) or 'equirectangular'
        front_fisheye_position: For dual fisheye equirectangular mode
    """
    map_kwargs = dict(input_size=input_size, fisheye_radius_factor=fisheye_radius_factor)

    if mode == "equirectangular":
        map, output_size = create_dual_fisheye_to_equirectangular_map(**map_kwargs)
        print(f"Dual fisheye to equirectangular: {output_size[0]}x{output_size[1]}")
    else:
        if fov_v is None:
            fov_v = (
                get_fov_v(output_aspect_ratio, fov_h)
                if output_aspect_ratio is not None
                else (
                    get_fov_v(output_size[0] / output_size[1], fov_h)
                    if output_size is not None
                    else fov_h
                )
            )
        map, output_size = create_fisheye_to_perspective_map(
            output_size=output_size, fov=(fov_h, fov_v), **map_kwargs
        )
        print(f"Fisheye to perspective: {output_size[0]}x{output_size[1]}, FOV: {fov_h}°x{fov_v}°")

    return partial(
        remap_frame, map=map, out_size=output_size, interpolation=cv2.INTER_AREA, antialias=8
    )


def convert_video(
    input_path: str,
    output_path: str,
    frame_converter_f: T.Callable = get_fisheye_to_perspective_converter,
    frame_iter_args: dict = dict(start_time=0, end_time=None, frames=None),
    video_writer_args: dict = None,
):
    """
    Convert a video using a frame converter function and save the result.

    Args:
        input_path: Path to input video file.
        output_path: Path to output video file.
        frame_converter_f: Function that returns a frame converter (should accept input_size as kwarg).
        frame_iter_args: Arguments for iterate_video_frames.
        video_writer_args: Optional dict for cv2.VideoWriter (e.g., fourcc, fps, frameSize).
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {input_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    input_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(f"Input: {input_size[0]}x{input_size[1]}, {fps} FPS")

    frame_iter = iterate_video_frames(cap, **frame_iter_args)
    convert_frame = frame_converter_f(input_size=input_size)

    out = None
    out_frame_size = None
    for frame_index, frame in frame_iter:
        out_frame = convert_frame(frame)
        if out is None:
            out_frame_size = (out_frame.shape[1], out_frame.shape[0])
            writer_args = dict(
                fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                fps=fps,
                frameSize=out_frame_size,
            )
            if video_writer_args:
                writer_args.update(video_writer_args)
            out = cv2.VideoWriter(output_path, **writer_args)
            print(f"Output: {out_frame_size[0]}x{out_frame_size[1]}, {fps} FPS")
        out.write(out_frame)
        png_path = (Path(output_path).with_suffix("")) / f"{frame_index:06d}.png"
        png_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(png_path), out_frame)
    cap.release()
    if out is not None:
        out.release()
        print(f"Conversion complete: {output_path}")
    else:
        print("No frames were written.")


# Example usage
if __name__ == "__main__":
    fisheye_radius_factor = 1.0

    # Single fisheye to perspective with custom aspect ratio and FOV
    convert_video(
        "input/20231223_unit1_58_59_836/VID_20241219_105926_00_145.insv",
        "output/20231223_unit1_58_59_836/VID_20241219_105926_00_145_perspective_hw.mp4",
        frame_converter_f=partial(
            get_fisheye_to_perspective_converter,
            fov_h=127,  # GoPro Hero 4 medium: 127°
            fov_v=None,
            output_size=(384, 255),
            fisheye_radius_factor=fisheye_radius_factor,
        ),
        frame_iter_args=dict(end_time=10),
    )

    # Single fisheye to perspective with custom aspect ratio and FOV
    convert_video(
        "input/20231223_unit1_58_59_836/VID_20241219_105926_00_145.insv",
        "output/20231223_unit1_58_59_836/VID_20241219_105926_00_145_perspective.mp4",
        frame_converter_f=partial(
            get_fisheye_to_perspective_converter,
            fov_h=90,  # GoPro Hero 4 medium: 127°
            fov_v=None,
            output_aspect_ratio=16 / 9,  # TODO
            fisheye_radius_factor=fisheye_radius_factor,
        ),
        frame_iter_args=dict(end_time=10),
    )

    # Dual fisheye to equirectangular 360°
    convert_video(
        "input/20231223_unit1_58_59_836/LRV_20241219_105926_01_145.lrv",
        "output/20231223_unit1_58_59_836/LRV_20241219_105926_01_145_equirectangular.mp4",
        frame_converter_f=partial(get_fisheye_to_perspective_converter, mode="equirectangular"),
    )
