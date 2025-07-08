import cv2
import numpy as np
import math
from tqdm import tqdm
import typing as T
from pathlib import Path
from functools import partial
import warnings

DEFAULT_FISHEYE_RADIUS_FACTOR = 0.94


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
    """Calculates the vertical field of view (FOV) given an aspect ratio and horizontal FOV.

    Args:
        aspect_ratio (float): Aspect ratio (width / height).
        fov_h (float): Horizontal field of view in degrees.

    Returns:
        float: Vertical field of view in degrees.
    """
    return 2 * math.degrees(math.atan(math.tan(math.radians(fov_h / 2)) / aspect_ratio))


def get_axial_rotation_matrix(axis, angle):
    """Returns a rotation matrix for a given axis ('x', 'y', or 'z') and angle in degrees.

    Args:
        axis (str): Axis of rotation ('x', 'y', or 'z').
        angle (float): Angle in degrees.

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    angle_rad = math.radians(angle)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    else:
        raise ValueError("Axis must be one of 'x', 'y', or 'z'.")


def get_rotation_matrix(yaw, pitch, roll):
    """Returns a combined rotation matrix for yaw, pitch, and roll angles in degrees.
    The order of rotations is yaw (Y), pitch (X), roll (Z).

    Returns:
        np.ndarray: 3x3 combined rotation matrix.
    """
    Rz = get_axial_rotation_matrix("z", roll)  # Roll around Z-axis
    Rx = get_axial_rotation_matrix("x", pitch)  # Pitch around X-axis
    Ry = get_axial_rotation_matrix("y", yaw)  # Yaw around Y-axis
    return Rz @ Rx @ Ry  # Combined rotation matrix


def create_fisheye_to_perspective_map(
    input_size,
    output_size=None,
    fov=(120.0, 120.0),
    yaw_pitch=(0.0, 0.0),
    roll=0.0,
    fisheye_center=None,
    fisheye_radius_factor=DEFAULT_FISHEYE_RADIUS_FACTOR,
):
    """
    Creates mapping or remapping an equidi image to a perspective projection with arbitrary cammapping era orientation.

    Args:
        input_size (tuple): (width, height) of the input fisheye image.
        output_size (tuple, optional): (width, height) of the output perspective image. If None, it will be computed.
        fov (tuple): (horizontal_fov, vertical_fov) in degrees for the output perspective view.
        yaw_pitch (tuple): (yaw, pitch) in degrees. Yaw: left/right, Pitch: up/down.
        roll (float): Roll angle (twist) in degrees.
        fisheye_center (tuple, optional): (x, y) coordinates of the fisheye circle center in the input image. If None, the image center is used.
        fisheye_radius_factor (float): Fraction of min(center_x, center_y) to use as the fisheye radius.

    Returns:
        tuple: (map_x, map_y), output_size
            map_x (np.ndarray): X coordinates for remapping.
            map_y (np.ndarray): Y coordinates for remapping.
            output_size (tuple): (width, height) of the output image.
    """
    input_width, input_height = input_size
    fov_h, fov_v = fov

    if output_size is None:
        output_size = get_perspective_output_size(input_size, fov, fisheye_radius_factor)
    output_width, output_height = output_size

    if fisheye_center is None:
        fisheye_center = (input_width / 2, input_height / 2)

    # Fisheye projection parameters
    fisheye_cx, fisheye_cy = fisheye_center
    fisheye_radius = min(fisheye_cx, fisheye_cy) * fisheye_radius_factor

    # Convert FOV to focal length equivalent for perspective projection
    focal_length_x = output_width / (2 * math.tan(math.radians(fov_h / 2)))
    focal_length_y = output_height / (2 * math.tan(math.radians(fov_v / 2)))
    if not np.isclose(focal_length_x, focal_length_y, rtol=2.0 / min(output_size)):
        warnings.warn("The output size does not match the aspect ratio of the FOV.")

    # Perspective projection center
    perspective_cx = output_width / 2
    perspective_cy = output_height / 2

    # Output image coordinates normalized to [-1 .. 1]
    y, x = np.indices((output_height, output_width), dtype=np.float32)
    x_norm = (x - perspective_cx) / focal_length_x
    y_norm = (y - perspective_cy) / focal_length_y

    # 3D unit vectors pitining at each pixel
    rays = np.stack([x_norm, y_norm, np.ones_like(x_norm)], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)

    # Rotate rays
    R = get_rotation_matrix(*yaw_pitch, roll)
    rays_rot = np.einsum("...j,ij->...i", rays, R)  # shape: (H, W, 3)
    ray_x_rot, ray_y_rot, ray_z_rot = rays_rot[..., 0], rays_rot[..., 1], rays_rot[..., 2]

    # Angle from optical axis (Z-axis)
    theta = np.arccos(np.clip(ray_z_rot, -1, 1))  # shape: (H, W, 3)
    # Azimuth
    phi = np.arctan2(ray_y_rot, ray_x_rot)

    # Convert to fisheye image coordinates
    r = fisheye_radius / (np.pi / 2) * theta  # equidistant fisheye: proportional to the angle
    fisheye_x = fisheye_cx + r * np.cos(phi)
    fisheye_y = fisheye_cy + r * np.sin(phi)

    # Only map rays in front hemisphere and within input bounds
    valid = (theta > 0) & (theta < np.pi / 2)
    valid &= (
        (fisheye_x >= 0) & (fisheye_x < input_width) & (fisheye_y >= 0) & (fisheye_y < input_height)
    )

    map_x, map_y = [np.zeros((output_height, output_width), dtype=np.float32) for _ in range(2)]
    map_x[valid] = fisheye_x[valid].astype(np.float32)
    map_y[valid] = fisheye_y[valid].astype(np.float32)

    return (map_x, map_y), output_size


def create_dual_fisheye_to_equirectangular_map(
    input_width,
    input_height,
    output_aspect_ratio=2.0,
    fisheye_radius_factor=DEFAULT_FISHEYE_RADIUS_FACTOR,
    front_fisheye_position="left",
):
    """Creates mapping tables for dual fisheye to equirectangular projection.

    Args:
        input_width (int): Input image width.
        input_height (int): Input image height.
        output_aspect_ratio (float): Output aspect ratio (2.0 = standard equirectangular).
        fisheye_radius_factor (float): Fisheye radius as fraction of available space.
        front_fisheye_position (str): 'right'/'left' for side_by_side, 'top'/'bottom' for top_bottom.

    Returns:
        tuple: (map_x, map_y, output_size)
            map_x (np.ndarray): X coordinates for remapping.
            map_y (np.ndarray): Y coordinates for remapping.
            output_size (tuple): (width, height) of the output image.
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
    """Remaps a frame using the provided mapping tables, with optional antialiasing.

    Args:
        frame (np.ndarray): Input image/frame.
        map (tuple): (map_x, map_y) mapping tables.
        out_size (tuple): (width, height) of the output image.
        interpolation (int): OpenCV interpolation flag.
        antialias (str|int): Antialiasing mode ('auto', 'none', 'gaussian', or integer upsampling factor).

    Returns:
        np.ndarray: Remapped output image.
    """
    assert antialias in ["auto", "none", "gaussian", 2, 4, 8, 16]

    # Apply smoothing before downsampling if output size is smaller than input
    if antialias in ["auto", "gaussian"] and (
        out_size[0] < frame.shape[1] or out_size[1] < frame.shape[0]
    ):
        import scipy.ndimage

        scale_x = frame.shape[1] / out_size[0]
        scale_y = frame.shape[0] / out_size[1]
        sigma = (max(0.01, scale_x / 2.0), max(0.01, scale_y / 2.0))
        smoothed_frame = scipy.ndimage.gaussian_filter(
            frame, sigma=sigma + ((0,) if frame.ndim == 3 else ())
        )
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


def iterate_video_frames(cap, *, start_time=0, end_time=None, frames=None, pbar_f=None):
    """Generator that yields (frame_index, frame) from a cv2.VideoCapture object.
    Handles seeking, start/end time, and non-contiguous frame indices.
    Optionally wraps the frame index iterator with a progress bar or other factory.
    Supports frames as None, list/array, or a slice object.

    Args:
        cap (cv2.VideoCapture): OpenCV video capture object.
        start_time (float): Start time in seconds.
        end_time (float, optional): End time in seconds.
        frames (list|slice|None): Frame indices or slice to yield from start time.
        pbar_f (callable): Progress bar factory or None.

    Yields:
        tuple: (frame_index (int), frame (np.ndarray))
    """
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(start_time * fps)
    stop_frame = int(end_time * fps) if end_time is not None else total_frames
    stop_frame = min(max(stop_frame, start_frame), total_frames)

    if frames is None:
        frames_iter = range(start_frame, stop_frame)
    elif isinstance(frames, slice):
        start, stop, step = frames.indices(stop_frame - start_frame)
        frames_iter = range(start + start_frame, stop + start_frame, step)
    else:
        frames_iter = sorted(frames)
        if frames[-1] >= stop_frame:
            raise ValueError(f"Frame index {frames_iter[-1]} exceeds final frame ({stop_frame-1}).")

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
    fov_h=120.0,
    fov_v=None,
    yaw_pitch=(0.0, 0.0),
    roll=0.0,
    output_size=None,
    output_aspect_ratio=None,
    fisheye_radius_factor=DEFAULT_FISHEYE_RADIUS_FACTOR,
    mode="perspective",
):
    """Returns a frame converter function for Insta360 fisheye video to perspective or equirectangular.

    Args:
        input_size (tuple): (width, height) of the input image.
        fov_h (float): Horizontal field of view in degrees (perspective mode).
        fov_v (float, optional): Vertical field of view in degrees (perspective mode).
        yaw_pitch (tuple): (yaw, pitch) in degrees.
        roll (float): Roll angle (twist) in degrees.
        output_size (tuple, optional): (width, height) of the output image.
        output_aspect_ratio (float, optional): Output aspect ratio.
        fisheye_radius_factor (float): Fisheye radius factor.
        mode (str): 'perspective' (default) or 'equirectangular'.

    Returns:
        Callable: Function that converts a frame (np.ndarray) to the desired projection.
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
            output_size=output_size,
            fov=(fov_h, fov_v),
            yaw_pitch=yaw_pitch,
            roll=roll,
            **map_kwargs,
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
    """Converts a video using a frame converter function and save the result.

    Args:
        input_path (str): Path to input video file.
        output_path (str): Path to output video file.
        frame_converter_f (Callable): Function that returns a frame converter (should accept input_size as kwarg).
        frame_iter_args (dict): Arguments for iterate_video_frames.
        video_writer_args (dict, optional): Optional dict for cv2.VideoWriter (e.g., fourcc, fps, frameSize).

    Returns:
        None
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {input_path}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    input_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(f"Input: {input_size[0]}x{input_size[1]}, {fps} FPS")

    frame_iter = iterate_video_frames(
        cap, **frame_iter_args, pbar_f=partial(tqdm, desc="Processing frames")
    )
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
        # png_path = (Path(output_path).with_suffix("")) / f"{frame_index:07d}.png"
        # png_path.parent.mkdir(parents=True, exist_ok=True)
        # cv2.imwrite(str(png_path), out_frame)
    cap.release()
    if out is not None:
        out.release()
        print(f"Conversion complete: {output_path}")
    else:
        print("No frames were written.")


# Example usage
if __name__ == "__main__":
    fisheye_radius_factor = 0.94

    # Single fisheye to perspective with custom aspect ratio and FOV
    convert_video(
        "input/20231223_unit1_58_59_836/VID_20241219_105926_00_145.insv",
        "output/20231223_unit1_58_59_836/VID_20241219_105926_00_145_perspective_hw.mp4",
        frame_converter_f=partial(
            get_fisheye_to_perspective_converter,
            fov_h=127,  # GoPro Hero 4 medium: 127°
            fov_v=None,
            yaw_pitch=(0, 0),
            roll=0.0,
            output_size=(384, 288),
            fisheye_radius_factor=fisheye_radius_factor,
        ),
        frame_iter_args=dict(start_time=80, end_time=150),
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
