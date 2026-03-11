import argparse
import os
import sys
from typing import Dict, List

from ffmpeg_cut import ffmpeg_cut_segments
from gpx_cut import cut_gpx_segments
from gpx_io import parse_gpx
from manual_cuts import cuts_to_segments, parse_pasted_table
from video_keyframes import get_keyframe_timestamps, snap_to_previous_keyframe
from video_meta import compute_time_alignment, extract_video_meta


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=("Cut videos into segments using manually provided cut timestamps."))
    p.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing the input video files.",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory where cut video segments will be written.",
    )
    p.add_argument(
        "--list-file",
        help="Path to a text file with lines: videoStem <whitespace> mm:ss [mm:ss ...]. "
        "If omitted, the list is read from stdin.",
    )
    p.add_argument(
        "--min-segment-duration-s",
        type=float,
        default=5.0,
        help="Minimum segment duration in seconds (default: 5).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print segments without running ffmpeg.",
    )
    return p.parse_args(argv)


def _build_stem_index(input_dir: str) -> Dict[str, str]:
    VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".insv"}
    index: Dict[str, str] = {}
    for root, _, files in os.walk(input_dir):
        for name in files:
            path = os.path.join(root, name)
            stem, ext = os.path.splitext(name)
            ext = ext.lower()

            if stem not in index:
                index[stem] = path
                continue

            # Conflict resolution
            existing_path = index[stem]
            _existing_stem, existing_ext = os.path.splitext(existing_path)
            existing_ext = existing_ext.lower()

            new_is_video = ext in VIDEO_EXTS
            old_is_video = existing_ext in VIDEO_EXTS

            if new_is_video and not old_is_video:
                # Prefer the new video file over the old non-video file
                index[stem] = path
            elif not new_is_video and old_is_video:
                # Keep the old video file
                continue
            else:
                # Both are video or both are non-video
                # If they are the same file (same path), ignore
                if os.path.abspath(path) == os.path.abspath(existing_path):
                    continue
                raise RuntimeError(
                    f"Multiple files with stem {stem!r} in {input_dir!r}; cannot disambiguate between {existing_path} and {path}."
                )
    return index


def main(argv: List[str]) -> int:
    args = _parse_args(argv)

    if args.list_file:
        with open(args.list_file, "r", encoding="utf-8") as f:
            table_text = f.read()
    else:
        table_text = sys.stdin.read()

    mapping = parse_pasted_table(table_text)
    if not mapping:
        print("No valid rows found in manual cut list.", file=sys.stderr)
        return 1

    try:
        stem_index = _build_stem_index(args.input_dir)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1

    from tqdm import tqdm

    had_error = False

    for stem, cuts in tqdm(mapping.items(), desc="Cutting videos", unit="video"):
        video_path = stem_index.get(stem)
        if video_path is None:
            print(f"No file found for stem {stem!r} in {args.input_dir!r}", file=sys.stderr)
            had_error = True
            continue

        try:
            meta = extract_video_meta(video_path)
        except Exception as e:  # noqa: BLE001
            print(f"ffprobe failed for {video_path!r}: {e}", file=sys.stderr)
            had_error = True
            continue

        segments = cuts_to_segments(
            cut_times_s=cuts,
            duration_s=meta.duration_s,
        )

        if not segments:
            print(f"No segments for {stem!r} (video duration {meta.duration_s:.3f}s)", file=sys.stderr)
            continue

        if args.dry_run:
            for seg in segments:
                print(
                    f"{stem}\t{seg.start_s:.3f}\t{seg.end_s:.3f}",
                    file=sys.stdout,
                )
            continue

        # SNAPPING: Detect keyframes and snap starts
        try:
            keyframes = get_keyframe_timestamps(video_path)
            snapped_starts = [snap_to_previous_keyframe(seg.start_s, keyframes) for seg in segments]
        except Exception as e:  # noqa: BLE001
            print(f"Keyframe detection failed for {stem!r}: {e}. Falling back to requested times.", file=sys.stderr)
            snapped_starts = [None] * len(segments)

        # Determine output directory based on relative path from input_dir
        rel_path = os.path.relpath(video_path, args.input_dir)
        rel_dir = os.path.dirname(rel_path)
        out_dir = os.path.join(args.output_dir, rel_dir)

        try:
            ffmpeg_cut_segments(
                video_path=video_path,
                segments=segments,
                output_dir=out_dir,
                base_name=stem,
                original_creation_time_s_epoch=meta.creation_time_s_epoch,
                snapped_starts_s=snapped_starts,
            )
        except Exception as e:  # noqa: BLE001
            print(f"ffmpeg failed for {stem!r}: {e}", file=sys.stderr)
            had_error = True
            continue

        # Look for matching GPX
        gpx_path = os.path.splitext(video_path)[0] + ".gpx"
        if os.path.isfile(gpx_path):
            try:
                alignment = compute_time_alignment(parse_gpx(gpx_path), meta)
                cut_gpx_segments(
                    gpx_path=gpx_path,
                    segments=segments,
                    alignment=alignment,
                    output_dir=out_dir,
                    base_name=stem,
                    snapped_starts_s=snapped_starts,
                )
            except Exception as e:  # noqa: BLE001
                print(f"GPX cut failed for {stem!r}: {e}", file=sys.stderr)
                had_error = True

    return 1 if had_error else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
