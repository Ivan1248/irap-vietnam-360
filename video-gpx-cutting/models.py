from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Segment:
    start_s: float
    end_s: float


def group_events_to_segments(
    event_times: List[float],
    duration_s: float,
    merge_overlaps: bool = True,
) -> List[Segment]:
    """
    Transform cut times into non-overlapping [start, end) segments, with optional padding.
    """
    if duration_s <= 0.0:
        return []

    min_segment_duration_s = 0.0
    sorted_times = sorted(t for t in event_times if 0.0 <= t <= duration_s)
    segments: List[Segment] = []

    current_start = 0.0
    for t_cut in sorted_times:
        if t_cut - current_start >= min_segment_duration_s:
            segments.append(Segment(start_s=current_start, end_s=t_cut))
            current_start = t_cut

    if duration_s - current_start >= min_segment_duration_s:
        segments.append(Segment(start_s=current_start, end_s=duration_s))

    if not segments:
        return []

    # Apply padding and optionally merge overlaps
    padded: List[Segment] = []
    for seg in segments:
        start = max(0.0, seg.start_s)
        end = min(duration_s, seg.end_s)
        if not padded:
            padded.append(Segment(start_s=start, end_s=end))
        else:
            last = padded[-1]
            if merge_overlaps and start <= last.end_s:
                merged = Segment(start_s=last.start_s, end_s=max(last.end_s, end))
                padded[-1] = merged
            else:
                padded.append(Segment(start_s=start, end_s=end))

    return padded
