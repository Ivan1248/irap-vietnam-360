## Context / notes

### 2026-03-11 video-gpx-cutting quick audit

- `video-gpx-cutting/gpx_io.py` imports `haversine_distance` but `video-gpx-cutting/geo_math.py` defines `haversine_distance_m`. This looks like a refactor mismatch and will raise `ImportError` if the GPX parser path is executed. (p≈0.95)
- `video-gpx-cutting/video_meta.py:to_video_local_times()` refers to `track.elev_m` / `track.speed_mps`, but `Track` currently defines `elev` / `speed` fields. This will raise `AttributeError` when used. (p≈0.95)
- The GPX-based cutting currently uses only `times/lat/lon` for reentry detection; speed/elevation fields and `analyze_trajectory()` features appear unused and may be removable to simplify. (p≈0.8)

