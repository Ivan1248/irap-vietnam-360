import os
import shutil
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Dict, List

from cut_manual import _build_stem_index
from ffmpeg_cut import ffmpeg_cut_segments
from gpx_cut import cut_gpx_segments
from gpx_io import parse_gpx
from manual_cuts import cuts_to_segments, parse_pasted_table
from video_keyframes import get_keyframe_timestamps, snap_to_previous_keyframe
from video_meta import compute_time_alignment, extract_video_meta


class ManualCutsApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Manual Video Cuts")

        self.input_dir_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.clear_output_var = tk.BooleanVar(value=False)

        self._build_ui()

    def _build_ui(self) -> None:
        top = tk.Frame(self)
        top.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Input/output directory selectors
        dir_frame = tk.Frame(top)
        dir_frame.pack(fill=tk.X, pady=(0, 8))

        tk.Label(dir_frame, text="Input directory:").grid(row=0, column=0, sticky="w")
        tk.Entry(dir_frame, textvariable=self.input_dir_var, width=40).grid(row=0, column=1, sticky="we", padx=(4, 4))
        tk.Button(dir_frame, text="Browse", command=self._choose_input_dir).grid(row=0, column=2, padx=(4, 0))

        tk.Label(dir_frame, text="Output directory:").grid(row=1, column=0, sticky="w")
        tk.Entry(dir_frame, textvariable=self.output_dir_var, width=40).grid(row=1, column=1, sticky="we", padx=(4, 4))
        tk.Button(dir_frame, text="Browse", command=self._choose_output_dir).grid(row=1, column=2, padx=(4, 0))

        tk.Checkbutton(
            dir_frame,
            text="Clear output directory before processing",
            variable=self.clear_output_var,
        ).grid(row=2, column=1, sticky="w", padx=(4, 4), pady=(2, 0))

        dir_frame.columnconfigure(1, weight=1)

        # Text area for pasted table
        tk.Label(top, text="Pasted cuts (videoStem whitespace mm:ss ...):").pack(anchor="w")
        self.text = scrolledtext.ScrolledText(top, height=12)
        self.text.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        # Buttons and log area
        btn_frame = tk.Frame(top)
        btn_frame.pack(fill=tk.X, pady=(0, 4))

        tk.Button(btn_frame, text="Preview", command=self._preview).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Run", command=self._run).pack(side=tk.LEFT, padx=(4, 0))

        self.log = scrolledtext.ScrolledText(top, height=10, state=tk.DISABLED)
        self.log.pack(fill=tk.BOTH, expand=True, pady=(0, 4))

        self.progress = ttk.Progressbar(top, orient=tk.HORIZONTAL, mode="determinate")
        self.progress.pack(fill=tk.X)

    def _choose_input_dir(self) -> None:
        d = filedialog.askdirectory(title="Select input directory")
        if d:
            self.input_dir_var.set(d)

    def _choose_output_dir(self) -> None:
        d = filedialog.askdirectory(title="Select output directory")
        if d:
            self.output_dir_var.set(d)

    def _append_log(self, line: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, line + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _parse_mapping(self) -> Dict[str, List[float]]:
        text = self.text.get("1.0", tk.END)
        return parse_pasted_table(text)

    def _preview(self) -> None:
        try:
            mapping = self._parse_mapping()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Parse error", str(e))
            return

        self.log.configure(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.configure(state=tk.DISABLED)

        if not mapping:
            self._append_log("No valid rows found.")
            return

        self._append_log(f"Parsed {len(mapping)} video stems:")
        for stem, cuts in mapping.items():
            cuts_str = ", ".join(f"{c:.1f}s" for c in cuts)
            self._append_log(f"  {stem}: {cuts_str}")

    def _run(self) -> None:
        input_dir = self.input_dir_var.get().strip()
        output_dir = self.output_dir_var.get().strip()
        if not input_dir or not os.path.isdir(input_dir):
            messagebox.showerror("Error", "Please select a valid input directory.")
            return
        if not output_dir:
            messagebox.showerror("Error", "Please select an output directory.")
            return

        try:
            mapping = self._parse_mapping()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Parse error", str(e))
            return

        if not mapping:
            messagebox.showinfo("Nothing to do", "No valid rows found in pasted text.")
            return

        clear_output = self.clear_output_var.get()
        if clear_output and os.path.isdir(output_dir):
            if not messagebox.askyesno(
                "Confirm clear",
                f"Delete all contents of:\n{output_dir}\n\nThis cannot be undone.",
            ):
                return

        self._append_log("Starting batch cutting...")

        thread = threading.Thread(
            target=self._run_worker,
            args=(input_dir, output_dir, mapping, clear_output),
            daemon=True,
        )
        thread.start()

    def _run_worker(
        self,
        input_dir: str,
        output_dir: str,
        mapping: Dict[str, List[float]],
        clear_output: bool,
    ) -> None:
        if clear_output and os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
            self.after(0, lambda: self._append_log(f"Cleared output directory: {output_dir}"))

        try:
            stem_index = _build_stem_index(input_dir)
        except Exception as e:  # noqa: BLE001
            self.after(0, lambda err=e: self._append_log(f"Error building index: {err}"))
            return

        total = len(mapping)
        self.after(0, lambda t=total: self.progress.configure(maximum=t, value=0))

        for idx, (stem, cuts) in enumerate(mapping.items(), start=1):
            video_path = stem_index.get(stem)
            if video_path is None:
                self.after(0, lambda s=stem: self._append_log(f"No file found for stem {s!r}."))
                continue

            try:
                meta = extract_video_meta(video_path)
            except Exception as e:  # noqa: BLE001
                self.after(
                    0,
                    lambda s=stem, err=e: self._append_log(f"ffprobe failed for {s!r}: {err}"),
                )
                continue

            segments = cuts_to_segments(
                cut_times_s=cuts,
                duration_s=meta.duration_s,
            )

            if not segments:
                self.after(
                    0,
                    lambda s=stem: self._append_log(f"No segments for {s!r} (duration {meta.duration_s:.3f}s)."),
                )
                continue

            seg_summary = ", ".join(f"[{seg.start_s:.1f}s–{seg.end_s:.1f}s]" for seg in segments)
            self.after(
                0,
                lambda s=stem, n=len(segments), dur=meta.duration_s, ss=seg_summary: self._append_log(
                    f"{s}: {dur:.1f}s → {ss}"
                ),
            )

            # SNAPPING: Detect keyframes and snap starts
            try:
                keyframes = get_keyframe_timestamps(video_path)
                snapped_starts = [snap_to_previous_keyframe(seg.start_s, keyframes) for seg in segments]
                snap_deltas = [seg.start_s - snapped for seg, snapped in zip(segments, snapped_starts)]
                if any(d > 0.001 for d in snap_deltas):
                    delta_str = ", ".join(f"{d:.3f}s" for d in snap_deltas)
                    self.after(
                        0,
                        lambda s=stem, ds=delta_str: self._append_log(f"  Keyframe snap deltas: {ds}"),
                    )
            except Exception as e:  # noqa: BLE001
                self.after(
                    0,
                    lambda s=stem, err=e: self._append_log(
                        f"  Keyframe detection failed for {s!r}: {err}. Falling back."
                    ),
                )
                snapped_starts = [None] * len(segments)

            # Determine output directory based on relative path from input_dir
            rel_path = os.path.relpath(video_path, input_dir)
            rel_dir = os.path.dirname(rel_path)
            # Final output dir for this video's segments
            video_out_dir = os.path.join(output_dir, rel_dir)

            try:
                ffmpeg_cut_segments(
                    video_path=video_path,
                    segments=segments,
                    output_dir=video_out_dir,
                    base_name=stem,
                    original_creation_time_s_epoch=meta.creation_time_s_epoch,
                    snapped_starts_s=snapped_starts,
                )
            except Exception as e:  # noqa: BLE001
                self.after(
                    0,
                    lambda s=stem, err=e: self._append_log(f"  ffmpeg failed for {s!r}: {err}"),
                )
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
                        output_dir=video_out_dir,
                        base_name=stem,
                        snapped_starts_s=snapped_starts,
                    )
                except Exception as e:  # noqa: BLE001
                    self.after(
                        0,
                        lambda s=stem, err=e: self._append_log(f"  GPX cut failed for {s!r}: {err}"),
                    )
                    continue
            else:
                self.after(0, lambda s=stem: self._append_log(f"  No GPX found for {s!r}."))

            self.after(
                0,
                lambda s=stem, v=idx: (self._append_log(f"  Done: {s}"), self.progress.configure(value=v)),
            )

        self.after(0, lambda: self._append_log("Batch cutting completed."))


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(a in ("-h", "--help") for a in argv):
        print("Usage: python video-gpx-cutting/manual_cuts_gui.py")
        return 0
    app = ManualCutsApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
