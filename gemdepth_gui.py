"""
GemDepth GUI - single / batch video depth processing.

Outputs one grayscale depth video per input video. The output frame rate
matches the input exactly, including irregular NTSC rates such as 24000/1001.

Run from the repository root (so ./checkpoint/gemdepth.pth and the model
package resolve correctly):

    python gemdepth_gui.py
"""

import os
import sys
import glob
import json
import queue
import threading
import traceback
from fractions import Fraction

import numpy as np
import cv2

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Make sure the repo root is importable regardless of where we're launched from.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Persisted GUI settings live next to the script.
SETTINGS_PATH = os.path.join(ROOT_DIR, "gui_settings.json")


# --------------------------------------------------------------------------- #
# FPS handling
# --------------------------------------------------------------------------- #
def resolve_fps_fraction(float_fps, video_path=None):
    """Recover an exact frame-rate fraction from a (possibly irregular) source.

    decord reports fps as a float, so 24000/1001 arrives as 23.976023976...
    Fraction(...).limit_denominator(1001) snaps that cleanly back to the exact
    NTSC rational while leaving true integer rates untouched. If an ffprobe
    binary is on PATH we prefer the container's reported r_frame_rate, which is
    the most authoritative source.
    """
    probed = _probe_fps_fraction(video_path) if video_path else None
    if probed is not None:
        return probed

    if not float_fps or float_fps <= 0:
        return Fraction(30, 1)  # sane fallback
    # limit_denominator(1001) recovers 24000/1001, 30000/1001, 60000/1001, etc.
    return Fraction(float_fps).limit_denominator(1001)


def _probe_fps_fraction(video_path):
    """Return an exact Fraction from ffprobe's r_frame_rate, or None."""
    import shutil
    import subprocess

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    try:
        out = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "default=nokey=1:noprint_wrappers=1",
                video_path,
            ],
            capture_output=True, text=True, timeout=15,
        )
        token = out.stdout.strip()
        if not token or token == "0/0":
            return None
        num, _, den = token.partition("/")
        if den:
            return Fraction(int(num), int(den))
        return Fraction(int(num), 1)
    except Exception:
        return None


def fps_to_str(frac):
    if frac.denominator == 1:
        return str(frac.numerator)
    return f"{frac.numerator}/{frac.denominator} ({float(frac):.3f})"


# --------------------------------------------------------------------------- #
# Memory presets (from README) -- patched onto model.gemdepth before inference.
# --------------------------------------------------------------------------- #
MEMORY_PRESETS = {
    "High (~44GB)": dict(
        INFER_LEN=32, OVERLAP=10,
        KEYFRAMES=[0, 12, 24, 25, 26, 27, 28, 29, 30, 31], INTERP_LEN=8,
    ),
    "Medium (~25GB)": dict(
        INFER_LEN=16, OVERLAP=6,
        KEYFRAMES=[0, 6, 12, 13, 14, 15], INTERP_LEN=4,
    ),
    "Low (~15GB)": dict(
        INFER_LEN=8, OVERLAP=4,
        KEYFRAMES=[0, 3, 6, 7], INTERP_LEN=2,
    ),
}

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


# --------------------------------------------------------------------------- #
# Model / inference (no tkinter in here -- reusable & testable)
# --------------------------------------------------------------------------- #
class ModelManager:
    """Loads the GemDepth checkpoint once and caches it on the device."""

    def __init__(self):
        self._model = None
        self._encoder = None
        self._ckpt = None
        self.device = None

    def ensure_loaded(self, encoder, checkpoint_path, log):
        import torch  # lazy: keeps GUI startup fast and import errors visible

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if (
            self._model is not None
            and self._encoder == encoder
            and self._ckpt == checkpoint_path
        ):
            return self._model

        from model.gemdepth import GemDepth

        if encoder not in MODEL_CONFIGS:
            raise ValueError(f"Unknown encoder: {encoder}")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        log(f"Loading {encoder} model on {self.device} ...")
        model = GemDepth(**MODEL_CONFIGS[encoder])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint, strict=True)
        model = model.to(self.device).eval()

        self._model = model
        self._encoder = encoder
        self._ckpt = checkpoint_path
        log("Model ready.")
        return model


def apply_memory_preset(preset_name, log):
    """Patch the inference window constants on the model module."""
    import model.gemdepth as gd

    preset = MEMORY_PRESETS[preset_name]
    for key, value in preset.items():
        setattr(gd, key, value)
    log(
        f"Memory preset '{preset_name}': INFER_LEN={preset['INFER_LEN']}, "
        f"OVERLAP={preset['OVERLAP']}, INTERP_LEN={preset['INTERP_LEN']}"
    )


def write_grayscale_depth_video(depths, output_path, fps_fraction, log):
    """Write a grayscale depth mp4 at the exact source frame rate.

    Normalization: per-video global 2nd-98th percentile, so brightness is
    temporally stable and robust to single-frame outliers.
    """
    if depths is None or len(depths) == 0:
        raise ValueError("no depth frames to write")

    depths = np.asarray(depths)
    if depths.ndim == 4:  # [N, 1, H, W] -> [N, H, W]
        depths = depths[:, 0]
    n, h, w = depths.shape[0], depths.shape[-2], depths.shape[-1]

    d_min, d_max = np.percentile(depths, 2), np.percentile(depths, 98)
    if not np.isfinite(d_min) or not np.isfinite(d_max) or d_max <= d_min:
        d_min, d_max = float(np.min(depths)), float(np.max(depths))
        if d_max <= d_min:
            d_min, d_max = 0.0, 1.0
    log(f"  depth range (2-98%): {d_min:.4f} - {d_max:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # cv2's FFmpeg backend converts this double to a rational timebase
    # (av_d2q), reproducing e.g. 24000/1001 from 23.976.
    fps_float = float(fps_fraction)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps_float, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cv2 could not open writer for {output_path}")

    try:
        for i in range(n):
            depth = depths[i]
            norm = np.clip((depth - d_min) / (d_max - d_min + 1e-8), 0, 1)
            gray = (norm * 255).astype(np.uint8)
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            writer.write(bgr)
    finally:
        writer.release()


class VideoProcessor:
    """Decode -> infer -> write, for a single video. No resampling."""

    def __init__(self, model_manager):
        self.mm = model_manager

    def process(self, video_path, output_path, *, encoder, checkpoint, input_size,
                max_res, fp32, preset, log, fps_callback=None):
        from model.utils.dc_utils import read_video_frames

        apply_memory_preset(preset, log)
        model = self.mm.ensure_loaded(encoder, checkpoint, log)

        log(f"Reading frames from {os.path.basename(video_path)} ...")
        # target_fps=-1 and max_len=-1 -> keep every frame, no resampling.
        frames, src_fps = read_video_frames(video_path, -1, -1, max_res)
        if frames is None or len(frames) == 0:
            raise ValueError("no frames decoded (unreadable or empty video)")

        fps_fraction = resolve_fps_fraction(src_fps, video_path)
        log(f"  {len(frames)} frames @ {fps_to_str(fps_fraction)} fps")
        if fps_callback:
            fps_callback(fps_to_str(fps_fraction))

        depths, _ = model.infer_video_depth(
            frames, src_fps, input_size=input_size, device=self.mm.device, fp32=fp32
        )
        # infer returns exactly len(frames) depths -> 1:1, duration preserved.
        log(f"  inferred {len(depths)} depth frames")

        write_grayscale_depth_video(depths, output_path, fps_fraction, log)
        log(f"✓ saved: {output_path}")


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
VIDEO_EXTS = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.MP4", "*.MOV", "*.MKV", "*.AVI")

# Queue message types
MSG_LOG = "log"
MSG_STATUS = "status"      # (index, state)
MSG_FILEFPS = "filefps"    # (index, fps_str)
MSG_PROGRESS = "progress"  # (done, total)
MSG_DONE = "done"


class App:
    def __init__(self, root):
        self.root = root
        root.title("GemDepth - Video Depth GUI")
        root.geometry("900x680")
        root.minsize(760, 560)

        self.queue = queue.Queue()
        self.worker = None
        self.cancel_event = threading.Event()
        self.model_manager = ModelManager()

        self.files = []  # list of input video paths (order = processing order)

        self._build_widgets()
        self._load_settings()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._drain_queue)

    # -- settings persistence ---------------------------------------------- #
    def _setting_vars(self):
        return {
            "encoder": self.var_encoder,
            "input_size": self.var_input_size,
            "max_res": self.var_max_res,
            "fp32": self.var_fp32,
            "preset": self.var_preset,
            "overwrite": self.var_overwrite,
            "checkpoint": self.var_ckpt,
            "outdir": self.var_outdir,
        }

    def _load_settings(self):
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        for key, var in self._setting_vars().items():
            if key in data:
                try:
                    var.set(data[key])
                except Exception:
                    pass  # ignore malformed/incompatible stored value
        geom = data.get("geometry")
        if isinstance(geom, str) and geom:
            try:
                self.root.geometry(geom)
            except Exception:
                pass

    def _save_settings(self):
        data = {}
        for key, var in self._setting_vars().items():
            try:
                data[key] = var.get()
            except Exception:
                pass  # skip a var that can't be read (e.g. invalid spinbox text)
        try:
            data["geometry"] = self.root.geometry()
        except Exception:
            pass
        try:
            with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass

    def _on_close(self):
        self._save_settings()
        self.root.destroy()

    # -- layout ------------------------------------------------------------- #
    def _build_widgets(self):
        pad = dict(padx=6, pady=4)

        # File list
        top = ttk.LabelFrame(self.root, text="Input videos")
        top.pack(fill="both", expand=True, **pad)

        cols = ("file", "fps", "status")
        self.tree = ttk.Treeview(top, columns=cols, show="headings", height=8)
        self.tree.heading("file", text="File")
        self.tree.heading("fps", text="Detected FPS")
        self.tree.heading("status", text="Status")
        self.tree.column("file", width=460, anchor="w")
        self.tree.column("fps", width=150, anchor="center")
        self.tree.column("status", width=120, anchor="center")
        self.tree.pack(side="left", fill="both", expand=True, padx=6, pady=6)

        sb = ttk.Scrollbar(top, orient="vertical", command=self.tree.yview)
        sb.pack(side="left", fill="y")
        self.tree.configure(yscrollcommand=sb.set)

        btns = ttk.Frame(top)
        btns.pack(side="left", fill="y", padx=6)
        ttk.Button(btns, text="Add files…", command=self.add_files).pack(fill="x", pady=2)
        ttk.Button(btns, text="Add folder…", command=self.add_folder).pack(fill="x", pady=2)
        ttk.Button(btns, text="Remove", command=self.remove_selected).pack(fill="x", pady=2)
        ttk.Button(btns, text="Clear", command=self.clear_files).pack(fill="x", pady=2)
        ttk.Separator(btns, orient="horizontal").pack(fill="x", pady=6)
        ttk.Button(btns, text="Move up", command=lambda: self.move(-1)).pack(fill="x", pady=2)
        ttk.Button(btns, text="Move down", command=lambda: self.move(1)).pack(fill="x", pady=2)

        # Settings
        cfg = ttk.LabelFrame(self.root, text="Settings")
        cfg.pack(fill="x", **pad)

        self.var_encoder = tk.StringVar(value="vitl")
        self.var_input_size = tk.IntVar(value=518)
        self.var_max_res = tk.IntVar(value=1280)
        self.var_fp32 = tk.BooleanVar(value=False)
        self.var_preset = tk.StringVar(value="High (~44GB)")
        self.var_overwrite = tk.BooleanVar(value=False)
        self.var_ckpt = tk.StringVar(value=os.path.join(ROOT_DIR, "checkpoint", "gemdepth.pth"))
        self.var_outdir = tk.StringVar(value=os.path.join(ROOT_DIR, "output"))

        row1 = ttk.Frame(cfg); row1.pack(fill="x", padx=6, pady=3)
        ttk.Label(row1, text="Encoder").pack(side="left")
        ttk.Combobox(row1, textvariable=self.var_encoder, values=["vits", "vitb", "vitl"],
                     width=6, state="readonly").pack(side="left", padx=(4, 14))
        ttk.Label(row1, text="Input size").pack(side="left")
        ttk.Spinbox(row1, from_=140, to=1540, increment=14, width=6,
                    textvariable=self.var_input_size).pack(side="left", padx=(4, 14))
        ttk.Label(row1, text="Max res").pack(side="left")
        ttk.Spinbox(row1, from_=256, to=4096, increment=64, width=6,
                    textvariable=self.var_max_res).pack(side="left", padx=(4, 14))
        ttk.Checkbutton(row1, text="fp32", variable=self.var_fp32).pack(side="left", padx=6)

        row2 = ttk.Frame(cfg); row2.pack(fill="x", padx=6, pady=3)
        ttk.Label(row2, text="GPU memory").pack(side="left")
        ttk.Combobox(row2, textvariable=self.var_preset, values=list(MEMORY_PRESETS),
                     width=16, state="readonly").pack(side="left", padx=(4, 14))
        ttk.Checkbutton(row2, text="Overwrite existing outputs",
                        variable=self.var_overwrite).pack(side="left", padx=6)

        row3 = ttk.Frame(cfg); row3.pack(fill="x", padx=6, pady=3)
        ttk.Label(row3, text="Checkpoint").pack(side="left")
        ttk.Entry(row3, textvariable=self.var_ckpt).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(row3, text="…", width=3, command=self.pick_ckpt).pack(side="left")

        row4 = ttk.Frame(cfg); row4.pack(fill="x", padx=6, pady=3)
        ttk.Label(row4, text="Output dir").pack(side="left")
        ttk.Entry(row4, textvariable=self.var_outdir).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(row4, text="…", width=3, command=self.pick_outdir).pack(side="left")

        # Run controls
        run = ttk.Frame(self.root); run.pack(fill="x", **pad)
        self.btn_start = ttk.Button(run, text="Start", command=self.start)
        self.btn_start.pack(side="left", padx=4)
        self.btn_cancel = ttk.Button(run, text="Cancel", command=self.cancel, state="disabled")
        self.btn_cancel.pack(side="left", padx=4)
        self.progress = ttk.Progressbar(run, mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, padx=10)
        self.lbl_overall = ttk.Label(run, text="0 / 0")
        self.lbl_overall.pack(side="left", padx=4)

        # Log
        logf = ttk.LabelFrame(self.root, text="Log")
        logf.pack(fill="both", expand=True, **pad)
        self.log_text = tk.Text(logf, height=8, wrap="word", state="disabled")
        self.log_text.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        logsb = ttk.Scrollbar(logf, orient="vertical", command=self.log_text.yview)
        logsb.pack(side="left", fill="y")
        self.log_text.configure(yscrollcommand=logsb.set)

    # -- file list ops ------------------------------------------------------ #
    def _refresh_tree(self):
        self.tree.delete(*self.tree.get_children())
        for path in self.files:
            self.tree.insert("", "end", values=(path, "-", "queued"))

    def add_files(self):
        paths = filedialog.askopenfilenames(
            title="Select videos",
            filetypes=[("Videos", " ".join(VIDEO_EXTS)), ("All files", "*.*")],
        )
        for p in paths:
            if p not in self.files:
                self.files.append(p)
        self._refresh_tree()

    def add_folder(self):
        folder = filedialog.askdirectory(title="Select folder of videos")
        if not folder:
            return
        found = []
        for ext in VIDEO_EXTS:
            found.extend(glob.glob(os.path.join(folder, ext)))
        for p in sorted(set(found)):
            if p not in self.files:
                self.files.append(p)
        self._refresh_tree()

    def _selected_index(self):
        sel = self.tree.selection()
        if not sel:
            return None
        return self.tree.index(sel[0])

    def remove_selected(self):
        idx = self._selected_index()
        if idx is not None:
            del self.files[idx]
            self._refresh_tree()

    def clear_files(self):
        self.files.clear()
        self._refresh_tree()

    def move(self, delta):
        idx = self._selected_index()
        if idx is None:
            return
        new = idx + delta
        if 0 <= new < len(self.files):
            self.files[idx], self.files[new] = self.files[new], self.files[idx]
            self._refresh_tree()
            child = self.tree.get_children()[new]
            self.tree.selection_set(child)

    def pick_ckpt(self):
        p = filedialog.askopenfilename(title="Select checkpoint",
                                       filetypes=[("Checkpoint", "*.pth *.pt"), ("All", "*.*")])
        if p:
            self.var_ckpt.set(p)

    def pick_outdir(self):
        p = filedialog.askdirectory(title="Select output directory")
        if p:
            self.var_outdir.set(p)

    # -- run ---------------------------------------------------------------- #
    def start(self):
        if self.worker and self.worker.is_alive():
            return
        if not self.files:
            messagebox.showwarning("No input", "Add at least one video first.")
            return
        outdir = self.var_outdir.get().strip()
        if not outdir:
            messagebox.showwarning("No output", "Choose an output directory.")
            return
        ckpt = self.var_ckpt.get().strip()
        if not os.path.isfile(ckpt):
            messagebox.showerror("Missing checkpoint", f"Checkpoint not found:\n{ckpt}")
            return

        os.makedirs(outdir, exist_ok=True)
        self._save_settings()

        params = dict(
            files=list(self.files),
            outdir=outdir,
            encoder=self.var_encoder.get(),
            checkpoint=ckpt,
            input_size=int(self.var_input_size.get()),
            max_res=int(self.var_max_res.get()),
            fp32=bool(self.var_fp32.get()),
            preset=self.var_preset.get(),
            overwrite=bool(self.var_overwrite.get()),
        )

        self.cancel_event.clear()
        self.btn_start.config(state="disabled")
        self.btn_cancel.config(state="normal")
        self.progress.config(maximum=len(self.files), value=0)
        self.lbl_overall.config(text=f"0 / {len(self.files)}")
        for child in self.tree.get_children():
            self.tree.set(child, "status", "queued")
            self.tree.set(child, "fps", "-")

        self.worker = threading.Thread(target=self._run_batch, args=(params,), daemon=True)
        self.worker.start()

    def cancel(self):
        self.cancel_event.set()
        self._log("Cancellation requested - will stop after the current video.")

    def _run_batch(self, params):
        q = self.queue
        processor = VideoProcessor(self.model_manager)
        total = len(params["files"])
        done = 0
        try:
            for idx, video_path in enumerate(params["files"]):
                if self.cancel_event.is_set():
                    q.put((MSG_LOG, "Cancelled."))
                    break

                base = os.path.splitext(os.path.basename(video_path))[0]
                out_path = os.path.join(params["outdir"], base + "_depth.mp4")

                if (not params["overwrite"]) and os.path.exists(out_path):
                    q.put((MSG_LOG, f"Skip (exists): {out_path}"))
                    q.put((MSG_STATUS, idx, "skipped"))
                    done += 1
                    q.put((MSG_PROGRESS, done, total))
                    continue

                q.put((MSG_STATUS, idx, "running"))
                q.put((MSG_LOG, "#" * 50))
                q.put((MSG_LOG, f"Processing: {os.path.basename(video_path)}"))
                try:
                    processor.process(
                        video_path, out_path,
                        encoder=params["encoder"],
                        checkpoint=params["checkpoint"],
                        input_size=params["input_size"],
                        max_res=params["max_res"],
                        fp32=params["fp32"],
                        preset=params["preset"],
                        log=lambda m: q.put((MSG_LOG, m)),
                        fps_callback=lambda s, i=idx: q.put((MSG_FILEFPS, i, s)),
                    )
                    q.put((MSG_STATUS, idx, "done"))
                except Exception as e:
                    q.put((MSG_STATUS, idx, "error"))
                    q.put((MSG_LOG, f"ERROR on {os.path.basename(video_path)}: {e}"))
                    q.put((MSG_LOG, traceback.format_exc()))

                done += 1
                q.put((MSG_PROGRESS, done, total))
        finally:
            q.put((MSG_DONE,))

    # -- queue draining (UI thread) ---------------------------------------- #
    def _drain_queue(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                kind = msg[0]
                if kind == MSG_LOG:
                    self._log(msg[1])
                elif kind == MSG_STATUS:
                    self._set_status(msg[1], msg[2])
                elif kind == MSG_FILEFPS:
                    self._set_fps(msg[1], msg[2])
                elif kind == MSG_PROGRESS:
                    done, total = msg[1], msg[2]
                    self.progress.config(value=done)
                    self.lbl_overall.config(text=f"{done} / {total}")
                elif kind == MSG_DONE:
                    self._on_done()
        except queue.Empty:
            pass
        self.root.after(100, self._drain_queue)

    def _on_done(self):
        self.btn_start.config(state="normal")
        self.btn_cancel.config(state="disabled")
        self._log("=" * 50)
        self._log("Finished.")

    def _set_status(self, idx, state):
        children = self.tree.get_children()
        if 0 <= idx < len(children):
            self.tree.set(children[idx], "status", state)

    def _set_fps(self, idx, fps_str):
        children = self.tree.get_children()
        if 0 <= idx < len(children):
            self.tree.set(children[idx], "fps", fps_str)

    def _log(self, text):
        self.log_text.config(state="normal")
        self.log_text.insert("end", text + "\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
