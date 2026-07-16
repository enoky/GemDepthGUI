"""
GemDepth GUI - single / batch video depth processing.

Outputs one grayscale HEVC Main10 depth video per input video. The output
frame rate matches the input exactly, including irregular NTSC rates such as
24000/1001.

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
import shutil
import subprocess
import gc
from fractions import Fraction

# On Windows, recent PyTorch builds can fail with WinError 1114 when torch is
# first imported after GUI/media libraries have already initialized their DLLs.
# Import it before OpenCV and Tkinter, and on the main thread, to keep DLL load
# order deterministic. This also makes a broken PyTorch install fail once at
# application startup instead of once for every queued video.
import torch

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
        # The checkpoint dictionary is only needed while loading weights. Drop
        # it now so a large CPU-side copy is not retained unnecessarily.
        del checkpoint
        gc.collect()
        log("Model ready.")
        return model

    def unload(self, log):
        """Release the cached model and all reclaimable CUDA memory.

        Deleting the CUDA model is required before empty_cache() can return its
        storage to the driver. A small CUDA-context footprint may remain until
        the GUI process exits; that memory is owned by the active CUDA runtime,
        not by the GemDepth model.
        """
        if self._model is None:
            _release_unused_cuda_memory(log, "CUDA cleanup")
            self._encoder = None
            self._ckpt = None
            self.device = None
            return

        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

        model = self._model
        self._model = None
        self._encoder = None
        self._ckpt = None
        self.device = None
        del model
        gc.collect()
        _release_unused_cuda_memory(log, "Model unloaded")


def _format_gib(num_bytes):
    return float(num_bytes) / (1024.0 ** 3)


def _release_unused_cuda_memory(log, label="CUDA cleanup"):
    """Return unused PyTorch allocations to the CUDA driver and log results."""
    if not torch.cuda.is_available():
        return

    try:
        before_alloc = torch.cuda.memory_allocated()
        before_reserved = torch.cuda.memory_reserved()
    except Exception:
        before_alloc = before_reserved = 0

    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception as exc:
        log(f"{label}: torch.cuda.empty_cache() failed: {exc}")
        return

    try:
        after_alloc = torch.cuda.memory_allocated()
        after_reserved = torch.cuda.memory_reserved()
        log(
            f"{label}: PyTorch allocated {_format_gib(before_alloc):.2f} -> "
            f"{_format_gib(after_alloc):.2f} GiB; reserved "
            f"{_format_gib(before_reserved):.2f} -> "
            f"{_format_gib(after_reserved):.2f} GiB."
        )
    except Exception:
        log(f"{label}: unused CUDA cache released.")


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


HEVC_CRF = "8"
HEVC_PRESET = "medium"


def _find_ffmpeg():
    """Return an ffmpeg executable suitable for HEVC Main10 export."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg

    # Optional fallback used by imageio on many Python installs. We do not make
    # this a hard dependency, but using it when present avoids a PATH issue.
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _startupinfo_no_window():
    """Avoid opening an ffmpeg console window on Windows."""
    if os.name != "nt":
        return None
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    return startupinfo


def write_grayscale_depth_video(depths, output_path, fps_fraction, log,
                                percentile_normalize=False):
    """Write a grayscale depth video as HEVC Main10 / yuv420p10le / CFR.

    By default, the complete per-video model range is mapped linearly to
    limited-range 10-bit luma (Y=64..940), without discarding either tail.
    Optional 2nd-98th percentile normalization clips extreme samples before
    mapping. Both modes use one range for the entire video, preserving temporal
    stability and avoiding per-frame brightness pumping.
    """
    if depths is None or len(depths) == 0:
        raise ValueError("no depth frames to write")

    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg is required for HEVC Main10 depth export. "
            "Install FFmpeg or add ffmpeg.exe to PATH."
        )

    depths = np.asarray(depths)
    if depths.ndim == 4:  # [N, 1, H, W] -> [N, H, W]
        depths = depths[:, 0]
    n, h, w = depths.shape[0], depths.shape[-2], depths.shape[-1]

    # yuv420p10le requires even dimensions. The decord path already enforces
    # this; this is a safety net for fallback decoders or unusual source sizes.
    even_h = h if h % 2 == 0 else h + 1
    even_w = w if w % 2 == 0 else w + 1
    if even_h != h or even_w != w:
        log(f"  padding depth frames for yuv420p10le: {w}x{h} -> {even_w}x{even_h}")
        depths = np.pad(
            depths,
            ((0, 0), (0, even_h - h), (0, even_w - w)),
            mode="edge",
        )
        h, w = even_h, even_w

    full_min, full_max = float(np.min(depths)), float(np.max(depths))
    if percentile_normalize:
        d_min, d_max = np.percentile(depths, 2), np.percentile(depths, 98)
        range_label = "2-98% clipped"
    else:
        d_min, d_max = full_min, full_max
        range_label = "full min/max (no percentile clipping)"

    if not np.isfinite(d_min) or not np.isfinite(d_max) or d_max <= d_min:
        d_min, d_max = full_min, full_max
        if not np.isfinite(d_min) or not np.isfinite(d_max) or d_max <= d_min:
            d_min, d_max = 0.0, 1.0
    log(f"  depth range [{range_label}]: {d_min:.4f} - {d_max:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tmp_output = output_path + ".tmp.mp4"
    try:
        if os.path.exists(tmp_output):
            os.remove(tmp_output)
    except OSError:
        pass

    fps_token = (
        str(fps_fraction.numerator)
        if fps_fraction.denominator == 1
        else f"{fps_fraction.numerator}/{fps_fraction.denominator}"
    )

    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "yuv420p10le",
        "-s:v", f"{w}x{h}",
        "-framerate", fps_token,
        "-i", "pipe:0",
        "-an", "-sn", "-dn",
        "-c:v", "libx265",
        "-preset", HEVC_PRESET,
        "-crf", HEVC_CRF,
        "-profile:v", "main10",
        "-pix_fmt", "yuv420p10le",
        "-tag:v", "hvc1",
        "-color_range", "tv",
        "-colorspace", "bt709",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-x265-params", "range=limited:log-level=error",
        "-fps_mode", "cfr",
        "-r", fps_token,
        tmp_output,
    ]

    log(
        f"  writing HEVC Main10 depth: {w}x{h} @ {fps_to_str(fps_fraction)} "
        f"CRF {HEVC_CRF}, {HEVC_PRESET}"
    )

    neutral_uv = np.full((h // 2, w // 2), 512, dtype=np.uint16)
    scale = np.float32(940.0 - 64.0)
    offset = np.float32(64.0)

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        startupinfo=_startupinfo_no_window(),
    )
    try:
        assert proc.stdin is not None
        for i in range(n):
            depth = depths[i]
            norm = np.clip((depth - d_min) / (d_max - d_min + 1e-8), 0, 1)
            y10 = np.rint(norm * scale + offset).astype(np.uint16, copy=False)
            if not y10.flags.c_contiguous:
                y10 = np.ascontiguousarray(y10)
            proc.stdin.write(y10.tobytes())
            proc.stdin.write(neutral_uv.tobytes())
            proc.stdin.write(neutral_uv.tobytes())
    except BrokenPipeError:
        # ffmpeg already failed; collect and report stderr below.
        pass
    finally:
        if proc.stdin:
            try:
                proc.stdin.close()
            except OSError:
                pass

    stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
    rc = proc.wait()
    if rc != 0:
        try:
            if os.path.exists(tmp_output):
                os.remove(tmp_output)
        except OSError:
            pass
        raise RuntimeError(f"ffmpeg HEVC Main10 export failed with code {rc}:\n{stderr.strip()}")

    os.replace(tmp_output, output_path)


class VideoProcessor:
    """Decode -> infer -> write, for a single video. No resampling."""

    def __init__(self, model_manager):
        self.mm = model_manager

    def process(self, video_path, output_path, *, encoder, checkpoint, input_size,
                max_res, fp32, preset, percentile_normalize, log,
                fps_callback=None):
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

        write_grayscale_depth_video(
            depths, output_path, fps_fraction, log,
            percentile_normalize=percentile_normalize,
        )
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
            "percentile_normalize": self.var_percentile_normalize,
            "unload_after_batch": self.var_unload_after_batch,
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
        self.var_percentile_normalize = tk.BooleanVar(value=False)
        self.var_unload_after_batch = tk.BooleanVar(value=True)
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
        ttk.Checkbutton(row2, text="Normalize 2-98%",
                        variable=self.var_percentile_normalize).pack(side="left", padx=6)
        ttk.Checkbutton(row2, text="Unload model after batch",
                        variable=self.var_unload_after_batch).pack(side="left", padx=6)
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
            percentile_normalize=bool(self.var_percentile_normalize.get()),
            unload_after_batch=bool(self.var_unload_after_batch.get()),
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
            # Load PyTorch/model once before entering the per-video loop. A DLL,
            # checkpoint, or model-startup failure is batch-wide; retrying it for
            # every clip only produces a wall of identical errors.
            try:
                q.put((MSG_LOG, "Preparing PyTorch and GemDepth model ..."))
                q.put((MSG_LOG,
                       f"PyTorch {torch.__version__} | CUDA runtime {torch.version.cuda} | "
                       f"CUDA available: {torch.cuda.is_available()}"))
                apply_memory_preset(params["preset"], lambda m: q.put((MSG_LOG, m)))
                self.model_manager.ensure_loaded(
                    params["encoder"], params["checkpoint"],
                    lambda m: q.put((MSG_LOG, m)),
                )
            except Exception as e:
                q.put((MSG_LOG, f"BATCH STARTUP ERROR: {e}"))
                q.put((MSG_LOG, traceback.format_exc()))
                for idx in range(total):
                    q.put((MSG_STATUS, idx, "error"))
                q.put((MSG_PROGRESS, total, total))
                return

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
                        percentile_normalize=params["percentile_normalize"],
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
            # GemDepth was previously kept resident on CUDA forever so the next
            # Start was faster. That also made Task Manager/nvidia-smi show most
            # of the VRAM as still occupied after a completed or cancelled batch.
            if params.get("unload_after_batch", True):
                q.put((MSG_LOG, "Unloading GemDepth model and releasing CUDA memory ..."))
                try:
                    self.model_manager.unload(lambda m: q.put((MSG_LOG, m)))
                except Exception as exc:
                    q.put((MSG_LOG, f"CUDA cleanup warning: {exc}"))
                    q.put((MSG_LOG, traceback.format_exc()))
            else:
                # Keep the model for a faster next batch, but still return any
                # no-longer-used activation cache to the CUDA driver.
                _release_unused_cuda_memory(
                    lambda m: q.put((MSG_LOG, m)),
                    "Batch cleanup (model retained)",
                )
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
