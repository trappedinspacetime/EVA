# -*- coding: utf-8 -*-
"""
EVA Head Mouse - hands-free mouse control with head movement and facial gestures
(Linux / uinput / MediaPipe Face Landmarker)

Controls
--------
  Cursor        : move your head (relative motion + speed-dependent acceleration)
  Single click  : open your mouth and close it again quickly
  Double click  : open/close your mouth twice, quickly
  Drag          : open your mouth and keep it open (released when you close it)
  Right click   : pucker your lips
  Scroll        : raise your eyebrows and hold, then tilt your head up/down (cursor freezes)
  Lock / unlock : smile and hold for ~0.5 s (acts as a toggle switch)
  Dwell click   : keep the cursor still for ~1 s (optional, on by default)

Keyboard shortcuts (preview window must be focused)
  q quit | r re-center | g gesture setup wizard | p lock/unlock | d toggle dwell

Settings: an optional eva_config.json next to this script overrides DEFAULT_CONFIG.
Per-user gesture thresholds are stored in eva_profile.json (created by the setup wizard).

Design notes
------------
  * Relative cursor control with a speed-dependent gain curve: slow = precise, fast = far reaching.
    Pushing past the screen edge keeps the cursor on the edge and re-centers naturally.
  * Multi-point Lucas-Kanade tracking of a set of nose landmarks (outlier rejection),
    anchored to the landmark centroid to prevent drift.
  * A single dt-based One Euro filter + a soft "rest" gain (no sticky dead zones).
  * Absolute (ABS) virtual pointer via uinput, written by a separate high-rate output thread
    that interpolates between camera frames.
  * Camera capture runs in its own thread (MJPG, buffer size 1) for low latency.
  * Cursor freezes while a gesture is being formed and right after a click ("Midas touch" fix).
  * Per-user gesture calibration wizard, hysteresis, and optional hold times.
  * Safety: buttons are released when the face is lost, on lock, on exit and on errors.
  * Optional hybrid mode ("tilt_influence", off by default) that slowly corrects relative drift
    using head orientation.
"""
import argparse
import json
import logging
import math
import os
import signal
import sys
import threading
import time

import cv2
import numpy as np

APP_NAME = "EVA"            # used for window title, virtual device names and config file names
VERSION = "1.0"

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(APP_NAME)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, f"{APP_NAME.lower()}_config.json")
PROFILE_PATH = os.path.join(SCRIPT_DIR, f"{APP_NAME.lower()}_profile.json")
MODEL_NAME = "face_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")

# =====================================================
# CONFIGURATION
# =====================================================
DEFAULT_CONFIG = {
    # --- camera / screen ---
    "camera_index": 0, "cam_w": 640, "cam_h": 480, "cam_fps": 30,
    "monitor_index": 0, "use_gpu": False,
    "show_preview": True,

    # --- tracking (multi-point Lucas-Kanade) ---
    # Nose / nose-bridge landmarks: little affected by facial expressions
    "track_landmarks": [1, 4, 5, 6, 168, 195, 197, 122, 351, 196, 419],
    "lk_win_size": 21, "lk_max_level": 3,
    "lk_anchor_blend": 0.08,      # per-frame pull toward the landmark centroid (prevents drift)
    "lk_drift_hard_reset": 30.0,  # px: larger deviation -> snap back to the landmarks
    "lk_outlier_px": 2.5,         # px: points deviating this much from the median motion are dropped
    "lk_min_points": 4,

    # --- filter (camera-pixel space, single filter) ---
    "min_cutoff": 1.2,   # Hz: lower = less jitter at rest, more lag
    "beta": 0.08,        # higher = less lag while moving
    "d_cutoff": 1.0,

    # --- acceleration curve (screen px per camera px, 1920x1080 reference) ---
    "gain_min": 7.0, "gain_max": 32.0, "gain_speed_ref": 120.0,   # camera px/s
    "sens_x": 2.0, "sens_y": 2.0,   # overall multipliers (cursor travel per head movement)
    "rest_speed_lo": 1.0, "rest_speed_hi": 5.0,   # below this speed the gain smoothly drops to 0
    "max_head_step_px": 40.0,                     # per-frame jumps larger than this are ignored as glitches
    "start_x": 0.5, "start_y": 0.5,

    # --- hybrid mode (drift correction using head orientation) ---
    "tilt_influence": 0.0,        # 0 = off; try 0.2 to experiment
    "yaw_range": 0.12, "pitch_range": 0.10,   # ratio change that maps to half the screen
    "pull_rate": 1.5, "pull_gate_speed": 300.0,

    # --- output ---
    "output_hz": 120, "interpolate": True,

    # --- gestures (defaults; the setup wizard overrides them per user) ---
    "gestures": {
        # mouth: inner-lip distance / face height. hold=0: detect instantly
        "mouth":  {"on": 0.04, "off": 0.025, "hold": 0.0, "cooldown": 0.0},
        "smile":  {"on": 0.50, "off": 0.35, "hold": 0.20, "cooldown": 0.5},
        "brow":   {"on": 0.45, "off": 0.30, "hold": 0.08, "cooldown": 0.3},
        "pucker": {"on": 0.60, "off": 0.40, "hold": 0.10, "cooldown": 0.5},
    },
    "freeze_on_arming": True,   # freeze the cursor while a gesture is forming
    "click_freeze_s": 0.35,     # freeze after a click (keeps double clicks on target)
    "drag_hold_s": 0.45,        # mouth open longer than this = drag, shorter = click
    "double_tap_s": 0.6,        # two taps closer than this show "DOUBLE CLICK" in the status text

    # --- lock switch (smile) ---
    "smile_lock_enabled": True,   # smile and hold to lock / unlock the mouse
    "lock_hold_s": 0.5,           # how long to hold the smile

    # --- scroll (eyebrows raised + head tilt) ---
    "scroll_enabled": True,
    "scroll_deadzone": 0.015, "scroll_full": 0.06, "scroll_max_rate": 12.0,  # notches per second
    "scroll_invert": False,

    # --- dwell click ---
    "dwell_enabled": True, "dwell_s": 1.0, "dwell_radius_px": 20,

    # --- edge nudge (for docks/panels that need "pressure" against the screen edge) ---
    # While your head keeps pushing the cursor against a screen edge, a separate tiny relative mouse
    # device sends small movements toward that edge, so edge triggers (e.g. auto-hide docks) fire.
    "edge_nudge": True, "edge_nudge_px": 3, "edge_hold_s": 0.6,

    # --- calibration ---
    "head_cal_s": 2.0, "wizard_step_s": 2.5, "wizard_ignore_s": 0.7,
}


def deep_merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        log.warning(f"Could not read {os.path.basename(path)}: {e}")
        return None


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


# =====================================================
# FILTERS / TRACKING
# =====================================================
class OneEuro2D:
    """dt-based One Euro filter; x and y share the same cutoff (isotropic)."""
    def __init__(self, min_cutoff, beta, d_cutoff):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self):
        self.t = None
        self.x = None
        self.raw = None
        self.dx = np.zeros(2)

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, p, t):
        p = np.asarray(p, dtype=np.float64)
        if self.t is None:
            self.t, self.x, self.raw = t, p.copy(), p.copy()
            self.dx = np.zeros(2)
            return p.copy()
        dt = max(t - self.t, 1e-3)
        self.t = t
        v = (p - self.raw) / dt
        self.raw = p.copy()
        ad = self._alpha(self.d_cutoff, dt)
        self.dx = ad * v + (1 - ad) * self.dx
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self.dx))
        a = self._alpha(cutoff, dt)
        self.x = a * p + (1 - a) * self.x
        return self.x.copy()


class MultiPointLK:
    """
    Places the landmark template (offsets from the centroid) around the tracked position,
    measures each point's motion with Lucas-Kanade, rejects outliers and accumulates the mean motion.
    A slow pull toward the landmark centroid (anchor) prevents drift.
    """
    def __init__(self, cfg):
        w = int(cfg["lk_win_size"])
        self.params = dict(winSize=(w, w), maxLevel=int(cfg["lk_max_level"]),
                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))
        self.blend = cfg["lk_anchor_blend"]
        self.hard = cfg["lk_drift_hard_reset"]
        self.outlier = cfg["lk_outlier_px"]
        self.min_pts = int(cfg["lk_min_points"])
        self.reset()

    def reset(self):
        self.prev_gray = None
        self.p0 = None
        self.pos = None

    def _seed(self, gray, pos, offsets):
        self.pos = np.asarray(pos, dtype=np.float64)
        p0 = (self.pos + offsets).astype(np.float32)
        h, w = gray.shape[:2]
        p0[:, 0] = np.clip(p0[:, 0], 0, w - 1)
        p0[:, 1] = np.clip(p0[:, 1], 0, h - 1)
        self.p0 = p0.reshape(-1, 1, 2)
        self.prev_gray = gray

    def update(self, gray, pts):
        pts = np.asarray(pts, dtype=np.float64)
        if not np.isfinite(pts).all():
            return None if self.pos is None else self.pos.copy()
        centroid = pts.mean(axis=0)
        offsets = pts - centroid
        if self.prev_gray is None or self.pos is None:
            self._seed(gray, centroid, offsets)
            return self.pos.copy()

        move = None
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.p0, None, **self.params)
        if p1 is not None and st is not None:
            ok = st.reshape(-1) == 1
            disp = p1.reshape(-1, 2).astype(np.float64) - self.p0.reshape(-1, 2).astype(np.float64)
            if ok.sum() >= self.min_pts:
                med = np.median(disp[ok], axis=0)
                inl = ok & (np.linalg.norm(disp - med, axis=1) <= self.outlier)
                if inl.sum() >= self.min_pts:
                    move = disp[inl].mean(axis=0)

        if move is None:
            pos = centroid
        else:
            pos = self.pos + move
            if np.linalg.norm(pos - centroid) > self.hard:
                pos = centroid
            else:
                pos = (1 - self.blend) * pos + self.blend * centroid
        self._seed(gray, pos, offsets)
        return self.pos.copy()


def head_gain(speed, cfg):
    """Screen px per camera px as a function of camera px/s (acceleration curve + rest zone)."""
    g = cfg["gain_min"] + (cfg["gain_max"] - cfg["gain_min"]) * smoothstep(speed / cfg["gain_speed_ref"])
    lo, hi = cfg["rest_speed_lo"], cfg["rest_speed_hi"]
    rest = smoothstep((speed - lo) / max(hi - lo, 1e-6))
    return g * rest


# =====================================================
# GESTURE STATE MACHINES
# =====================================================
class Gesture:
    """Hysteresis + minimum hold time + cooldown + 'arming' (gesture forming) state."""
    ARM_TIMEOUT = 0.8

    def __init__(self, on, off, hold=0.1, cooldown=0.0):
        self.set_thresholds(on, off)
        self.hold, self.cooldown = hold, cooldown
        self.reset()

    def set_thresholds(self, on, off):
        self.on, self.off = float(on), float(off)
        self.pre = self.off

    def reset(self):
        self.active = False
        self.arming = False
        self.active_since = 0.0
        self.last_duration = 0.0
        self._on_since = None
        self._pre_since = None
        self._cool_until = 0.0

    def update(self, v, t):
        ev = []
        if v >= self.pre:
            if self._pre_since is None:
                self._pre_since = t
        else:
            self._pre_since = None
        if not self.active:
            if v >= self.on:
                if self._on_since is None:
                    self._on_since = t
                if t - self._on_since >= self.hold and t >= self._cool_until:
                    self.active, self.active_since = True, t
                    ev.append("start")
            else:
                self._on_since = None
        elif v <= self.off:
            self.active = False
            self.last_duration = t - self.active_since
            self._cool_until = t + self.cooldown
            self._on_since = None
            ev.append("end")
        self.arming = (not self.active and self._pre_since is not None
                       and (t - self._pre_since) <= self.ARM_TIMEOUT)
        return ev


class DwellClicker:
    def __init__(self, cfg):
        self.dwell_s, self.radius = cfg["dwell_s"], cfg["dwell_radius_px"]
        self.reset()

    def reset(self):
        self.anchor, self.since, self.armed, self.progress = None, 0.0, False, 0.0

    def update(self, x, y, t, allowed):
        if not allowed or self.anchor is None:
            self.anchor, self.since, self.progress = (x, y), t, 0.0
            # after any other action the cursor must move away before dwell can fire again
            self.armed = bool(allowed)
            return False
        if math.hypot(x - self.anchor[0], y - self.anchor[1]) > self.radius:
            self.anchor, self.since, self.armed, self.progress = (x, y), t, True, 0.0
            return False
        if not self.armed:
            self.progress = 0.0
            return False
        self.progress = min((t - self.since) / self.dwell_s, 1.0)
        if self.progress >= 1.0:
            self.armed, self.progress = False, 0.0
            return True
        return False


# =====================================================
# FACE FEATURES
# =====================================================
def extract_scores(blendshapes):
    d = {bs.category_name: bs.score for bs in blendshapes} if blendshapes else {}
    return {
        "mouth": 0.0,   # computed from landmarks in main() and filled in there
        "smile": max(d.get("mouthSmileLeft", 0.0), d.get("mouthSmileRight", 0.0)),
        "brow": d.get("browInnerUp", 0.0),
        "pucker": d.get("mouthPucker", 0.0),
    }


def lip_ratio(lm, w, h):
    """Inner-lip distance (landmarks 13-14) / face height (landmarks 10-152)."""
    lip = math.hypot((lm[13].x - lm[14].x) * w, (lm[13].y - lm[14].y) * h)
    face_h = math.hypot((lm[10].x - lm[152].x) * w, (lm[10].y - lm[152].y) * h)
    return lip / face_h if face_h > 0 else 0.0


def head_proxies(lm, w, h):
    """
    Geometric ratios for head orientation with an unambiguous sign (on the mirrored image):
      yaw   > 0 : nose appears to the right (head turned right)
      pitch > 0 : nose sits lower relative to the face (head tilted down)
    """
    nx, ny = lm[4].x * w, lm[4].y * h
    lx, rx = lm[234].x * w, lm[454].x * w
    ty, cy = lm[10].y * h, lm[152].y * h
    fw = max(abs(rx - lx), 1e-3)
    fh = max(abs(cy - ty), 1e-3)
    return {"yaw": (nx - (lx + rx) / 2.0) / fw, "pitch": (ny - (ty + cy) / 2.0) / fh}


# =====================================================
# VIRTUAL POINTER (uinput, ABS) + OUTPUT THREAD
# =====================================================
class VirtualPointer:
    """ABS position + 3 buttons + wheel. Falls back to a dry run (preview only) without uinput."""
    def __init__(self, desk_w, desk_h, off_x=0, off_y=0):
        self.desk_w, self.desk_h, self.off_x, self.off_y = desk_w, desk_h, off_x, off_y
        self.ui = None
        self.ui_rel = None
        self._lock = threading.RLock()
        self._held = set()
        try:
            from evdev import UInput, AbsInfo, ecodes as e
            self.e = e
            cap = {
                e.EV_KEY: [e.BTN_LEFT, e.BTN_RIGHT, e.BTN_MIDDLE],
                e.EV_ABS: [(e.ABS_X, AbsInfo(0, 0, desk_w - 1, 0, 0, 0)),
                           (e.ABS_Y, AbsInfo(0, 0, desk_h - 1, 0, 0, 0))],
                e.EV_REL: [e.REL_WHEEL],
            }
            self.ui = UInput(cap, name=f"{APP_NAME} Virtual Pointer", version=0x3)
            # Plain relative mouse used for the edge nudge (BTN_LEFT is declared, never pressed,
            # so libinput recognizes the device as a pointer)
            self.ui_rel = UInput({e.EV_KEY: [e.BTN_LEFT], e.EV_REL: [e.REL_X, e.REL_Y]},
                                 name=f"{APP_NAME} Edge Nudge", version=0x3)
            log.info(f"[{APP_NAME}] Virtual pointer ready (ABS {desk_w}x{desk_h})")
        except Exception as ex:
            log.error(f"[{APP_NAME}] Could not open uinput: {ex}\n"
                      "      You need the 'uinput' kernel module and permission for /dev/uinput "
                      "(add your user to the 'input' group + a udev rule, or use sudo). See the README.\n"
                      "      DRY RUN mode: the mouse will not move, the preview still works.")

    def _btn(self, name):
        return {"left": self.e.BTN_LEFT, "right": self.e.BTN_RIGHT, "middle": self.e.BTN_MIDDLE}[name]

    def move_abs(self, x, y):
        if not self.ui:
            return
        with self._lock:
            self.ui.write(self.e.EV_ABS, self.e.ABS_X, int(min(max(x + self.off_x, 0), self.desk_w - 1)))
            self.ui.write(self.e.EV_ABS, self.e.ABS_Y, int(min(max(y + self.off_y, 0), self.desk_h - 1)))
            self.ui.syn()

    def press(self, name):
        self._held.add(name)
        if self.ui:
            with self._lock:
                self.ui.write(self.e.EV_KEY, self._btn(name), 1)
                self.ui.syn()

    def release(self, name):
        self._held.discard(name)
        if self.ui:
            with self._lock:
                self.ui.write(self.e.EV_KEY, self._btn(name), 0)
                self.ui.syn()

    def click(self, name, count=1):
        for _ in range(count):
            self.press(name)
            time.sleep(0.015)
            self.release(name)
            time.sleep(0.015)

    def nudge(self, dx, dy):
        """Edge nudge: tiny relative movement (no visible effect if the cursor is already on the edge)."""
        if self.ui_rel and (dx or dy):
            with self._lock:
                if dx:
                    self.ui_rel.write(self.e.EV_REL, self.e.REL_X, int(dx))
                if dy:
                    self.ui_rel.write(self.e.EV_REL, self.e.REL_Y, int(dy))
                self.ui_rel.syn()

    def wheel(self, n):
        """n > 0: scroll up, n < 0: scroll down (Linux REL_WHEEL convention)."""
        if self.ui and n:
            with self._lock:
                self.ui.write(self.e.EV_REL, self.e.REL_WHEEL, int(n))
                self.ui.syn()

    def release_all(self):
        for name in list(self._held):
            self.release(name)

    def close(self):
        try:
            self.release_all()
            if self.ui:
                self.ui.close()
            if self.ui_rel:
                self.ui_rel.close()
        except Exception:
            pass


class OutputThread(threading.Thread):
    """Writes to the pointer at a high rate, interpolating between the ~30 Hz camera targets."""
    def __init__(self, pointer, hz, interpolate):
        super().__init__(daemon=True)
        self.pointer, self.period, self.interpolate = pointer, 1.0 / max(hz, 30), interpolate
        self._lock = threading.Lock()
        self.start_pt = self.target = np.zeros(2)
        self.t0 = time.monotonic()
        self.interval = 1 / 30
        self._stop_evt = threading.Event()
        self._last = None

    def _current(self, now):
        if not self.interpolate:
            return self.target
        a = min(max((now - self.t0) / self.interval, 0.0), 1.0)
        return self.start_pt + (self.target - self.start_pt) * a

    def set_target(self, x, y, interval=None):
        with self._lock:
            now = time.monotonic()
            self.start_pt = self._current(now).copy()
            self.target = np.array([x, y], dtype=np.float64)
            self.t0 = now
            if interval:
                self.interval = min(max(interval, 0.015), 0.1)

    def jump(self, x, y):
        with self._lock:
            self.start_pt = self.target = np.array([x, y], dtype=np.float64)
            self.t0 = time.monotonic()
        self._last = None

    def run(self):
        nxt = time.monotonic()
        while not self._stop_evt.is_set():
            now = time.monotonic()
            with self._lock:
                p = self._current(now)
            pi = (int(round(p[0])), int(round(p[1])))
            if pi != self._last:
                self.pointer.move_abs(*pi)
                self._last = pi
            nxt += self.period
            delay = nxt - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                nxt = time.monotonic()

    def stop(self):
        self._stop_evt.set()


# =====================================================
# CONTROLLER (all behaviour logic)
# =====================================================
WIZARD_STEPS = [
    ("rest", "REST: relax your face", None),
    ("mouth", "OPEN MOUTH slightly (small, comfortable gap) and hold", "mouth"),
    ("smile", "SMILE and hold", "smile"),
    ("brow", "RAISE EYEBROWS and hold", "brow"),
    ("pucker", "PUCKER LIPS and hold", "pucker"),
]
# gesture -> (min signal span, on fraction, off fraction, upper limit for the on threshold)
WIZARD_PARAMS = {
    "default": (0.12, 0.55, 0.30, 0.95),
    "mouth":   (0.015, 0.50, 0.30, 0.06),   # small-scale ratio; capped so it stays sensitive
}
GESTURE_NAMES = ("mouth", "smile", "brow", "pucker")


class Controller:
    HEAD, WIZ, RUN = "head", "wiz", "run"

    def __init__(self, cfg, pointer, out, W, H, profile=None):
        self.cfg, self.ptr, self.out, self.W, self.H = cfg, pointer, out, W, H
        self.tracker = MultiPointLK(cfg)
        self.filt = OneEuro2D(cfg["min_cutoff"], cfg["beta"], cfg["d_cutoff"])
        self.gest = {n: Gesture(**cfg["gestures"][n]) for n in GESTURE_NAMES}
        self.have_profile = False
        if profile and "gestures" in profile:
            for n, d in profile["gestures"].items():
                if n in self.gest:
                    self.gest[n].set_thresholds(d["on"], d["off"])
            self.have_profile = True
        self.dwell = DwellClicker(cfg)
        self.dwell_enabled = cfg["dwell_enabled"]
        self.smile_lock_enabled = bool(cfg["smile_lock_enabled"])
        self.cx, self.cy = W * cfg["start_x"], H * cfg["start_y"]
        self.out.jump(self.cx, self.cy)
        self.locked = False
        self.drag_on = False
        self._mouth_state = None    # None | "pending" | "drag" | "blocked"
        self._last_tap_t = -99.0
        self.freeze_until = 0.0
        self.prev_fp = self.prev_t = None
        self.neutral = None
        self.lost_since = None
        self.lost_handled = False
        self.lock_fired = False
        self.scroll_ref = None
        self.scroll_acc = 0.0
        self._edge_dir = [0, 0]
        self._edge_until = [0.0, 0.0]
        self.interval = 1 / 30
        self.status, self.status_color, self.status_until = "", (200, 200, 200), 0.0
        self.scores = {n: 0.0 for n in GESTURE_NAMES}
        self.last_gain = 0.0
        self.request_recal(force_wizard=not self.have_profile)

    # ---------- helpers ----------
    def set_status(self, text, color=(255, 255, 255), t=0.0, hold=0.9):
        self.status, self.status_color, self.status_until = text, color, t + hold

    def status_text(self, t):
        if t < self.status_until:
            return self.status, self.status_color
        if self.phase == self.HEAD:
            return "Calibrating", (0, 255, 255)
        if self.phase == self.WIZ:
            return "Gesture setup", (0, 255, 255)
        if self.locked:
            hint = "smile to unlock" if self.smile_lock_enabled else "press p to unlock"
            return f"LOCKED ({hint})", (0, 0, 255)
        if self.drag_on:
            return "DRAGGING", (0, 255, 0)
        return "READY", (200, 200, 200)

    def release_all(self):
        self.drag_on = False
        self._mouth_state = None
        self.ptr.release_all()
        self.scroll_ref = None

    def request_recal(self, force_wizard=False):
        self.release_all()
        self.phase = self.HEAD
        self._force_wizard = force_wizard
        self._head_start = None
        self._head_samples = []
        self.tracker.reset()
        self.filt.reset()
        self.prev_fp = self.prev_t = None

    def toggle_lock(self, t=0.0):
        self.locked = not self.locked
        self.release_all()
        self.dwell.reset()
        if self.locked:
            self.set_status("LOCKED", (0, 0, 255), t, 1.5)
        else:
            self.set_status("UNLOCKED", (0, 255, 0), t, 1.5)

    # ---------- no face ----------
    def on_no_face(self, t):
        if self.lost_since is None:
            self.lost_since = t
        lost = t - self.lost_since
        if lost > 0.3:
            self.tracker.reset()
            self.filt.reset()
            self.prev_fp = self.prev_t = None
            self._head_samples, self._head_start = [], None
        if lost > 1.0 and not self.lost_handled:
            self.release_all()          # safety: never leave a button held
            for g in self.gest.values():
                g.reset()
            self.lost_handled = True
        self.set_status("FACE LOST", (0, 0, 255), t, 0.3)

    # ---------- face present ----------
    def on_face(self, t, gray, pts, proxies, scores, interval=None):
        self.lost_since, self.lost_handled = None, False
        if interval:
            self.interval = interval
        self.scores = scores
        pos = self.tracker.update(gray, pts)
        if pos is None:
            return
        fp = self.filt(pos, t)
        if self.phase == self.HEAD:
            self._head_cal(t, pos, proxies)
        elif self.phase == self.WIZ:
            self._wizard(t, scores)
        else:
            self._run(t, fp, proxies, scores)
        self.prev_fp, self.prev_t = fp, t

    # ---------- head calibration ----------
    def _head_cal(self, t, pos, proxies):
        if self._head_start is None:
            self._head_start, self._head_samples = t, []
        self._head_samples.append((pos.copy(), proxies["yaw"], proxies["pitch"]))
        P = np.array([s[0] for s in self._head_samples])
        if np.ptp(P, axis=0).max() > 12.0:      # moved -> start over
            self._head_start, self._head_samples = t, []
        el = t - self._head_start
        if el >= self.cfg["head_cal_s"] and len(self._head_samples) >= 10:
            self.neutral = (float(np.median([s[1] for s in self._head_samples])),
                            float(np.median([s[2] for s in self._head_samples])))
            self.cx, self.cy = self.W * self.cfg["start_x"], self.H * self.cfg["start_y"]
            self.out.jump(self.cx, self.cy)
            self.dwell.reset()
            if self._force_wizard:
                self._start_wizard(t)
            else:
                self.phase = self.RUN
                self.set_status("Calibration done", (0, 255, 0), t)
        else:
            self.set_status(f"HOLD YOUR HEAD STILL: {max(self.cfg['head_cal_s'] - el, 0):.1f} s",
                            (0, 255, 255), t, 0.2)

    # ---------- gesture setup wizard ----------
    def _start_wizard(self, t):
        self.phase = self.WIZ
        self._wiz_i = 0
        self._wiz_t0 = t
        # Skip steps for gestures that are disabled
        skip = set()
        if not self.smile_lock_enabled:
            skip.add("smile")
        if not self.cfg["scroll_enabled"]:
            skip.add("brow")
        self._wiz_steps = [st for st in WIZARD_STEPS if st[0] not in skip]
        self._wiz_rec = {name: {g: [] for g in GESTURE_NAMES} for name, _, _ in self._wiz_steps}

    def _wizard(self, t, scores):
        name, label, _ = self._wiz_steps[self._wiz_i]
        el = t - self._wiz_t0
        if el >= self.cfg["wizard_ignore_s"]:
            for g in GESTURE_NAMES:
                self._wiz_rec[name][g].append(scores[g])
        self.set_status(f"SETUP {self._wiz_i + 1}/{len(self._wiz_steps)}: {label} "
                        f"({max(self.cfg['wizard_step_s'] - el, 0):.0f})", (0, 255, 255), t, 0.2)
        if el >= self.cfg["wizard_step_s"]:
            if len(self._wiz_rec[name]["mouth"]) < 8:      # not enough samples -> repeat the step
                self._wiz_t0 = t
                self._wiz_rec[name] = {g: [] for g in GESTURE_NAMES}
                return
            self._wiz_i += 1
            self._wiz_t0 = t
            if self._wiz_i >= len(self._wiz_steps):
                self._finish_wizard(t)

    def _finish_wizard(self, t):
        rest = self._wiz_rec["rest"]
        # keep thresholds of gestures that were not part of this run
        old = (load_json(PROFILE_PATH) or {}).get("gestures", {})
        prof = {"gestures": {k: v for k, v in old.items() if k in GESTURE_NAMES}}
        notes = []
        for name, _, g in self._wiz_steps:
            if g is None:
                continue
            min_span, on_frac, off_frac, on_max = WIZARD_PARAMS.get(g, WIZARD_PARAMS["default"])
            base = float(np.percentile(rest[g], 95))
            peak = float(np.percentile(self._wiz_rec[name][g], 75))
            span = peak - base
            if span < min_span:
                notes.append(g)
                continue
            on = min(base + on_frac * span, on_max)
            off = min(base + off_frac * span, on * 0.8)
            self.gest[g].set_thresholds(on, off)
            prof["gestures"][g] = {"on": round(on, 3), "off": round(off, 3)}
        try:
            with open(PROFILE_PATH, "w", encoding="utf-8") as f:
                json.dump(prof, f, indent=2)
        except Exception as e:
            log.warning(f"Could not save profile: {e}")
        self.have_profile = True
        self.phase = self.RUN
        msg = "Setup done" + (f" (weak signal, using defaults: {', '.join(notes)})" if notes else "")
        self.set_status(msg, (0, 255, 0) if not notes else (0, 200, 255), t, 3.0)
        log.info(msg + f" -> {prof}")

    # ---------- normal operation ----------
    def _handle_gestures(self, t, proxies):
        c = self.cfg
        vals = dict(self.scores)
        if not self.smile_lock_enabled:
            vals["smile"] = 0.0         # disabled gestures are ignored completely
        if not c["scroll_enabled"]:
            vals["brow"] = 0.0
        ev = {n: self.gest[n].update(vals[n], t) for n in GESTURE_NAMES}
        mouth, smile, brow, pucker = (self.gest[n] for n in GESTURE_NAMES)

        # Smile (held ~lock_hold_s): toggle lock / unlock. Works even while locked.
        if smile.active and not self.lock_fired and (t - smile.active_since) >= c["lock_hold_s"]:
            self.lock_fired = True
            self.toggle_lock(t)
        if "end" in ev["smile"]:
            self.lock_fired = False

        if self.locked:
            return

        # Mouth:
        #   open and close quickly            -> single click (two quick ones -> OS sees a double click)
        #   keep open (longer than drag_hold_s) -> drag; released when you close your mouth
        if "start" in ev["mouth"]:
            if smile.active or pucker.active or smile.arming or pucker.arming or brow.active:
                self._mouth_state = "blocked"
            else:
                self._mouth_state = "pending"
        if mouth.active and self._mouth_state == "pending" and (t - mouth.active_since) >= c["drag_hold_s"]:
            self.ptr.press("left")
            self.drag_on = True
            self._mouth_state = "drag"
            self.set_status("DRAG (close mouth to release)", (0, 255, 0), t, 0.6)
        if "end" in ev["mouth"]:
            if self._mouth_state == "drag":
                self.ptr.release("left")
                self.drag_on = False
                self.freeze_until = t + 0.15
                self.set_status("DRAG ENDED", (0, 180, 0), t)
            elif self._mouth_state == "pending":
                dbl = (t - self._last_tap_t) <= c["double_tap_s"]
                self.ptr.click("left")
                self._last_tap_t = t
                self.freeze_until = t + c["click_freeze_s"]
                self.set_status("DOUBLE CLICK" if dbl else "SINGLE CLICK", (255, 255, 0), t)
            self._mouth_state = None

        # Pucker: right click
        if "start" in ev["pucker"] and not (self._mouth_state in ("pending", "drag") or smile.active or brow.active):
            self.ptr.click("right")
            self.freeze_until = t + c["click_freeze_s"]
            self.set_status("RIGHT CLICK", (255, 0, 255), t)

        # Eyebrows (held): scroll mode, tilt the head up/down
        if "start" in ev["brow"]:
            self.scroll_ref, self.scroll_acc = proxies["pitch"], 0.0
        if "end" in ev["brow"]:
            self.scroll_ref = None

    def _run(self, t, fp, proxies, scores):
        c = self.cfg
        dt = (t - self.prev_t) if self.prev_t else 1 / 30
        self._handle_gestures(t, proxies)
        gs = list(self.gest.values())
        frozen = (t < self.freeze_until or self._mouth_state == "pending"
                  or (self.smile_lock_enabled and self.gest["smile"].active)
                  or (c["scroll_enabled"] and self.gest["brow"].active)
                  or (c["freeze_on_arming"] and any(g.arming for g in gs)))

        dx = dy = 0.0
        if not (frozen or self.locked) and self.prev_fp is not None and 0 < dt <= 0.25:
            d = fp - self.prev_fp
            step = float(np.hypot(d[0], d[1]))
            if step <= c["max_head_step_px"]:
                speed = step / dt
                g = head_gain(speed, c)
                self.last_gain = g
                dx = d[0] * g * c["sens_x"] * self.W / 1920.0
                dy = d[1] * g * c["sens_y"] * self.H / 1080.0
        nx, ny = self.cx + dx, self.cy + dy
        self.cx = min(max(nx, 0.0), self.W - 1.0)
        self.cy = min(max(ny, 0.0), self.H - 1.0)

        # Edge nudge: if the head keeps pushing past an edge (overflow), keep nudging for a short time
        if c["edge_nudge"] and not (frozen or self.locked):
            for ax, over in ((0, nx - self.cx), (1, ny - self.cy)):
                if abs(over) > 1e-6:
                    self._edge_dir[ax] = 1 if over > 0 else -1
                    self._edge_until[ax] = t + c["edge_hold_s"]
            push = [0, 0]
            for ax, pos, lim in ((0, self.cx, self.W - 1.0), (1, self.cy, self.H - 1.0)):
                d_ = self._edge_dir[ax]
                at_edge = (d_ > 0 and pos >= lim - 0.5) or (d_ < 0 and pos <= 0.5)
                if t < self._edge_until[ax] and at_edge:
                    push[ax] = d_ * int(c["edge_nudge_px"])
            if push[0] or push[1]:
                self.ptr.nudge(push[0], push[1])

        # Hybrid: while moving, slowly pull toward the head-orientation target (drift correction)
        if c["tilt_influence"] > 0 and self.neutral is not None and not (frozen or self.locked):
            tx = self.W * c["start_x"] + (proxies["yaw"] - self.neutral[0]) / c["yaw_range"] * (self.W / 2)
            ty = self.H * c["start_y"] + (proxies["pitch"] - self.neutral[1]) / c["pitch_range"] * (self.H / 2)
            tx, ty = min(max(tx, 0), self.W - 1), min(max(ty, 0), self.H - 1)
            gate = min(math.hypot(dx, dy) / max(dt, 1e-3) / c["pull_gate_speed"], 1.0)
            k = 1 - math.exp(-c["pull_rate"] * c["tilt_influence"] * gate * dt)
            self.cx += (tx - self.cx) * k
            self.cy += (ty - self.cy) * k

        if not self.locked:
            self.out.set_target(self.cx, self.cy, self.interval)

        # Scroll: proportional to head tilt (head down = scroll down)
        if (c["scroll_enabled"] and self.gest["brow"].active and not self.locked
                and self.scroll_ref is not None):
            d = proxies["pitch"] - self.scroll_ref
            mag = (abs(d) - c["scroll_deadzone"]) / max(c["scroll_full"] - c["scroll_deadzone"], 1e-6)
            mag = min(max(mag, 0.0), 1.0)
            if mag > 0:
                direction = -1 if d > 0 else 1          # REL_WHEEL: + up, - down
                if c["scroll_invert"]:
                    direction = -direction
                self.scroll_acc += direction * mag * c["scroll_max_rate"] * dt
                n = int(self.scroll_acc)
                if n:
                    self.ptr.wheel(n)
                    self.scroll_acc -= n
                self.set_status("SCROLL " + ("DOWN" if direction < 0 else "UP"), (0, 255, 255), t, 0.2)

        # Dwell click
        allowed = (self.dwell_enabled and not self.locked and not frozen and not self.drag_on
                   and not any(g.active for g in gs))
        if self.dwell.update(self.cx, self.cy, t, allowed):
            self.ptr.click("left")
            self.freeze_until = t + 0.1
            self.set_status("DWELL CLICK", (255, 255, 0), t)


# =====================================================
# CAMERA THREAD
# =====================================================
class CameraThread(threading.Thread):
    def __init__(self, cfg):
        super().__init__(daemon=True)
        idx = cfg["camera_index"]
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2) if sys.platform.startswith("linux") else cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["cam_w"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["cam_h"])
        cap.set(cv2.CAP_PROP_FPS, cfg["cam_fps"])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap = cap
        self.cond = threading.Condition()
        self.frame, self.t, self.id = None, 0.0, 0     # id 0 = no frame yet
        self.failed = not cap.isOpened()
        self._stop_flag = False

    def run(self):
        while not self._stop_flag and not self.failed:
            ok, f = self.cap.read()
            t = time.monotonic()
            with self.cond:
                if not ok:
                    self.failed = True
                else:
                    self.frame, self.t, self.id = f, t, self.id + 1
                self.cond.notify_all()

    def wait_new(self, last_id, timeout=0.2):
        with self.cond:
            self.cond.wait_for(lambda: self.id != last_id or self.failed or self._stop_flag, timeout)
            if self.id == last_id or self.frame is None:
                return None
            return self.id, self.frame, self.t

    def stop(self):
        self._stop_flag = True
        with self.cond:
            self.cond.notify_all()
        self.join(timeout=1.0)
        self.cap.release()


# =====================================================
# PREVIEW OVERLAY
# =====================================================
def render_status_bar(lines, width, height=84):
    bar = np.zeros((height, width, 3), np.uint8)
    for i, (text, col) in enumerate(lines):
        cv2.putText(bar, text, (8, 20 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1, cv2.LINE_AA)
    return bar


# =====================================================
# MAIN
# =====================================================
def get_screen_layout(cfg):
    """(target monitor w, h, whole desktop w, h, monitor offset x, y)"""
    try:
        from screeninfo import get_monitors
        mons = get_monitors()
        m = mons[min(cfg["monitor_index"], len(mons) - 1)]
        min_x, min_y = min(x.x for x in mons), min(x.y for x in mons)
        dw = max(x.x + x.width for x in mons) - min_x
        dh = max(x.y + x.height for x in mons) - min_y
        return m.width, m.height, dw, dh, m.x - min_x, m.y - min_y
    except Exception:
        log.warning(f"[{APP_NAME}] screeninfo missing or failed, assuming 1920x1080 (pip install screeninfo)")
        return 1920, 1080, 1920, 1080, 0, 0


def find_model():
    for p in (os.path.join(SCRIPT_DIR, MODEL_NAME), os.path.join(os.getcwd(), MODEL_NAME)):
        if os.path.exists(p):
            return p
    return None


def ensure_model():
    """Return the model path, trying to download it next to the script if it is missing."""
    path = find_model()
    if path:
        return path
    dest = os.path.join(SCRIPT_DIR, MODEL_NAME)
    try:
        import urllib.request
        log.info(f"[{APP_NAME}] {MODEL_NAME} not found, downloading it (about 4 MB) ...")
        urllib.request.urlretrieve(MODEL_URL, dest + ".part")
        os.replace(dest + ".part", dest)
        return dest
    except Exception as e:
        try:
            os.remove(dest + ".part")
        except OSError:
            pass
        log.error(f"[{APP_NAME}] Download failed ({e}). Download it manually:\n  {MODEL_URL}\n"
                  f"and place it next to this script as {MODEL_NAME}")
        return None


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Hands-free mouse control with head movement and facial gestures.")
    ap.add_argument("--config", default=CONFIG_PATH, help="path to a JSON config file")
    ap.add_argument("--profile", default=PROFILE_PATH, help="path to the gesture profile JSON file")
    ap.add_argument("--no-preview", action="store_true", help="do not open the preview window")
    ap.add_argument("--wizard", action="store_true", help="run the gesture setup wizard at startup")
    return ap.parse_args(argv)


def main(argv=None):
    global PROFILE_PATH
    args = parse_args(argv)
    PROFILE_PATH = args.profile
    cfg = deep_merge(DEFAULT_CONFIG, load_json(args.config))
    if args.no_preview:
        cfg["show_preview"] = False
    profile = None if args.wizard else load_json(PROFILE_PATH)

    model_path = ensure_model()
    if not model_path:
        return

    import mediapipe as mp
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    def make_landmarker(gpu):
        base = BaseOptions(model_asset_path=model_path,
                           delegate=BaseOptions.Delegate.GPU if gpu else BaseOptions.Delegate.CPU)
        return FaceLandmarker.create_from_options(FaceLandmarkerOptions(
            base_options=base, running_mode=RunningMode.VIDEO, num_faces=1,
            min_face_detection_confidence=0.6, min_tracking_confidence=0.6,
            output_face_blendshapes=True))

    try:
        landmarker = make_landmarker(cfg["use_gpu"])
    except Exception as e:
        if cfg["use_gpu"]:
            log.warning(f"GPU delegate unavailable ({e}), falling back to CPU")
            landmarker = make_landmarker(False)
        else:
            raise

    W, H, DW, DH, OX, OY = get_screen_layout(cfg)
    pointer = VirtualPointer(DW, DH, OX, OY)
    out = OutputThread(pointer, cfg["output_hz"], cfg["interpolate"])
    ctrl = Controller(cfg, pointer, out, W, H, profile)
    out.start()
    cam = CameraThread(cfg)
    if cam.failed:
        print("ERROR: could not open the camera (check camera_index, or whether another app uses it).")
        out.stop()
        pointer.close()
        return
    cam.start()

    idx = cfg["track_landmarks"]
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(0))

    lock_line = ("  Lock / unlock : Smile and hold (~%.1f s), like a switch\n" % cfg["lock_hold_s"]
                 if cfg["smile_lock_enabled"] else
                 "  Lock / unlock : (smile lock disabled; press p in the preview window)\n")
    scroll_line = ("  Scroll        : Raise eyebrows and hold + tilt head up/down (cursor freezes)\n"
                   if cfg["scroll_enabled"] else "  Scroll        : (disabled)\n")
    print(f"\n{APP_NAME} Head Mouse v{VERSION}\n"
          "  Cursor        : Move your head\n"
          "  Single click  : Open your mouth and close it quickly\n"
          "  Double click  : Open and close your mouth twice, quickly\n"
          "  Drag          : Open your mouth and keep it open (closing releases)\n"
          "  Right click   : Pucker your lips\n"
          + scroll_line + lock_line +
          "  q quit | r re-center | g gesture wizard | p lock/unlock | d toggle dwell\n")

    start = time.monotonic()
    last_id, last_ts, last_t = 0, -1, None
    fps = 0.0
    interval = 1 / 30
    try:
        while True:
            got = cam.wait_new(last_id, 0.2)
            if cam.failed:
                print("Camera stream lost.")
                break
            if got is None:
                continue
            last_id, frame, t = got
            if last_t is not None:
                dt = max(t - last_t, 1e-3)
                interval = 0.9 * interval + 0.1 * dt
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            last_t = t

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts = int((t - start) * 1000)
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

            if result.face_landmarks:
                lm = result.face_landmarks[0]
                pts = np.array([[lm[i].x * w, lm[i].y * h] for i in idx], dtype=np.float64)
                scores = extract_scores(result.face_blendshapes[0] if result.face_blendshapes else None)
                scores["mouth"] = lip_ratio(lm, w, h)
                if cfg["show_preview"]:
                    m_on = scores["mouth"] > ctrl.gest["mouth"].on
                    cv2.line(frame, (int(lm[13].x * w), int(lm[13].y * h)),
                             (int(lm[14].x * w), int(lm[14].y * h)),
                             (0, 255, 0) if m_on else (0, 0, 255), 3)
                ctrl.on_face(t, gray, pts, head_proxies(lm, w, h), scores, interval)
            else:
                ctrl.on_no_face(t)

            if cfg["show_preview"]:
                if ctrl.tracker.pos is not None:
                    px, py = int(ctrl.tracker.pos[0]), int(ctrl.tracker.pos[1])
                    cv2.circle(frame, (px, py), 5, (255, 100, 0), -1)
                    if ctrl.dwell.progress > 0:
                        cv2.ellipse(frame, (px, py), (18, 18), 0, -90,
                                    -90 + int(360 * ctrl.dwell.progress), (0, 255, 0), 3)
                if ctrl.locked:
                    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)   # red border = locked
                text, col = ctrl.status_text(t)
                s = ctrl.scores
                lines = [
                    (text, col),
                    (f"Mouse: {'LOCKED' if ctrl.locked else 'ACTIVE'} | Drag: {'ON' if ctrl.drag_on else 'off'}"
                     f" | Dwell: {'ON' if ctrl.dwell_enabled else 'off'} | FPS {fps:.0f}", (255, 255, 255)),
                    (f"mouth {s['mouth']:.3f}/{ctrl.gest['mouth'].on:.3f}  smile {s['smile']:.2f}  "
                     f"brow {s['brow']:.2f}  pucker {s['pucker']:.2f}  gain {ctrl.last_gain:.1f}", (200, 255, 200)),
                ]
                frame = np.vstack([frame, render_status_bar(lines, w)])
                cv2.imshow(f"{APP_NAME} Head Mouse", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    ctrl.request_recal(False)
                elif key == ord("g"):
                    ctrl.request_recal(True)
                elif key == ord("p"):
                    ctrl.toggle_lock(t)
                elif key == ord("d"):
                    ctrl.dwell_enabled = not ctrl.dwell_enabled
                    ctrl.dwell.reset()
    except KeyboardInterrupt:
        pass
    finally:
        # Safety: always release held buttons and shut the threads down
        try:
            ctrl.release_all()
        except Exception:
            pass
        out.stop()
        cam.stop()
        pointer.close()
        try:
            landmarker.close()
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
