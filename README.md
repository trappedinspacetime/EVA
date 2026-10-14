# EVA Head Mouse

Hands-free mouse control for Linux using a webcam: **move your head to move the cursor, use facial gestures to click, drag, scroll and lock/unlock.**

It runs entirely locally. A webcam image goes through [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker), the head motion is tracked and filtered, and the result is sent to the system through a virtual `uinput` mouse. No data leaves your computer.

> **Status:** a personal assistive-technology project, shared as-is. It is **not** a medical device and comes with no warranty. It was developed and used on Linux with a regular webcam. Other setups (Wayland, multi-monitor, unusual cameras) are not fully verified, so please report what you find.

## Features

- **Relative head control with an acceleration curve** – slow movements are precise, fast movements cover the screen. Pushing past the screen edge keeps the cursor on the edge, so the cursor re-centers itself naturally.
- **Stable and light-weight tracking** – multi-point Lucas-Kanade tracking of nose landmarks (with outlier rejection and drift anchoring) plus a single One Euro filter. No sticky dead zones.
- **Smooth output** – a separate output thread (120 Hz by default) interpolates between camera frames.
- **Gesture control** with per-user calibration, hysteresis and a freeze-on-gesture mechanism, so forming a gesture does not move the cursor.
- **Lock switch** – smile and hold to lock the mouse, smile again to unlock.
- **Dwell click** (optional) – hold the cursor still to click.
- **Edge nudge** – makes screen-edge triggers (auto-hide docks, hot corners) work even though the main pointer is an absolute device.
- **Safety** – held buttons are released when the face is lost, when locking, on exit and on errors.

## Gestures

| Gesture | Action |
|---|---|
| Move your head | Move the cursor |
| Open your mouth and close it quickly | Single click |
| Open and close your mouth twice, quickly | Double click |
| Open your mouth and keep it open (~0.45 s) | Drag (released when you close your mouth) |
| Pucker your lips | Right click |
| Raise your eyebrows and hold, then tilt your head up/down | Scroll (cursor freezes while scrolling) |
| Smile and hold (~0.5 s) | **Lock / unlock** the mouse (toggle) |
| Keep the cursor still (~1 s) | Dwell click (can be disabled) |

While the mouse is locked, only the smile gesture is active. The preview window shows a red border and the status line says `LOCKED`.

Keyboard shortcuts (the preview window must be focused): `q` quit · `r` re-center · `g` gesture setup wizard · `p` lock/unlock · `d` toggle dwell click.

## Requirements

- Linux with the `uinput` kernel module
- Python 3.9–3.12 recommended (limited by MediaPipe wheel availability)
- A webcam and reasonably even lighting (a lamp in front of you helps a lot)
- A GUI-enabled OpenCV (`opencv-python`, not the `-headless` variant) if you want the preview window

## Installation

```bash
git clone https://github.com/trappedinspacetime/EVA.git
cd EVA
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`evdev` may need build tools on some systems: `sudo apt install build-essential python3-dev`.

The face model (`face_landmarker.task`, ~4 MB) is downloaded automatically on first start. If you are offline, download it manually from
<https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task>
and place it next to `eva_headmouse.py`.

### Permission to create a virtual mouse (uinput)

The script needs write access to `/dev/uinput`. The quick way is to run it with `sudo`. The cleaner way is a udev rule:

```bash
echo 'KERNEL=="uinput", MODE="0660", GROUP="input", OPTIONS+="static_node=uinput"' | sudo tee /etc/udev/rules.d/99-uinput.rules
echo uinput | sudo tee /etc/modules-load.d/uinput.conf
sudo usermod -aG input $USER
sudo modprobe uinput
sudo udevadm control --reload-rules && sudo udevadm trigger
# log out and back in so the group change takes effect
```

If `uinput` cannot be opened, the program starts in **dry-run mode**: the mouse does not move, but the preview and calibration still work. This is handy for a first test.

## Usage

```bash
python eva_headmouse.py
```

Options:

| Option | Meaning |
|---|---|
| `--config PATH` | use a different JSON config file |
| `--profile PATH` | use a different gesture profile file |
| `--no-preview` | do not open the preview window |
| `--wizard` | run the gesture setup wizard at startup |

### First start

1. **Head calibration** – keep your head still for ~2 seconds.
2. **Gesture wizard** (only if there is no saved profile) – the status line asks you to rest, open your mouth slightly, smile, raise your eyebrows and pucker your lips, 2.5 seconds each. The thresholds are stored in `eva_profile.json` and reused on the next start. Press `g` in the preview window to run it again.

The line under the camera image shows live gesture values, for example `mouth 0.031/0.040`: current value / trigger threshold. A line between your lips turns green when the mouth gesture is active.

## Configuration

Create an `eva_config.json` next to the script (see `eva_config.example.json`). Values you set there override the defaults in `DEFAULT_CONFIG`; everything else keeps its default.

| Key | Default | Purpose |
|---|---|---|
| `sens_x`, `sens_y` | `2.0` | Overall cursor travel per head movement. Raise if you cannot reach the screen edges comfortably |
| `gain_min`, `gain_max` | `7.0`, `32.0` | Gain for slow / fast head movement. Lower `gain_min` for finer precision |
| `min_cutoff`, `beta` | `1.2`, `0.08` | Filter: lower `min_cutoff` = less jitter, more lag. Higher `beta` = less lag while moving |
| `gestures.mouth.on` / `.off` | `0.04` / `0.025` | Mouth trigger/release thresholds (inner-lip distance ÷ face height). Lower = more sensitive |
| `drag_hold_s` | `0.45` | Mouth open longer than this = drag instead of click |
| `click_freeze_s` | `0.35` | Cursor stays still this long after a click (keeps double clicks on target) |
| `smile_lock_enabled` | `true` | Smile-to-lock switch on/off |
| `lock_hold_s` | `0.5` | How long to hold the smile to toggle the lock |
| `scroll_enabled` | `true` | Eyebrow scroll on/off. When off, eyebrows are ignored completely |
| `scroll_invert` | `false` | Flip the scroll direction |
| `dwell_enabled`, `dwell_s`, `dwell_radius_px` | `true`, `1.0`, `20` | Dwell clicking |
| `edge_nudge`, `edge_nudge_px` | `true`, `3` | Pressure against screen edges for auto-hide docks etc. |
| `tilt_influence` | `0.0` | Experimental drift correction from head orientation (0 = off, try `0.2`) |
| `camera_index` | `0` | Which camera to use |
| `monitor_index` | `0` | Which monitor the cursor should cover (needs `screeninfo`) |
| `output_hz` | `120` | Pointer update rate |
| `use_gpu` | `false` | Try the MediaPipe GPU delegate (falls back to CPU) |

Disabled gestures are ignored completely and their steps are skipped in the setup wizard.

Individual gesture thresholds can be overridden too, for example:

```json
{ "gestures": { "mouth": { "on": 0.03, "off": 0.018 } } }
```

## Tuning tips

| Problem | Try |
|---|---|
| Cursor jitters | lower `min_cutoff` (e.g. 0.9), lower `gain_min` (e.g. 5), improve lighting |
| Cursor feels heavy / laggy | raise `min_cutoff` and `beta` (e.g. 1.8 and 0.12) |
| Need too much head movement | raise `sens_x` / `sens_y` |
| Hard to hit small targets | lower `gain_min`, raise `rest_speed_hi` |
| Mouth clicks too sensitive / not sensitive enough | adjust `gestures.mouth.on/off` or re-run the wizard with `g` |
| Accidental clicks while talking | lock the mouse with a smile before talking |
| Accidental dwell clicks | raise `dwell_s` or disable with `d` |
| Auto-hide dock does not appear at the screen edge | keep `edge_nudge` on and raise `edge_nudge_px` (e.g. 6); some docks also have a "pressure reveal" setting |
| Low FPS in the status line | close other camera apps, use a lower `cam_w`/`cam_h`, try `use_gpu` |

## How it works (short)

1. **Capture** – a camera thread grabs frames (MJPG, buffer size 1) so the newest frame is always processed.
2. **Face landmarks and blendshapes** – MediaPipe Face Landmarker. Mouth opening is the inner-lip distance relative to face height; smile, eyebrow raise and lip pucker come from blendshape scores.
3. **Head motion** – a template of nose landmarks is tracked frame-to-frame with Lucas-Kanade optical flow. The mean motion of the inlier points is accumulated and gently anchored to the landmark centroid to prevent drift.
4. **Filtering and gain** – a One Euro filter smooths the position; the speed-dependent gain curve converts camera pixels to screen pixels.
5. **Gesture state machines** – hysteresis, optional hold times, cooldowns and a short "arming" freeze so gestures do not move the cursor.
6. **Output** – an absolute virtual pointer (`uinput`) written at a high rate with interpolation, plus a tiny relative device for the edge nudge.

## Known limitations

- Linux only (it relies on `uinput`).
- Mouth gestures can trigger while talking. Use the smile lock when you talk.
- A 30 fps camera limits how fast a gesture can be detected; very short gestures can fall between two frames.
- Multi-monitor support (absolute coordinates over the whole desktop) is implemented but lightly tested.
- The hybrid `tilt_influence` mode is experimental and off by default.

## Credits and inspiration

- [Tracky Mouse](https://github.com/1j01/tracky-mouse) by 1j01 – an excellent head-tracking mouse whose approach (relative motion, acceleration, multi-point tracking) inspired this project. Highly recommended, especially if you want a cross-platform app.
- [MediaPipe](https://github.com/google-ai-edge/mediapipe) by Google – face landmarks and blendshapes.
- [eViacam](https://eviacam.crea-si.com/) – the classic head-tracking mouse.
- Casiez, Roussel, Vogel: *1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems* (CHI 2012).

## AI assistance

This project was built by its author together with [Claude](https://www.anthropic.com/claude), an AI assistant made by Anthropic. The idea, the original script, the choice of gestures, all testing on real hardware and every decision about what to change came from the author. Claude's contributions:

- **Analysis** – reviewed the original script and diagnosed the causes of the jitter and the heavy, laggy feel (very high gain from absolute mapping, several stacked filters, a slow processing loop, and cursor drift with a relative virtual mouse).
- **Redesign and implementation** – relative head control with an acceleration curve, multi-point Lucas-Kanade tracking, a single dt-based One Euro filter, an absolute `uinput` pointer with a high-rate interpolating output thread, and a separate low-latency camera thread.
- **Interaction design** – gesture state machines (hysteresis, hold times, freeze-while-forming), the per-user calibration wizard, mouth tap / double-tap / drag detection, the smile lock switch, eyebrow scrolling, the edge nudge for auto-hide docks, and the safety releases.
- **Testing and fixes** – wrote automated checks that run the control logic on synthetic camera and gesture data, and fixed bugs found along the way (for example a startup race in the camera thread).
- **Documentation** – the English translation of the script and this README.

Claude had no access to a real camera or to the author's desktop, so all real-world behavior and tuning were verified by the author. As with any AI-assisted code, please read it and test it carefully before relying on it.

## License

Released under the [MIT License](LICENSE).

## Contributing

Issues and pull requests are welcome, especially for other desktop environments, cameras and accessibility needs. If you change a default, please describe the problem it solves.
