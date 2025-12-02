#!/usr/bin/env python3
"""take_timer_photo.py

Cross-platform script to take a timed photo from the user's webcam.

Usage examples:
  python take_timer_photo.py --timer 5 --output out.jpg
  python take_timer_photo.py --timer 10 --camera 1 --no-window

Works on Windows and macOS (attempts platform-appropriate OpenCV backends).
"""
from __future__ import annotations

import argparse
import math
import os
import platform
import sys
import time
from typing import Optional

import cv2
import numpy as np
from tkinter import messagebox, filedialog
import random
import re
import threading


def gaussian_kernel1d(sigma: float, radius: Optional[int] = None) -> np.ndarray:
    """Create a 1-D Gaussian kernel with given sigma. If radius is None use 3*sigma."""
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    if radius is None:
        radius = int(max(1, math.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    g = np.exp(-(x * x) / (2.0 * sigma * sigma))
    g /= g.sum()
    return g.astype(np.float32)


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply a separable Gaussian blur to a 2D or 3D image using numpy only.

    This uses np.convolve along rows and columns with a 1-D kernel.
    """
    if sigma <= 0 or image is None:
        return image.copy()

    kernel = gaussian_kernel1d(sigma)

    # work on float for accuracy
    arr = image.astype(np.float32)
    if arr.ndim == 2:
        # rows then cols
        tmp = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=1, arr=arr)
        out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=tmp)
        return np.clip(out, 0, 255).astype(image.dtype)

    # color image
    out = np.empty_like(arr)
    for c in range(arr.shape[2]):
        channel = arr[:, :, c]
        tmp = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=1, arr=channel)
        filtered = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=tmp)
        out[:, :, c] = filtered

    return np.clip(out, 0, 255).astype(image.dtype)


def absdiff_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute per-element absolute difference between two images using numpy."""
    # cast to int32 to avoid overflow then take abs
    ai = a.astype(np.int32)
    bi = b.astype(np.int32)
    diff = np.abs(ai - bi)
    return np.clip(diff, 0, 255).astype(np.uint8)

import tkinter as tk
from tkinter import ttk
try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None


def get_video_capture(index: int = 0) -> Optional[cv2.VideoCapture]:
    """Attempt to open a VideoCapture using platform-appropriate backends.

    Tries a small list of common backends for Windows, macOS (Darwin) and Linux.
    Returns an opened VideoCapture or None.
    """
    system = platform.system()
    backends = []
    if system == "Windows":
        # DirectShow and Media Foundation are common on Windows
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif system == "Darwin":
        # AVFoundation is commonly available on macOS builds of OpenCV
        backends = [getattr(cv2, 'CAP_AVFOUNDATION', cv2.CAP_ANY), cv2.CAP_ANY]
    else:
        # On Linux try V4L2 then any
        backends = [getattr(cv2, 'CAP_V4L2', cv2.CAP_ANY), cv2.CAP_ANY]

    for b in backends:
        try:
            cap = cv2.VideoCapture(index, b)
            if cap is None:
                continue
            if cap.isOpened():
                return cap
            cap.release()
        except Exception:
        
            continue
    try:
        cap = cv2.VideoCapture(index)
        if cap is not None and cap.isOpened():
            return cap
        if cap is not None:
            cap.release()
    except Exception:
        pass

    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Take a timed photo from the webcam")
    p.add_argument("--timer", "-t", type=float, default=3.0, help="Seconds to wait before taking the photo (default: 3)")
    p.add_argument("--output", "-o", type=str, default="photo.jpg", help="Output image path")
    p.add_argument("--camera", "-c", type=int, default=0, help="Camera index (default 0)")
    p.add_argument("--no-window", action="store_true", help="Do not show preview window")
    p.add_argument("--quiet", action="store_true", help="Minimal console output")
    p.add_argument("--gui", action="store_true", help="Open a simple GUI to take multiple pictures")
    return p.parse_args()


def overlay_countdown(frame, seconds_left: int) -> None:
    """Draw a large countdown number centered on the frame."""
    text = str(seconds_left)
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(2.0, min(w, h) / 300.0)
    thickness = max(2, int(scale * 2))
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (w - text_w) // 2
    y = (h + text_h) // 2
    # shadow
    cv2.putText(frame, text, (x + 2, y + 2), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, (0, 255, 0), thickness, cv2.LINE_AA)


def main() -> int:
    args = parse_args()

    if args.timer < 0:
        print("Timer must be non-negative", file=sys.stderr)
        return 2

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if args.gui:
        # Run the GUI app
        if Image is None or ImageTk is None:
            print("Pillow is required for GUI mode. Install with: pip install Pillow", file=sys.stderr)
            return 5

        if not args.quiet:
            print(f"Starting GUI and opening camera index {args.camera}...")

        app = CameraApp(camera_index=args.camera, quiet=args.quiet)
        return app.run()

    # CLI mode: open capture and perform single timed capture (original behavior)
    if not args.quiet:
        print(f"Opening camera index {args.camera}...")

    cap = get_video_capture(args.camera)
    if cap is None:
        print("ERROR: Could not open the camera. Check permissions and camera index.", file=sys.stderr)
        return 3

    try:

        warm_frames = 5
        for _ in range(warm_frames):
            cap.read()

        start = time.time()
        end = start + float(args.timer)
        last_seconds = math.ceil(end - time.time())

        captured_frame = None
        window_name = "Timer Camera"

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                # Try again briefly
                time.sleep(0.05)
                continue

            now = time.time()
            seconds_left = max(0, int(math.ceil(end - now)))

       
            if seconds_left > 0:
                overlay_countdown(frame, seconds_left)

            if not args.no_window:
                cv2.imshow(window_name, frame)

            if now >= end:
                captured_frame = frame.copy()
                if not args.quiet:
                    print("Capturing photo...")
                break

            # allow user to cancel with 'q'
            if not args.no_window:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    if not args.quiet:
                        print("Cancelled by user")
                    captured_frame = None
                    break
            else:
                # If no window, just sleep a bit to avoid busy loop
                time.sleep(0.01)

        if captured_frame is None:
            if not args.quiet:
                print("No photo captured.")
            return 0

        # Save image
        saved = cv2.imwrite(args.output, captured_frame)
        if saved:
            print(f"Saved photo to {args.output}")
        else:
            print(f"Failed to save photo to {args.output}", file=sys.stderr)
            return 4

    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    return 0


class CameraApp:
    """Simple Tkinter GUI around the webcam preview that supports multiple captures.

    UI elements:
    - Preview area showing camera frames
    - Entry for base filename
    - Spinboxes for count, timer, and interval
    - Start and Stop buttons
    - Status label
    """

    def __init__(self, camera_index: int = 0, quiet: bool = False):
        self.camera_index = camera_index
        self.quiet = quiet

        self.cap = get_video_capture(camera_index)
        if self.cap is None:
            raise RuntimeError("Could not open camera")

        self.root = tk.Tk()
        self.root.title("Timer Camera")

        # UI state for a single capture sequence
        self.sequence_running = False
        self.sequence_index = 0
        self.sequence_timer = 3.0
        self.sequence_basename = "photo"
        self.countdown_end_time = 0.0
        self.captured_this_step = False

        # Build UI
        self.preview_label = ttk.Label(self.root)
        self.preview_label.grid(row=0, column=0, columnspan=4, padx=8, pady=8)

        ttk.Label(self.root, text="Base name:").grid(row=1, column=0, sticky='e')
        self.name_var = tk.StringVar(value="photo")
        self.name_entry = ttk.Entry(self.root, textvariable=self.name_var, width=30)
        self.name_entry.grid(row=1, column=1, columnspan=2, sticky='w')
        # Background button
        self.bg_button = ttk.Button(self.root, text="Take background", command=self.take_background)
        self.bg_button.grid(row=1, column=3)

        ttk.Label(self.root, text="Timer (s):").grid(row=2, column=0, sticky='e')
        self.timer_var = tk.DoubleVar(value=3.0)
        self.timer_spin = ttk.Spinbox(self.root, from_=0, to=60, increment=0.5, textvariable=self.timer_var, width=6)
        self.timer_spin.grid(row=2, column=1, sticky='w')

        self.start_button = ttk.Button(self.root, text="Start", command=self.start_sequence)
        self.start_button.grid(row=2, column=2)
        self.stop_button = ttk.Button(self.root, text="Stop", command=self.stop_sequence, state='disabled')
        self.stop_button.grid(row=2, column=3)
        # MHI button: capture short burst and build a motion history image
        # MHI button is disabled until a background is captured
        self.mhi_button = ttk.Button(self.root, text="Take MHI", command=self.take_mhi, state='disabled')
        self.mhi_button.grid(row=2, column=4)

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self.root, textvariable=self.status_var)
        self.status_label.grid(row=4, column=0, columnspan=4)

        # For displaying the latest frame
        self.photo_image = None

        # Game score
        self.score_count = 0

        # Score label (top-right)
        self.score_var = tk.StringVar(value=f"Score: {self.score_count}")
        self.score_label = ttk.Label(self.root, textvariable=self.score_var, background='yellow')
        # place so it overlays top-right of window
        self.score_label.place(relx=0.98, rely=0.02, anchor='ne')

        # Background storage (temporary)
        self.background = None
        self.bg_running = False
        self.bg_countdown_end = 0.0
        self.bg_burst_count = 10
        self.bg_captured = False

        # MHI state
        self.mhi_running = False
        self.mhi_countdown_end = 0.0
        self.mhi_captured = False

        # Final-level MHI game state
        self.final_level_active = False
        self.final_mhi_target = None
        self.final_mhi_filename = None
        self.final_mhi_index = None

        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def run(self) -> int:
        try:
            self._update_frame()
            self.root.mainloop()
            return 0
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 6

    def _on_close(self):
        self.stop_sequence()
        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        self.root.destroy()

    def start_sequence(self):
        if self.sequence_running:
            return
        name = self.name_var.get().strip()
        if not name:
            self.status_var.set("Please enter a filename")
            return

        self.sequence_basename = name
        self.sequence_index = 1
        self.sequence_timer = max(0.0, float(self.timer_var.get()))
        self.sequence_running = True
        self.captured_this_step = False
        self.countdown_end_time = time.time() + self.sequence_timer
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.status_var.set("Starting countdown")

        # If final level is active, pressing Start should trigger the MHI countdown
        if getattr(self, 'final_level_active', False):
            # cancel the normal photo capture and start MHI countdown instead
            self.sequence_running = False
            self.captured_this_step = False
            self.countdown_end_time = 0.0
            self.stop_button.config(state='disabled')
            # start MHI countdown using same timer value
            self.mhi_running = True
            self.mhi_countdown_end = time.time() + float(self.timer_var.get())
            self.mhi_captured = False
            try:
                self.mhi_button.config(state='disabled')
            except Exception:
                pass
            self.status_var.set("Starting Final-Level MHI countdown...")

    def stop_sequence(self):
        if not self.sequence_running:
            return
        self.sequence_running = False
        self.countdown_end_time = 0.0
        self.captured_this_step = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.status_var.set("Stopped")

    def _save_frame(self, frame) -> bool:

        name = self.sequence_basename

        if not os.path.splitext(name)[1]:
            name = name + ".jpg"

        filename = name
        if os.path.exists(filename):
   
            i = 1
            base_no_ext = os.path.splitext(name)[0]
            ext = os.path.splitext(name)[1]
            while True:
                candidate = f"{base_no_ext}_{i}{ext}"
                if not os.path.exists(candidate):
                    filename = candidate
                    break
                i += 1

        try:
            saved = cv2.imwrite(filename, frame)
            if saved:
                self.status_var.set(f"Saved {filename}")
                return filename
            else:
                self.status_var.set(f"Failed to save {filename}")
                return None
        except Exception as e:
            self.status_var.set(f"Error saving {filename}: {e}")
            return None

    def take_background(self):

        if self.bg_running or self.sequence_running:
            self.status_var.set("Busy: finish current action first")
            return

        ok = messagebox.askokcancel("Take background", "Make sure you are outside the camera area. Click OK to start a 5s countdown.")
        if not ok:
            return

        self.bg_running = True
        self.bg_countdown_end = time.time() + 5.0
        self.bg_captured = False
        self.status_var.set("Starting background countdown...")

    def take_mhi(self):
        """Start capture of a short burst to build an MHI (60 frames over 2 seconds).

        Starts a GUI countdown (so the player can get into position). The actual
        capture runs in a background thread. Segmentation uses the stored
        background and the same processing pipeline as timer captures.
        """
        if self.bg_running or self.sequence_running:
            self.status_var.set("Busy: finish current action first")
            return

        if self.background is None:
            ask = messagebox.askyesno("Background required", "No background captured. Take background now?")
            if ask:
                # start background capture flow
                self.take_background()
            else:
                self.status_var.set("Background required for MHI")
            return

        # start a countdown using the shared timer value so the user can get in position
        self.mhi_running = True
        self.mhi_countdown_end = time.time() + float(self.timer_var.get())
        self.mhi_captured = False
        try:
            self.mhi_button.config(state='disabled')
        except Exception:
            pass
        self.status_var.set("Starting MHI countdown...")

    def _capture_mhi_worker(self):
        try:
            self.status_var.set("Capturing MHI burst...")
            # Target 60 frames over ~2 seconds to capture motion more fully
            n = 60
            duration = 2.0
            interval = duration / float(n)

            # warm camera with a few reads
            for _ in range(4):
                self.cap.read()

            frames = []
            start_ts = time.perf_counter()
            for i in range(n):
                # target time for this frame
                target = start_ts + i * interval
                r, f = self.cap.read()
                if not r or f is None:
                    if len(frames) > 0:
                        frames.append(frames[-1].copy())
                    else:
                        # create a placeholder matching expected frame size if possible
                        frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
                else:
                    frames.append(f.copy())

                # wait until the next target time (use perf_counter for accuracy)
                now = time.perf_counter()
                sleep_for = target + interval - now
                if sleep_for > 0:
                    time.sleep(sleep_for)

            # segmentation per frame
            masks = []
            kernel = np.ones((3, 3), np.uint8)
            if self.background is not None:
                # ensure size match
                bg = self.background
                if bg.shape[:2] != frames[0].shape[:2]:
                    bg = cv2.resize(bg, (frames[0].shape[1], frames[0].shape[0]))
                for f in frames:
                    diff = absdiff_numpy(f, bg)
                    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                    blur = gaussian_blur(gray, sigma=1.0)
                    _, mask = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
                    mask = cv2.dilate(mask, kernel, iterations=1)
                    masks.append(mask)
            else:
                # frame differencing
                prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
                masks.append(np.zeros_like(prev_gray))
                for f in frames[1:]:
                    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                    diff = cv2.absdiff(g, prev_gray)
                    blur = gaussian_blur(diff, sigma=1.0)
                    _, mask = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
                    mask = cv2.dilate(mask, kernel, iterations=1)
                    masks.append(mask)
                    prev_gray = g

            # construct motion history image (MHI)
            if not masks:
                raise RuntimeError("No masks generated for MHI")

            h, w = masks[0].shape[:2]
            mhi = np.zeros((h, w), dtype=np.float32)
            tau = float(len(masks))
            for mask in masks:
                motion = (mask > 0).astype(np.float32)
                # set recent motion to tau, decay older motion by 1
                mhi = np.where(motion > 0, tau, np.maximum(0.0, mhi - 1.0))

            # normalize to 0-255 uint8
            mhi_norm = np.clip((mhi / tau) * 255.0, 0, 255).astype(np.uint8)

            # save file
            base = self.sequence_basename or 'mhi'
            if not os.path.splitext(base)[1]:
                base = base + ""
            fname = f"{base}_mhi.png"
            if os.path.exists(fname):
                i = 1
                base_no_ext = os.path.splitext(fname)[0]
                while True:
                    candidate = f"{base_no_ext}_{i}.png"
                    if not os.path.exists(candidate):
                        fname = candidate
                        break
                    i += 1

            cv2.imwrite(fname, mhi_norm)
            # Schedule UI updates on the main thread so Tkinter calls are safe
            def _on_mhi_complete():
                try:
                    # If this is the final-level attempt, perform comparison to target
                    if getattr(self, 'final_level_active', False) and self.final_mhi_target is not None:
                        try:
                            target = self.final_mhi_target
                            _, tgt_bin = cv2.threshold(target, 127, 255, cv2.THRESH_BINARY)
                            _, att_bin = cv2.threshold(mhi_norm, 127, 255, cv2.THRESH_BINARY)
                            if tgt_bin.shape != att_bin.shape:
                                att_bin = cv2.resize(att_bin, (tgt_bin.shape[1], tgt_bin.shape[0]), interpolation=cv2.INTER_NEAREST)
                            inter = np.logical_and(tgt_bin > 0, att_bin > 0).sum()
                            union = np.logical_or(tgt_bin > 0, att_bin > 0).sum()
                            score = 0.0 if union == 0 else float(inter) / float(union) * 100.0
                            if score >= 65.0:
                                self.score_count += 2
                                self.score_var.set(f"Score: {self.score_count}")
                                self.final_level_active = False
                                self.status_var.set("Final level complete — you won!")
                                messagebox.showinfo("Final level", f"Final MHI match: {score:.1f}% — You earned 2 points!")
                            else:
                                messagebox.showinfo("Final level", f"Final MHI match: {score:.1f}% — below threshold, try again.")
                                try:
                                    self._select_and_show_final_mhi_target()
                                    self.status_var.set("Try the next final-level MHI target and press Start")
                                except Exception as e:
                                    self.final_level_active = False
                                    self.status_var.set(f"Failed to load next final target: {e}")
                                    messagebox.showerror("Final level error", f"Failed to load next final target: {e}")
                        except Exception as e:
                            self.status_var.set(f"Final-level scoring failed: {e}")
                            messagebox.showerror("Final level error", f"Scoring failed: {e}")
                    else:
                        self.status_var.set(f"MHI saved to {fname}")
                        messagebox.showinfo("MHI", f"Saved MHI to {fname}")

                finally:
                    # Re-enable the MHI button (allow multiple captures without retaking background)
                    try:
                        self.mhi_button.config(state='normal')
                    except Exception:
                        pass

            try:
                self.root.after(0, _on_mhi_complete)
            except Exception:
                # fallback if root isn't available; try direct calls
                _on_mhi_complete()
        except Exception as e:
            self.status_var.set(f"MHI capture failed: {e}")
            messagebox.showerror("MHI Error", f"Failed to create MHI: {e}")

    # Removed manual reference capture flow: the app now relies solely on
    # templates in `./templates` (match_me_#.png with corresponding masks).

    def start_template_game(self):
        """Pick a random template from ./templates and set it as the target for the game.

        Templates expected: `templates/match_me_#.png` and `templates/match_me_#_mask.png` (# in 0..5).
        """
        tpl_dir = os.path.join(os.getcwd(), 'templates')
        if not os.path.isdir(tpl_dir):
            raise RuntimeError("templates directory not found")

        # find match_me_#.png files
        files = os.listdir(tpl_dir)
        pattern = re.compile(r'^match_me_(\d+)\.jpg$', re.IGNORECASE)
        candidates = []
        for fn in files:
            m = pattern.match(fn)
            if m:
                candidates.append((int(m.group(1)), os.path.join(tpl_dir, fn)))

        if not candidates:
            raise RuntimeError("No match_me_#.png templates found in templates/")

        idx, chosen = random.choice(candidates)
        mask_name = os.path.join(tpl_dir, f"match_me_{idx}_mask.png")
        img = cv2.imread(chosen)
        if img is None:
            raise RuntimeError(f"Failed to read template {chosen}")
        if not os.path.exists(mask_name):
            raise RuntimeError(f"Mask not found for template {chosen}")

        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask {mask_name}")

        # ensure binary
        _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        self.target_mask = mask_bin
        self.game_active = True
        self.target_ref_name = chosen
        self.target_mask_name = mask_name
        self._show_target_window(img, mask_bin)

    def start_final_level(self):
        """Initialize and show the first final-level MHI target, and set final level active."""
        # verify background available
        if self.background is None:
            raise RuntimeError("Background must be set before starting final level")

        # mark final level active
        self.final_level_active = True
        # select and show target
        self._select_and_show_final_mhi_target()
        self.status_var.set("Final level started: press Start to capture MHI")

    # Removed choose_existing_reference: manual external reference selection
    # is no longer supported. The app uses templates in `./templates` only.

    # Removed choose_existing_reference_preselected: no longer using external
    # manual reference files.

    def _show_target_window(self, image, mask):
        """Show a small window with the reference image and overlayed mask outline (templates only)."""
        try:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            # create overlay red where mask>0
            mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            mask_rgb[mask > 0] = (255, 0, 0)
            pil_mask = Image.fromarray(mask_rgb)
            overlay = Image.blend(pil_img, pil_mask, alpha=0.4)

            win = tk.Toplevel(self.root)
            win.title('Target')
            tk_img = ImageTk.PhotoImage(overlay)
            lbl = ttk.Label(win, image=tk_img)
            lbl.image = tk_img
            lbl.pack()
            ttk.Label(win, text='Match this outline and then press Start to capture').pack()
        except Exception as e:
            self.status_var.set(f"Failed to show target: {e}")

    def _show_mhi_target_window(self, mhi_image: np.ndarray):
        """Show a small window with the target MHI image for the final level."""
        try:
            # mhi_image is grayscale 0-255; convert to RGB for display
            if mhi_image.ndim == 2:
                cmap = cv2.applyColorMap(mhi_image, cv2.COLORMAP_JET)
            else:
                # if already 3-channel
                cmap = mhi_image
            img_rgb = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)

            win = tk.Toplevel(self.root)
            win.title(f'Final MHI Target: {self.final_mhi_filename}')
            tk_img = ImageTk.PhotoImage(pil_img)
            lbl = ttk.Label(win, image=tk_img)
            lbl.image = tk_img
            lbl.pack()
            ttk.Label(win, text='Try to reproduce this motion history. Press Start when ready.').pack()
        except Exception as e:
            self.status_var.set(f"Failed to show MHI target: {e}")

    def _select_and_show_final_mhi_target(self):
        """Randomly pick photo_mhi_0.png or photo_mhi_1.png (search cwd and templates/) and show it."""
        candidates = []
        for base in ['photo_mhi_0.png', 'photo_mhi_1.png']:
            if os.path.exists(base):
                candidates.append(base)
        tpl_dir = os.path.join(os.getcwd(), 'templates')
        if os.path.isdir(tpl_dir):
            for base in ['photo_mhi_0.png', 'photo_mhi_1.png']:
                p = os.path.join(tpl_dir, base)
                if os.path.exists(p):
                    candidates.append(p)

        if not candidates:
            raise RuntimeError("No final-level MHI files found (photo_mhi_0.png or photo_mhi_1.png)")

        chosen = random.choice(candidates)
        mhi = cv2.imread(chosen, cv2.IMREAD_GRAYSCALE)
        if mhi is None:
            raise RuntimeError(f"Failed to read final-level MHI file {chosen}")

        self.final_mhi_target = mhi
        self.final_mhi_filename = os.path.basename(chosen)
        # set index 0 or 1 when possible
        m = re.search(r'photo_mhi_(\d)\.png$', self.final_mhi_filename)
        if m:
            try:
                self.final_mhi_index = int(m.group(1))
            except Exception:
                self.final_mhi_index = None
        else:
            self.final_mhi_index = None

        # show target to the player
        self._show_mhi_target_window(mhi)

    def _compare_masks(self, target_mask: np.ndarray, attempt_mask: np.ndarray) -> float:
        """Compute IoU between target and attempt masks after centroid alignment.

        Steps:
        - Convert masks to binary uint8 (0/1).
        - Find the largest connected component in each mask and compute its centroid.
        - Translate the attempt mask so its centroid aligns with the target centroid (integer shift, zero-filled).
        - Compute IoU (intersection / union) on the aligned masks and return percentage.
        """
        # Ensure binary masks
        tgt = (target_mask > 0).astype(np.uint8)
        att = (attempt_mask > 0).astype(np.uint8)

        # Resize attempt if shapes differ (resize before centroid calc)
        if tgt.shape != att.shape:
            att = cv2.resize(att, (tgt.shape[1], tgt.shape[0]), interpolation=cv2.INTER_NEAREST)

        def largest_component_centroid(bin_mask: np.ndarray):
            # connected components: background=0
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
            if num_labels <= 1:
                return None  # no foreground
            # skip label 0 (background); find label with max area
            areas = stats[1:, cv2.CC_STAT_AREA]
            max_idx = int(np.argmax(areas)) + 1
            cx, cy = centroids[max_idx]
            return (float(cx), float(cy), int(max_idx))

        tgt_cent = largest_component_centroid(tgt)
        att_cent = largest_component_centroid(att)

        if tgt_cent is None or att_cent is None:
            # fallback to pixel-wise IoU if either has no components
            inter = np.logical_and(tgt, att).sum()
            union = np.logical_or(tgt, att).sum()
            if union == 0:
                return 0.0
            return float(inter) / float(union) * 100.0

        tgt_cx, tgt_cy, _ = tgt_cent
        att_cx, att_cy, att_label = att_cent

        # compute integer shift (dx in cols, dy in rows) needed to align centroids
        dx = int(round(tgt_cx - att_cx))
        dy = int(round(tgt_cy - att_cy))

        # translate attempt mask by (dx, dy) with zero padding (no wrap)
        h, w = tgt.shape[:2]
        shifted = np.zeros_like(att)
        # source coordinates
        src_x0 = max(0, -dx)
        src_x1 = min(w, w - dx)  # exclusive
        src_y0 = max(0, -dy)
        src_y1 = min(h, h - dy)
        # dest coordinates
        dst_x0 = max(0, dx)
        dst_x1 = min(w, w + dx)
        dst_y0 = max(0, dy)
        dst_y1 = min(h, h + dy)

        # validate ranges and copy
        if src_x1 > src_x0 and src_y1 > src_y0 and dst_x1 > dst_x0 and dst_y1 > dst_y0:
            shifted[dst_y0:dst_y1, dst_x0:dst_x1] = att[src_y0:src_y1, src_x0:src_x1]

        # compute IoU on tgt vs shifted
        inter = np.logical_and(tgt.astype(bool), shifted.astype(bool)).sum()
        union = np.logical_or(tgt.astype(bool), shifted.astype(bool)).sum()
        if union == 0:
            return 0.0
        iou = float(inter) / float(union)
        return iou * 100.0

    def _update_frame(self):
        ret, frame = self.cap.read()
        if ret and frame is not None:
            now = time.time()
   
            if self.bg_running:
                seconds_left = max(0, int(math.ceil(self.bg_countdown_end - now)))
                if seconds_left > 0:
                    overlay_countdown(frame, seconds_left)
                else:
                    if not self.bg_captured:
               
                        self.status_var.set("Capturing background burst...")
                        n = max(3, int(self.bg_burst_count))
                        acc = np.zeros_like(frame, dtype=np.float32)
                        captured = 0
                        for i in range(n):
                            r, f = self.cap.read()
                            if not r or f is None:
                                time.sleep(0.02)
                                continue
                            acc += f.astype(np.float32)
                            captured += 1
                            time.sleep(0.03)
                        if captured > 0:
                            avg = (acc / float(captured)).astype(np.uint8)
                            self.background = avg
                            self.bg_captured = True
                            self.bg_running = False
                            # Enable MHI button now that a background exists
                            try:
                                self.mhi_button.config(state='normal')
                            except Exception:
                                pass
                            self.status_var.set("Background captured")
                            # After background is captured, offer to play
                            play = messagebox.askyesno("Play?", "Background captured. Would you like to play the matching game?")
                            if play:
                                # Use templates only: look for ./templates/match_me_#.png
                                tpl_dir = os.path.join(os.getcwd(), 'templates')
                                if os.path.isdir(tpl_dir):
                                    try:
                                        self.start_template_game()
                                    except Exception as e:
                                        self.status_var.set(f"Failed to start template game: {e}")
                                        messagebox.showerror("Template Error", f"Failed to start template game: {e}")
                                else:
                                    # Do not fallback to external reference files. Inform the user.
                                    self.status_var.set("No templates directory found (./templates)")
                                    messagebox.showerror("Templates missing", "No templates found in ./templates. Add 'match_me_#.png' and 'match_me_#_mask.png' files to play the matching game.")
                            else:
                                # ask to retake or quit
                                retake = messagebox.askyesno("Background set", "Do you want to retake the background? (Yes = retake, No = keep background)")
                                if retake:
                                    # start background capture again
                                    self.take_background()
                        else:
                            self.status_var.set("Failed to capture background")
                            self.bg_running = False
                            self.bg_captured = False


            if self.mhi_running:
                seconds_left = max(0, int(math.ceil(self.mhi_countdown_end - now)))
                if seconds_left > 0:
                    overlay_countdown(frame, seconds_left)
                else:
                    if not self.mhi_captured:
                        # start MHI worker in background
                        try:
                            threading.Thread(target=self._capture_mhi_worker, daemon=True).start()
                        except Exception:
                            pass
                        self.mhi_captured = True
                        self.mhi_running = False

            if self.sequence_running:
                # countdown in progress for a single capture
                seconds_left = max(0, int(math.ceil(self.countdown_end_time - now)))
                if seconds_left > 0:
                    overlay_countdown(frame, seconds_left)
                else:
                    if not self.captured_this_step:
                        self.status_var.set("Capturing...")
                        saved = self._save_frame(frame.copy())
                        self.captured_this_step = True
                        self.sequence_running = False
                        self.start_button.config(state='normal')
                        self.stop_button.config(state='disabled')
                        if saved and self.background is not None:
                            # perform subtraction and save results
                            try:
                                                mask, fg_name, mask_name = self._process_subtraction(frame.copy(), saved)
                                                self.status_var.set("Capture and subtraction complete")
                                                # If in game mode, compare masks and show score
                                                if getattr(self, 'game_active', False) and hasattr(self, 'target_mask') and self.target_mask is not None:
                                                    try:
                                                        score = self._compare_masks(self.target_mask, mask)
                                                        awarded = False
                                                        # award point if >=75%
                                                        if score >= 70.0:
                                                            self.score_count += 1
                                                            self.score_var.set(f"Score: {self.score_count}")
                                                            awarded = True
                                                            messagebox.showinfo("Score", f"Match score: {score:.1f}% — Good! Point awarded.")
                                                        else:
                                                            messagebox.showinfo("Score", f"Match score: {score:.1f}%")

                                                        # Check for game end (3 points)
                                                        if self.score_count >= 3:
                                                            # Start final MHI level instead of immediate congratulations
                                                            self.game_active = False
                                                            try:
                                                                cont = messagebox.askokcancel("Final level", "Final level: Match the MHI\nAn MHI will be captured over a 2 second interval. Press OK to continue.")
                                                            except Exception:
                                                                cont = True
                                                            if cont:
                                                                # initialize final level
                                                                try:
                                                                    self.start_final_level()
                                                                except Exception as e:
                                                                    self.status_var.set(f"Failed to start final level: {e}")
                                                                    messagebox.showerror("Final level error", f"Failed to start final level: {e}")
                                                            else:
                                                                # user cancelled; treat as game complete
                                                                self.final_level_active = False
                                                                self.status_var.set("Game complete")
                                                        else:
                                                            # Provide a new target template for the next attempt
                                                            try:
                                                                self.start_template_game()
                                                                self.status_var.set("New target loaded. Ready for next attempt.")
                                                            except Exception as e:
                                                                # If templates fail to load, end game and show error
                                                                self.game_active = False
                                                                self.status_var.set(f"Failed to load next template: {e}")
                                                                messagebox.showerror("Template Error", f"Failed to load next template: {e}")

                                                    except Exception as e:
                                                        self.status_var.set(f"Scoring failed: {e}")
                            except Exception as e:
                                self.status_var.set(f"Subtraction failed: {e}")
                        else:
                            self.status_var.set("Capture complete")

            # Convert BGR to RGB for Tkinter display
            try:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=im_pil)
                self.photo_image = imgtk
                self.preview_label.config(image=imgtk)
            except Exception:
                # If display conversion fails, ignore and continue
                pass

        # schedule next frame update
        self.root.after(30, self._update_frame)

    def _process_subtraction(self, frame, saved_filename: str) -> None:
        """Perform background subtraction using the stored background and save the masked region and mask."""
        if self.background is None:
            raise RuntimeError("No background set")

        # Ensure same size
        bg = self.background
        if bg.shape[:2] != frame.shape[:2]:
            # resize background to match frame
            bg = cv2.resize(bg, (frame.shape[1], frame.shape[0]))

        # Use numpy-based absolute difference and gaussian blur
        diff = absdiff_numpy(frame, bg)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        # apply small gaussian blur implemented above
        blur = gaussian_blur(gray, sigma=1.5)
        _, mask = cv2.threshold(blur, 30, 255, cv2.THRESH_BINARY)

        # clean small noise
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        masked_color = cv2.bitwise_and(frame, frame, mask=mask)

        base = os.path.splitext(saved_filename)[0]
        fg_name = f"{base}_fg.png"
        mask_name = f"{base}_mask.png"

        cv2.imwrite(fg_name, masked_color)
        cv2.imwrite(mask_name, mask)
        # return mask and filenames for further processing
        return mask, fg_name, mask_name


if __name__ == '__main__':
    raise SystemExit(main())
