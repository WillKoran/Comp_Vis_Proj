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
            # Some builds may not support specific backend constants
            continue

    # Fallback: try opening without explicit backend
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
        # Warm up a little
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

            # Overlay countdown only while > 0
            if seconds_left > 0:
                overlay_countdown(frame, seconds_left)

            if not args.no_window:
                cv2.imshow(window_name, frame)

            # If time's up capture and break
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
        self.name_entry.grid(row=1, column=1, columnspan=3, sticky='w')

        ttk.Label(self.root, text="Timer (s):").grid(row=2, column=0, sticky='e')
        self.timer_var = tk.DoubleVar(value=3.0)
        self.timer_spin = ttk.Spinbox(self.root, from_=0, to=60, increment=0.5, textvariable=self.timer_var, width=6)
        self.timer_spin.grid(row=2, column=1, sticky='w')

        self.start_button = ttk.Button(self.root, text="Start", command=self.start_sequence)
        self.start_button.grid(row=2, column=2)
        self.stop_button = ttk.Button(self.root, text="Stop", command=self.stop_sequence, state='disabled')
        self.stop_button.grid(row=2, column=3)

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self.root, textvariable=self.status_var)
        self.status_label.grid(row=4, column=0, columnspan=4)

        # For displaying the latest frame
        self.photo_image = None

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
        # Use the user-provided name. Append .jpg if missing. If file exists, append suffix.
        name = self.sequence_basename
        # ensure extension
        if not os.path.splitext(name)[1]:
            name = name + ".jpg"

        filename = name
        if os.path.exists(filename):
            # find a non-colliding name
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
            else:
                self.status_var.set(f"Failed to save {filename}")
            return saved
        except Exception as e:
            self.status_var.set(f"Error saving {filename}: {e}")
            return False

    def _update_frame(self):
        ret, frame = self.cap.read()
        if ret and frame is not None:
            now = time.time()
            # Handle sequence/countdown state
            if self.sequence_running:
                # countdown in progress for a single capture
                seconds_left = max(0, int(math.ceil(self.countdown_end_time - now)))
                if seconds_left > 0:
                    overlay_countdown(frame, seconds_left)
                else:
                    if not self.captured_this_step:
                        self.status_var.set("Capturing...")
                        self._save_frame(frame.copy())
                        self.captured_this_step = True
                        self.sequence_running = False
                        self.start_button.config(state='normal')
                        self.stop_button.config(state='disabled')
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


if __name__ == '__main__':
    raise SystemExit(main())
