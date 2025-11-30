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

        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self.root, textvariable=self.status_var)
        self.status_label.grid(row=4, column=0, columnspan=4)

        # For displaying the latest frame
        self.photo_image = None

        # Background storage (temporary)
        self.background = None
        self.bg_running = False
        self.bg_countdown_end = 0.0
        self.bg_burst_count = 10
        self.bg_captured = False

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

    def capture_reference_and_start_game(self):
        """Capture a reference (O-photo), compute its mask, display it, and enable game mode."""
        # Capture a small burst and average to reduce noise
        n = 3
        acc = None
        captured = 0
        for i in range(n):
            r, f = self.cap.read()
            if not r or f is None:
                time.sleep(0.05)
                continue
            if acc is None:
                acc = f.astype(np.float32)
            else:
                acc += f.astype(np.float32)
            captured += 1
            time.sleep(0.05)

        if captured == 0:
            raise RuntimeError("Failed to capture reference frame")

        ref = (acc / float(captured)).astype(np.uint8)
        # Save reference image as O-photo.jpg (avoid overwriting existing)
        ref_name = 'O-photo.jpg'
        if os.path.exists(ref_name):
            i = 1
            base, ext = os.path.splitext(ref_name)
            while True:
                candidate = f"{base}_{i}{ext}"
                if not os.path.exists(candidate):
                    ref_name = candidate
                    break
                i += 1
        cv2.imwrite(ref_name, ref)

        # compute mask and save files via _process_subtraction
        mask, fg_name, mask_name = self._process_subtraction(ref, ref_name)
        # store target mask for scoring
        self.target_mask = mask
        self.game_active = True
        self.target_ref_name = ref_name
        self.target_mask_name = mask_name
        # display the reference and mask to the user
        self._show_target_window(ref, mask)

    def choose_existing_reference(self):
        """Let the user pick an existing O-photo image and load its mask if available.

        If a mask file is not found, ask whether to compute a mask from the background.
        """
        fn = filedialog.askopenfilename(title="Select O-photo image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff" )], initialdir='.')
        if not fn:
            self.status_var.set("No reference chosen")
            return

        # try to find mask candidates
        base, ext = os.path.splitext(fn)
        candidates = [f"{base}_mask.png", f"{base}_mask.jpg", f"{base}_mask.jpeg", f"{base}_mask.bmp", f"{base}_mask.tif"]
        mask_path = None
        for c in candidates:
            if os.path.exists(c):
                mask_path = c
                break

        # load reference image
        ref = cv2.imread(fn)
        if ref is None:
            raise RuntimeError("Failed to read selected reference image")

        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                # fallback to computing mask
                compute = messagebox.askyesno("Mask not readable", "Found mask file but couldn't read it. Compute mask from background instead?")
                if not compute:
                    self.status_var.set("No usable mask found")
                    return
                mask, _, _ = self._process_subtraction(ref, fn)
        else:
            # no mask file found
            compute = messagebox.askyesno("No mask found", "No mask file found for selected O-photo. Compute mask from background? (Requires background to be set)")
            if not compute:
                self.status_var.set("No mask available for reference")
                return
            if self.background is None:
                raise RuntimeError("Background not set; cannot compute mask for reference")
            mask, _, _ = self._process_subtraction(ref, fn)

        # store and activate game
        self.target_mask = mask
        self.game_active = True
        self.target_ref_name = fn
        self.target_mask_name = mask_path if mask_path else f"{os.path.splitext(fn)[0]}_mask.png"
        self._show_target_window(ref, mask)

    def choose_existing_reference_preselected(self, path: str):
        """Load a specific existing O-photo (path) and its mask if available.

        This is like `choose_existing_reference` but accepts a preselected path.
        """
        if not path or not os.path.exists(path):
            raise RuntimeError("Provided reference path does not exist")

        fn = path
        # try to find mask candidates
        base, ext = os.path.splitext(fn)
        candidates = [f"{base}_mask.png", f"{base}_mask.jpg", f"{base}_mask.jpeg", f"{base}_mask.bmp", f"{base}_mask.tif"]
        mask_path = None
        for c in candidates:
            if os.path.exists(c):
                mask_path = c
                break

        ref = cv2.imread(fn)
        if ref is None:
            raise RuntimeError("Failed to read reference image")

        if mask_path and os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                # compute if unreadable
                mask, _, _ = self._process_subtraction(ref, fn)
        else:
            if self.background is None:
                raise RuntimeError("Background not set; cannot compute mask for reference")
            mask, _, _ = self._process_subtraction(ref, fn)

        self.target_mask = mask
        self.game_active = True
        self.target_ref_name = fn
        self.target_mask_name = mask_path if mask_path else f"{base}_mask.png"
        self._show_target_window(ref, mask)

    def _show_target_window(self, image, mask):
        """Show a small window with the reference image and overlayed mask outline."""
        try:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            # create overlay red where mask>0
            mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
            mask_rgb[mask > 0] = (255, 0, 0)
            pil_mask = Image.fromarray(mask_rgb)
            overlay = Image.blend(pil_img, pil_mask, alpha=0.4)

            win = tk.Toplevel(self.root)
            win.title('Target (O-photo)')
            tk_img = ImageTk.PhotoImage(overlay)
            lbl = ttk.Label(win, image=tk_img)
            lbl.image = tk_img
            lbl.pack()
            ttk.Label(win, text='Match this outline and then press Start to capture').pack()
        except Exception as e:
            self.status_var.set(f"Failed to show target: {e}")

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
                            self.status_var.set("Background captured")
                            # After background is captured, offer to play
                            play = messagebox.askyesno("Play?", "Background captured. Would you like to play the matching game?")
                            if play:
                                # Auto-load O-photo if present in working directory
                                candidates = ['O-photo.jpg', 'O-photo.jpeg', 'O-photo.png', 'O-photo.bmp', 'O-photo.tif']
                                found = None
                                for c in candidates:
                                    if os.path.exists(c):
                                        found = c
                                        break
                                if found:
                                    try:
                                        # Load the existing reference directly
                                        self.choose_existing_reference_preselected(found)
                                    except Exception as e:
                                        self.status_var.set(f"Failed to load O-photo: {e}")
                                else:
                                    # No auto O-photo found: ask user whether to pick or capture
                                    use_existing = messagebox.askyesno("Reference", "No O-photo found automatically. Pick an existing file? (Yes = choose file, No = capture new reference)")
                                    if use_existing:
                                        try:
                                            self.choose_existing_reference()
                                        except Exception as e:
                                            self.status_var.set(f"Failed to load existing reference: {e}")
                                    else:
                                        try:
                                            self.capture_reference_and_start_game()
                                        except Exception as e:
                                            self.status_var.set(f"Failed to start game: {e}")
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
                                                        messagebox.showinfo("Score", f"Match score: {score:.1f}%")
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
