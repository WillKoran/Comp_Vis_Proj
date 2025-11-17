Requirements & Setup
====================

This document explains how to set up a Python environment for `take_timer_photo.py` on Windows and macOS.

Prerequisites
-------------
- Python 3.8+ (3.10+ recommended).
- A working system webcam.

Windows (PowerShell)
---------------------
1. Open PowerShell.
2. Create a virtual environment and activate it:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

3. Upgrade pip and install requirements:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4. Run the script:

```powershell
python take_timer_photo.py --timer 5 --output out.jpg
```

macOS
-----
1. Open Terminal.
2. (Optional) Install Python via Homebrew if your system Python is old:

```bash
brew install python
```

3. Create and activate a virtual environment:

```bash
python3 -m venv .venv; source .venv/bin/activate
```

4. Upgrade pip and install requirements:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

5. macOS camera permissions: the first time you run an app that uses the camera, macOS will ask for permission. If the camera doesn't show up:

- Open System Settings -> Privacy & Security -> Camera and ensure the Terminal (or the Python runtime) has permission.
- For GUI camera preview (OpenCV windows), you may need to grant screen recording permissions in some macOS versions.

Notes & Troubleshooting
-----------------------
- If the camera won't open, try running the script with a different camera index (0, 1, 2...) using `--camera`.
- On Windows, OpenCV may work more reliably with the DirectShow backend; the script attempts platform-appropriate backends automatically.
- If `opencv-python` installation fails on your machine, consult the error and consider installing build tools or using a pre-built wheel for your platform.

Example usage
-------------

```bash
python take_timer_photo.py --timer 10 --output myphoto.jpg
```

Advanced
--------
- If you want to embed this into a GUI app later, keep this script as a simple command-line utility and call it from your app or import the capture logic.
