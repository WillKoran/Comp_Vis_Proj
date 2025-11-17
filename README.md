# Webcam Timer Photo

Small cross-platform Python utility to take a timed photo from your webcam.

Files added:
- `take_timer_photo.py` — main script
- `requirements.txt` — runtime dependencies
- `docs/requirements_document.md` — setup and troubleshooting notes

Quick example (PowerShell):

```powershell
python take_timer_photo.py --timer 5 --output photo.jpg
```

Press 'q' while the preview window is open to cancel.

See `docs/requirements_document.md` for environment setup steps on Windows and macOS.

GUI mode
--------

You can run a small GUI that lets you type a base filename, choose how many pictures to take, the countdown timer, and the interval between pictures:

```powershell
python take_timer_photo.py --gui
```

The GUI requires Pillow (installed via `pip install Pillow` or `pip install -r requirements.txt`).
