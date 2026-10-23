"""Render assets/src/*.html to PNG with headless Chrome or Edge (2x pixel density).

    python scripts/render_assets.py

Requires Pillow. Outputs: assets/banner.png, assets/social-preview.png, assets/workflow.png.
Set BROWSER to a Chrome/Edge/Chromium executable to override discovery.
"""

import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "assets" / "src"
JOBS = [("banner.html", "banner.png", 1280, 320), ("social.html", "social-preview.png", 1280, 640), ("workflow.html", "workflow.png", 1600, 740)]
CANDIDATES = [
    os.environ.get("BROWSER", ""),
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "google-chrome", "chromium", "chromium-browser",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]


def browser() -> str:
    for c in CANDIDATES:
        if c and (pathlib.Path(c).exists() or shutil.which(c)):
            return c
    sys.exit("no Chrome/Edge/Chromium found; set BROWSER")


def main() -> None:
    exe = browser()
    for src, out, w, h in JOBS:
        dest = ROOT / "assets" / out
        with tempfile.TemporaryDirectory() as profile:
            subprocess.run([exe, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--user-data-dir={profile}",
                            "--force-device-scale-factor=2", f"--window-size={w},{h}", f"--screenshot={dest}",
                            (SRC / src).as_uri()], check=True, capture_output=True)
        # 256-colour palette: halves file size with no visible change on these flat designs.
        from PIL import Image

        Image.open(dest).convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT).save(dest, optimize=True)
        print(f"rendered {dest.relative_to(ROOT)} ({dest.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
