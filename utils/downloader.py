"""Download the official Depth Pro checkpoint.

The URL matches get_pretrained_models.sh in apple-aiml-research/ml-depth-pro.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

from utils.paths import CHECKPOINT_URL, MIN_CHECKPOINT_BYTES, checkpoint_path, ensure_dirs


class DownloadCancelled(Exception):
    pass


def checkpoint_info(path: Path | None = None) -> dict:
    path = checkpoint_path() if path is None else Path(path)
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "size": 0,
            "status": "MISSING",
        }
    size = path.stat().st_size
    status = "INSTALLED" if size >= MIN_CHECKPOINT_BYTES else "INCOMPLETE"
    return {
        "path": str(path),
        "exists": True,
        "size": size,
        "status": status,
    }


def download_checkpoint(dest: Path | None = None, progress=None, cancel_event=None) -> Path:
    ensure_dirs()
    dest = checkpoint_path() if dest is None else Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(".pt.part")
    request = urllib.request.Request(
        CHECKPOINT_URL,
        headers={"User-Agent": "DepthProVisionStudio"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        with temporary.open("wb") as handle:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise DownloadCancelled("Checkpoint download cancelled")
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done, total)
    temporary.replace(dest)
    return dest
