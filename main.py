"""Depth Pro Vision Studio entry point."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow
from app.theme import apply_theme
from utils.logger import get_logger, setup_logging


def main() -> int:
    setup_logging()
    logger = get_logger("app")
    logger.info("Application start")
    logger.info("Python version %s", sys.version.replace("\n", " "))

    def _excepthook(exc_type, exc, tb):
        logger.error("Unhandled exception\n%s", "".join(traceback.format_exception(exc_type, exc, tb)))
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _excepthook
    app = QApplication(sys.argv)
    app.setApplicationName("Depth Pro Vision Studio")
    app.setFont(QFont("Segoe UI", 10))
    apply_theme(app)
    window = MainWindow()
    window.show()
    code = app.exec()
    logger.info("Shutdown")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
