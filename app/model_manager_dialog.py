"""Checkpoint download, size check, and folder shortcut."""

from __future__ import annotations

import threading

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from utils.downloader import DownloadCancelled, checkpoint_info, download_checkpoint
from utils.gpu_info import format_bytes
from utils.logger import get_logger
from utils.paths import CHECKPOINT_DIR


class _DownloadThread(QThread):
    progress = Signal(int, int)
    succeeded = Signal(str)
    failed = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.cancel_event = threading.Event()

    def run(self) -> None:
        try:
            path = download_checkpoint(
                progress=lambda done, total: self.progress.emit(done, total),
                cancel_event=self.cancel_event,
            )
            self.succeeded.emit(str(path))
        except DownloadCancelled:
            self.failed.emit("Download cancelled")
        except Exception as exc:
            self.failed.emit(str(exc))


class _VerifyThread(QThread):
    succeeded = Signal(str)
    failed = Signal(str)

    def run(self) -> None:
        info = checkpoint_info()
        if info["status"] != "INSTALLED":
            self.failed.emit(f"Checkpoint status: {info['status']}")
            return
        try:
            import torch

            state = torch.load(info["path"], map_location="cpu", weights_only=False)
            count = len(state) if hasattr(state, "__len__") else 0
            del state
            if count <= 0:
                self.failed.emit("Checkpoint loaded but contains no tensors.")
                return
            self.succeeded.emit(f"Checkpoint verified. Tensors: {count}. Size: {format_bytes(info['size'])}")
        except TypeError:
            try:
                import torch

                state = torch.load(info["path"], map_location="cpu")
                count = len(state)
                del state
                self.succeeded.emit(f"Checkpoint verified. Tensors: {count}.")
            except Exception as exc:
                self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit(str(exc))


class ModelManagerDialog(QDialog):
    checkpoint_ready = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Model Manager")
        self.resize(560, 260)
        self._logger = get_logger("model")
        self.path_label = QLabel()
        self.path_label.setWordWrap(True)
        self.exists_label = QLabel()
        self.size_label = QLabel()
        self.status_label = QLabel()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.download_btn = QPushButton("DOWNLOAD")
        self.verify_btn = QPushButton("VERIFY")
        self.folder_btn = QPushButton("OPEN FOLDER")
        self.close_btn = QPushButton("CLOSE")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Official Depth Pro checkpoint"))
        layout.addWidget(self.path_label)
        layout.addWidget(self.exists_label)
        layout.addWidget(self.size_label)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress)
        row = QHBoxLayout()
        for button in (self.download_btn, self.verify_btn, self.folder_btn, self.close_btn):
            row.addWidget(button)
        layout.addLayout(row)
        self.download_btn.clicked.connect(self._download)
        self.verify_btn.clicked.connect(self._verify)
        self.folder_btn.clicked.connect(self._open_folder)
        self.close_btn.clicked.connect(self.accept)
        self._thread = None
        self.refresh()

    def refresh(self) -> None:
        info = checkpoint_info()
        self.path_label.setText(f"Path: {info['path']}")
        self.exists_label.setText(f"File exists: {'yes' if info['exists'] else 'no'}")
        self.size_label.setText(f"Size: {format_bytes(info['size'])}")
        self.status_label.setText(f"Status: {info['status']}")

    def _download(self) -> None:
        self.download_btn.setEnabled(False)
        self.progress.setValue(0)
        thread = _DownloadThread()
        self._thread = thread
        thread.progress.connect(self._on_progress)
        thread.succeeded.connect(self._on_downloaded)
        thread.failed.connect(self._on_failed)
        thread.start()

    def _on_progress(self, done: int, total: int) -> None:
        if total > 0:
            self.progress.setValue(int(done * 100 / total))
        self.status_label.setText(f"Downloading {format_bytes(done)}")

    def _on_downloaded(self, path: str) -> None:
        self._logger.info("Checkpoint downloaded %s", path)
        self.download_btn.setEnabled(True)
        self.refresh()
        self.checkpoint_ready.emit()
        QMessageBox.information(self, "Model Manager", "Checkpoint downloaded.")

    def _on_failed(self, message: str) -> None:
        self._logger.error("Checkpoint download failed: %s", message)
        self.download_btn.setEnabled(True)
        self.refresh()
        QMessageBox.warning(self, "Model Manager", message)

    def _verify(self) -> None:
        self.verify_btn.setEnabled(False)
        self.status_label.setText("Verifying checkpoint...")
        thread = _VerifyThread()
        self._thread = thread
        thread.succeeded.connect(self._verify_ok)
        thread.failed.connect(self._verify_failed)
        thread.start()

    def _verify_ok(self, message: str) -> None:
        self.verify_btn.setEnabled(True)
        self.status_label.setText("Status: VERIFIED")
        QMessageBox.information(self, "Verify", message)

    def _verify_failed(self, message: str) -> None:
        self.verify_btn.setEnabled(True)
        QMessageBox.warning(self, "Verify", message)

    @staticmethod
    def _open_folder() -> None:
        import os

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        os.startfile(CHECKPOINT_DIR)
