"""Dark engineering theme."""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


def apply_theme(app: QApplication) -> None:
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(22, 24, 28))
    palette.setColor(QPalette.WindowText, QColor(230, 232, 235))
    palette.setColor(QPalette.Base, QColor(16, 18, 22))
    palette.setColor(QPalette.AlternateBase, QColor(30, 33, 38))
    palette.setColor(QPalette.Text, QColor(230, 232, 235))
    palette.setColor(QPalette.Button, QColor(36, 40, 46))
    palette.setColor(QPalette.ButtonText, QColor(230, 232, 235))
    palette.setColor(QPalette.Highlight, QColor(40, 120, 170))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QPalette.ToolTipBase, QColor(36, 40, 46))
    palette.setColor(QPalette.ToolTipText, QColor(230, 232, 235))
    palette.setColor(QPalette.PlaceholderText, QColor(140, 145, 150))
    app.setPalette(palette)
    app.setStyleSheet(
        """
        QWidget { color: #e6e8eb; font-size: 13px; }
        QMainWindow, QDialog { background: #16181c; }
        QGroupBox {
            border: 1px solid #2c3138;
            margin-top: 10px;
            padding: 8px;
            font-weight: 600;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
        QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QListWidget {
            background: #101216;
            border: 1px solid #343a42;
            padding: 4px;
        }
        QPushButton {
            background: #2a3038;
            border: 1px solid #3c4450;
            padding: 6px 10px;
        }
        QPushButton:hover { background: #343c46; }
        QPushButton:disabled { color: #808890; }
        QPushButton#recordButton {
            background: #8d1d2c;
            border: 1px solid #c4384a;
            font-weight: 700;
            padding: 8px 14px;
        }
        QPushButton#recordButton:hover { background: #a82436; }
        QSlider::groove:horizontal { height: 6px; background: #2c3138; }
        QSlider::handle:horizontal { background: #6eb6ff; width: 14px; margin: -5px 0; }
        QStatusBar { background: #101216; }
        QScrollArea { border: none; }
        QCheckBox { spacing: 6px; }
        """
    )
