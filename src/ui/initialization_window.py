import os
import sys
import time

from PyQt5.QtCore import QRectF, Qt, QTimer
from PyQt5.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from paths import resource_path
from ui import theme


class InitializationWindow(QMainWindow):
    """Short-lived startup card using the same visual language as snippet status."""

    MIN_DISPLAY_TIME = 1.2

    def __init__(self):
        super().__init__()
        self.show_time = None
        self._build_ui()

    def _build_ui(self) -> None:
        self.setWindowTitle("Koe")
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFixedSize(184, 64)

        central = QWidget(self)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(14, 8, 14, 8)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignCenter)

        self.icon_label = QLabel()
        self.icon_label.setFixedSize(34, 34)
        self.icon_label.setAlignment(Qt.AlignCenter)
        icon_path = resource_path("assets", "koe-icon.png")
        if icon_path.exists():
            pixmap = QPixmap(str(icon_path)).scaled(
                32,
                32,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.icon_label.setPixmap(pixmap)
        layout.addWidget(self.icon_label)

        text_layout = QVBoxLayout()
        text_layout.setSpacing(3)
        title = QLabel("Koe")
        title.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        title.setStyleSheet(
            f"color: {theme.TEXT_COLOR}; font-family: {theme.FONT_FAMILY}; "
            "font-size: 12pt; font-weight: 650;"
        )
        self.status_label = QLabel("Initializing…")
        self.status_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.status_label.setStyleSheet(
            f"color: {theme.SECONDARY_TEXT}; font-family: {theme.FONT_FAMILY}; font-size: 9pt;"
        )
        text_layout.addWidget(title)
        text_layout.addWidget(self.status_label)
        layout.addLayout(text_layout)
        self.setCentralWidget(central)

    def paintEvent(self, event) -> None:
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()).adjusted(1, 1, -1, -1), 14, 14)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        background = QColor(theme.SURFACE_COLOR)
        background.setAlpha(250)
        painter.setBrush(QBrush(background))
        painter.setPen(QPen(QColor(theme.BORDER_COLOR), 1))
        painter.drawPath(path)

    def show(self) -> None:
        screen = QApplication.primaryScreen()
        available = screen.availableGeometry()
        x = available.x() + (available.width() - self.width()) // 2
        y = available.y() + available.height() - self.height() - 36
        self.move(x, y)
        super().show()
        self.raise_()
        self.show_time = time.time()

    def close(self) -> bool:
        if self.show_time:
            remaining = self.MIN_DISPLAY_TIME - (time.time() - self.show_time)
            if remaining > 0:
                QTimer.singleShot(int(remaining * 1000), self._do_close)
                return False
        return self._do_close()

    def _do_close(self) -> bool:
        return super().close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = InitializationWindow()
    window.show()
    QTimer.singleShot(2500, window.close)
    sys.exit(app.exec_())
