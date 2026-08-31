import os
import sys
import time
from datetime import datetime

from PyQt5.QtCore import QRectF, Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QBrush, QColor, QMouseEvent, QPainter, QPainterPath, QPen
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QWidget,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from paths import logs_dir
from ui import theme

_DEBUG_LOG = logs_dir() / "debug.log"


def _debug(message: str) -> None:
    try:
        _DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_DEBUG_LOG, "a", encoding="utf-8") as log_file:
            log_file.write(
                f"[{datetime.now().strftime('%H:%M:%S')}] [status_window] {message}\n"
            )
    except Exception:
        pass


class StatusWindow(QMainWindow):
    """Compact always-on-top card for snippet capture and transcription state."""

    statusSignal = pyqtSignal(str)
    closeSignal = pyqtSignal()
    cancelSignal = pyqtSignal()
    dismissSignal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.recording_start_time = None
        self._drag_pos = None
        self._pulse_on = True
        self.current_status = "idle"

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer)
        self.pulse_timer = QTimer(self)
        self.pulse_timer.timeout.connect(self._pulse_indicator)

        self._build_ui()
        self.statusSignal.connect(self.updateStatus)

    def _build_ui(self) -> None:
        self.setWindowTitle("Koe")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFixedSize(207, 52)

        central = QWidget(self)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(12, 9, 10, 9)
        layout.setSpacing(7)

        self.indicator = QLabel("●")
        self.indicator.setStyleSheet(
            f"color: {theme.RECORDING_COLOR}; font-size: 14px;"
        )
        self.indicator.setFixedWidth(12)
        layout.addWidget(self.indicator)

        self.status_label = QLabel("Listening")
        self.status_label.setStyleSheet(
            f"color: {theme.TEXT_COLOR}; font-family: {theme.FONT_FAMILY}; "
            "font-size: 11pt; font-weight: 650;"
        )
        self.status_label.setFixedWidth(88)
        self.status_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(self.status_label)

        self.timer_label = QLabel("00:00")
        self.timer_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.timer_label.setFixedWidth(42)
        self.timer_label.setStyleSheet(
            f"color: {theme.SECONDARY_TEXT}; background: transparent; border: none; "
            f"font-family: {theme.FONT_FAMILY}; font-size: 10pt; font-weight: 600;"
        )
        layout.addWidget(self.timer_label)

        self.cancel_button = QPushButton("×")
        self.cancel_button.setObjectName("snippetCancelButton")
        self.cancel_button.setAccessibleName("Cancel snippet")
        self.cancel_button.setToolTip("Cancel without transcribing")
        self.cancel_button.setCursor(Qt.PointingHandCursor)
        self.cancel_button.setFixedSize(22, 22)
        self.cancel_button.setStyleSheet(
            "QPushButton { color: #F0A7AE; background: transparent; "
            "border: none; border-radius: 7px; font-size: 15pt; font-weight: 500; "
            "padding: 0; margin: 0; } "
            "QPushButton:hover { color: #FF7884; background: #2A171D; } "
            "QPushButton:pressed { color: #FF5F6D; background: #351A21; }"
        )
        self.cancel_button.clicked.connect(self._handle_close)
        self.cancel_button.hide()
        layout.addWidget(self.cancel_button)
        self.setCentralWidget(central)

    def _handle_close(self) -> None:
        if self.current_status == "recording":
            self.cancel_button.setEnabled(False)
            self.cancelSignal.emit()
            return
        if self.current_status != "transcribing":
            return
        self.cancel_button.setEnabled(False)
        self.dismissSignal.emit()
        self.current_status = "dismissed"
        self.close()

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

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.LeftButton and self._drag_pos is not None:
            self.move(event.globalPos() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_pos = None
            event.accept()

    def _pulse_indicator(self) -> None:
        self._pulse_on = not self._pulse_on
        color = self._indicator_color if self._pulse_on else theme.DIM_TEXT
        self.indicator.setStyleSheet(f"color: {color}; font-size: 14px;")

    def update_timer(self) -> None:
        if self.recording_start_time is None:
            return
        elapsed = time.time() - self.recording_start_time
        minutes, seconds = divmod(int(elapsed), 60)
        self.timer_label.setText(f"{minutes:02d}:{seconds:02d}")

    def show(self) -> None:
        screen = QApplication.primaryScreen()
        available = screen.availableGeometry()
        x = available.x() + (available.width() - self.width()) // 2
        y = available.y() + available.height() - self.height() - 36
        self.move(x, y)
        super().show()
        self.raise_()

    def _set_state(self, title: str, color: str, *, cancellable: bool = False) -> None:
        self._indicator_color = color
        self.indicator.setStyleSheet(f"color: {color}; font-size: 14px;")
        self.status_label.setText(title)
        self.cancel_button.setEnabled(cancellable)
        self.cancel_button.setVisible(cancellable)

    @pyqtSlot(str)
    def updateStatus(self, status: str) -> None:
        _debug(f"updateStatus: {status}")
        if not self.isVisible() and status != "recording":
            return

        self.current_status = status

        if status == "recording":
            self._set_state("Listening", theme.RECORDING_COLOR, cancellable=True)
            self.cancel_button.setAccessibleName("Cancel snippet")
            self.cancel_button.setToolTip("Cancel without transcribing")
            self.timer_label.setText("00:00")
            self.recording_start_time = time.time()
            self.timer.start(1000)
            self.pulse_timer.start(700)
            self.show()
        elif status == "transcribing":
            self.timer.stop()
            self._set_state("Transcribing", theme.ACCENT_COLOR, cancellable=True)
            self.cancel_button.setAccessibleName("Dismiss transcription")
            self.cancel_button.setToolTip("Dismiss without copying to the clipboard")
            self.pulse_timer.start(700)
        elif status in ("complete", "cancelled"):
            self.timer.stop()
            self.pulse_timer.stop()
            self.recording_start_time = None
            self.cancel_button.hide()
            self.close()
        elif status == "error":
            self.timer.stop()
            self.pulse_timer.stop()
            self.recording_start_time = None
            self._set_state("Failed", theme.ERROR_COLOR, cancellable=False)
            QTimer.singleShot(3500, self.close)

    @pyqtSlot(str)
    def showError(self, error_msg: str) -> None:
        self.timer.stop()
        self.pulse_timer.stop()
        self.current_status = "error"
        self._set_state("Failed", theme.ERROR_COLOR, cancellable=False)
        self.timer_label.setText("—")
        QTimer.singleShot(3500, self.close)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = StatusWindow()
    window.updateStatus("recording")
    sys.exit(app.exec_())
