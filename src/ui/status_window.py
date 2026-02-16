import sys
import os
import time
from datetime import datetime
from pathlib import Path
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer, QRectF
from PyQt5.QtGui import QFont, QPixmap, QIcon, QPainter, QBrush, QColor, QPainterPath, QPen, QMouseEvent
from PyQt5.QtWidgets import QApplication, QLabel, QHBoxLayout, QVBoxLayout, QWidget, QMainWindow

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ui import theme

# Debug logging
_DEBUG_LOG = Path(__file__).parent.parent.parent / "logs" / "debug.log"

def _debug(msg: str):
    """Write debug message to file with timestamp."""
    try:
        with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
            timestamp = datetime.now().strftime("%H:%M:%S")
            f.write(f"[{timestamp}] [status_window] {msg}\n")
    except:
        pass


class StatusWindow(QMainWindow):
    statusSignal = pyqtSignal(str)
    closeSignal = pyqtSignal()

    # Terminal color scheme (from centralized theme)
    BG_COLOR = QColor(10, 10, 15, 245)  # Near black with alpha
    BORDER_COLOR = QColor(0, 255, 136)  # Terminal green (QColor for painting)
    TEXT_COLOR = theme.TEXT_COLOR
    SECONDARY_TEXT = theme.SECONDARY_TEXT
    RECORDING_COLOR = theme.RECORDING_COLOR

    def __init__(self):
        """
        Initialize the status window.
        """
        super().__init__()
        self.recording_start_time = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_timer)
        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(self.toggle_blink)
        self.blink_state = True
        self._drag_pos = None  # For window dragging
        self.initUI()
        self.statusSignal.connect(self.updateStatus)

    def initUI(self):
        """
        Initialize the user interface.
        """
        self.setWindowTitle('Recording')
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFixedSize(300, 60)  # Wider for DPI scaling on laptops

        self.main_widget = QWidget(self)
        self.main_layout = QHBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(16, 10, 16, 10)
        self.main_layout.setSpacing(12)

        # Recording indicator (blinking dot)
        self.indicator = QLabel("●")
        self.indicator.setFont(QFont('Cascadia Code', 14))
        self.indicator.setStyleSheet(f"color: {self.RECORDING_COLOR};")
        self.indicator.setFixedWidth(20)
        self.main_layout.addWidget(self.indicator)

        # Status and timer in vertical layout (centered)
        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)

        self.status_label = QLabel('> Recording_')
        self.status_label.setFont(QFont('Cascadia Code', 11, QFont.Bold))
        self.status_label.setStyleSheet(f"color: {self.TEXT_COLOR};")
        self.status_label.setAlignment(Qt.AlignCenter)

        self.timer_label = QLabel('00:00')
        self.timer_label.setFont(QFont('Cascadia Code', 10))
        self.timer_label.setStyleSheet(f"color: {self.SECONDARY_TEXT};")
        self.timer_label.setAlignment(Qt.AlignCenter)

        text_layout.addWidget(self.status_label)
        text_layout.addWidget(self.timer_label)

        self.main_layout.addLayout(text_layout, 1)  # Stretch factor of 1 to center

        self.setCentralWidget(self.main_widget)

    def paintEvent(self, event):
        """
        Create a rounded rectangle with terminal styling.
        """
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()).adjusted(1, 1, -1, -1), 6, 6)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Fill background
        painter.setBrush(QBrush(self.BG_COLOR))
        painter.setPen(Qt.NoPen)
        painter.drawPath(path)

        # Draw border
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(self.BORDER_COLOR, 1))
        painter.drawPath(path)

    def mousePressEvent(self, event: QMouseEvent):
        """Start dragging on left mouse button press."""
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        """Move window while dragging."""
        if event.buttons() == Qt.LeftButton and self._drag_pos is not None:
            self.move(event.globalPos() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Stop dragging on mouse release."""
        if event.button() == Qt.LeftButton:
            self._drag_pos = None
            event.accept()

    def toggle_blink(self):
        """Toggle the recording indicator visibility."""
        self.blink_state = not self.blink_state
        if self.blink_state:
            self.indicator.setStyleSheet(f"color: {self.RECORDING_COLOR};")
        else:
            self.indicator.setStyleSheet("color: transparent;")

    def update_timer(self):
        """Update the timer display."""
        if self.recording_start_time:
            elapsed = time.time() - self.recording_start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self.timer_label.setText(f'{minutes:02d}:{seconds:02d}')

    def show(self):
        """
        Position the window in the bottom center of the screen and show it.
        """
        screen = QApplication.primaryScreen()
        screen_geometry = screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()
        window_width = self.width()
        window_height = self.height()

        x = (screen_width - window_width) // 2
        y = screen_height - window_height - 120

        self.move(x, y)
        super().show()
        self.activateWindow()  # Ensure window can receive key events

    @pyqtSlot(str)
    def updateStatus(self, status):
        """
        Update the status window based on the given status.
        """
        _debug(f"updateStatus() called with status: {status}")
        # Safety check: don't update if window was closed (e.g., by cancel)
        if not self.isVisible() and status != 'recording':
            _debug(f"updateStatus() early return - window not visible")
            return

        if status == 'recording':
            _debug("updateStatus() handling 'recording'")
            self.indicator.setText("●")
            self.indicator.setStyleSheet(f"color: {self.RECORDING_COLOR};")
            self.status_label.setText('> Recording_')
            self.status_label.setStyleSheet(f"color: {self.TEXT_COLOR};")
            self.timer_label.setText('00:00')
            self.recording_start_time = time.time()
            self.timer.start(1000)  # Update every second
            self.blink_timer.start(500)  # Blink every 500ms
            self.show()
        elif status == 'transcribing':
            _debug("updateStatus() handling 'transcribing'")
            self.timer.stop()
            # Keep blinking during transcription
            self.indicator.setText("●")
            self.indicator.setStyleSheet(f"color: {self.TEXT_COLOR};")
            self.status_label.setText('> Transcribing_')
            self.status_label.setStyleSheet(f"color: {self.TEXT_COLOR};")

        elif status in ('complete', 'error'):
            _debug(f"updateStatus() handling '{status}' - stopping timers")
            self.timer.stop()
            self.blink_timer.stop()
            self.recording_start_time = None

            if status == 'complete':
                _debug("updateStatus() - closing immediately (beep is sufficient feedback)")
                self.close()
            elif status == 'error':
                # Error status - showError() will display the message
                # If no message provided via showError(), show generic error briefly
                self.indicator.setText("✗")
                self.indicator.setStyleSheet("color: #ff6666;")
                if '> Error:' not in self.status_label.text():
                    self.status_label.setText('> Error')
                    self.status_label.setStyleSheet("color: #ff6666;")
                QTimer.singleShot(3000, self.close)

        else:
            # Unknown status (e.g., 'idle') - do nothing
            _debug(f"updateStatus() - unknown status '{status}', ignoring")
            pass

    @pyqtSlot(str)
    def showError(self, error_msg):
        """Show error message in status window before closing."""
        self.timer.stop()
        self.blink_timer.stop()
        self.indicator.setText("✗")
        self.indicator.setStyleSheet("color: #ff6666;")
        # Truncate long error messages
        display_msg = error_msg[:30] + '...' if len(error_msg) > 30 else error_msg
        self.status_label.setText(f'> Error: {display_msg}')
        self.status_label.setStyleSheet("color: #ff6666;")
        self.timer_label.setText('')
        # Close after 3 seconds
        QTimer.singleShot(3000, self.close)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    status_window = StatusWindow()
    status_window.statusSignal.emit('recording')

    # Simulate status updates
    QTimer.singleShot(5000, lambda: status_window.statusSignal.emit('transcribing'))
    QTimer.singleShot(8000, lambda: status_window.statusSignal.emit('idle'))

    sys.exit(app.exec_())
