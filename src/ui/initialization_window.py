import sys
import os
import time
from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtGui import QFont, QPainter, QBrush, QColor, QPainterPath, QPen
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QMainWindow

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ui import theme


class InitializationWindow(QMainWindow):
    # Terminal color scheme (from centralized theme)
    BG_COLOR = QColor(10, 10, 15, 245)  # Near black with alpha
    BORDER_COLOR = QColor(0, 255, 136)  # Terminal green (QColor for painting)
    TEXT_COLOR = theme.TEXT_COLOR

    MIN_DISPLAY_TIME = 1.5  # Minimum seconds to display

    def __init__(self):
        """Initialize the initialization window."""
        super().__init__()
        self.blink_timer = QTimer()
        self.blink_timer.timeout.connect(self.toggle_blink)
        self.blink_state = True
        self.show_time = None
        self.initUI()

    def initUI(self):
        """Initialize the user interface."""
        self.setWindowTitle('Koe')
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFixedSize(300, 60)

        self.main_widget = QWidget(self)
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(16, 10, 16, 10)
        self.main_layout.setSpacing(0)

        # Status label with blinking cursor
        self.status_label = QLabel('> Initializing_')
        self.status_label.setFont(QFont('Cascadia Code', 12, QFont.Bold))
        self.status_label.setStyleSheet(f"color: {self.TEXT_COLOR};")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(self.status_label)

        self.setCentralWidget(self.main_widget)

    def paintEvent(self, event):
        """Create a rounded rectangle with terminal styling."""
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

    def toggle_blink(self):
        """Toggle the cursor visibility."""
        self.blink_state = not self.blink_state
        if self.blink_state:
            self.status_label.setText('> Initializing_')
        else:
            self.status_label.setText('> Initializing')

    def show(self):
        """Position the window in the bottom center of the screen and show it."""
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
        self.show_time = time.time()
        self.blink_timer.start(500)  # Blink every 500ms

    def close(self):
        """Stop the blink timer and close the window after minimum display time."""
        if self.show_time:
            elapsed = time.time() - self.show_time
            remaining = self.MIN_DISPLAY_TIME - elapsed
            if remaining > 0:
                # Delay close to meet minimum display time
                QTimer.singleShot(int(remaining * 1000), self._do_close)
                return
        self._do_close()

    def _do_close(self):
        """Actually close the window."""
        self.blink_timer.stop()
        super().close()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    init_window = InitializationWindow()
    init_window.show()

    # Simulate closing after 3 seconds
    QTimer.singleShot(3000, init_window.close)

    sys.exit(app.exec_())
