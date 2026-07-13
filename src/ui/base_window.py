from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget

from compat import enable_dark_titlebar


class BaseWindow(QMainWindow):
    """Native, resizable base window with Koe's dark title-bar treatment."""

    def __init__(self, title: str, width: int, height: int):
        super().__init__()
        self.setWindowTitle(title)
        self.resize(width, height)
        self.setMinimumSize(min(width, 480), min(height, 420))

        self.main_widget = QWidget(self)
        self.main_layout = QVBoxLayout(self.main_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        self.setCentralWidget(self.main_widget)

    def showEvent(self, event):
        super().showEvent(event)
        enable_dark_titlebar(self)
