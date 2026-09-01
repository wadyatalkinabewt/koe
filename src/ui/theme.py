"""Shared visual system for every Koe desktop surface."""

FONT_FAMILY = '"Segoe UI"'

# Foundations
BG_COLOR = "#0B0F17"
SURFACE_COLOR = "#121826"
SURFACE_ELEVATED = "#171F2E"
SURFACE_HOVER = "#1D2738"
BORDER_COLOR = "#29354A"
DIVIDER_COLOR = "#222C3D"
CONTROL_BORDER = "#344159"

# Type
TEXT_COLOR = "#F4F7FB"
SECONDARY_TEXT = "#A7B1C2"
DIM_TEXT = "#6F7B8E"

# Accents and state
ACCENT_COLOR = "#7C8DFF"
ACCENT_HOVER = "#93A1FF"
ACCENT_SOFT = "#252E52"
RECORDING_COLOR = "#FF6F79"
ERROR_COLOR = "#FF6F79"
SUCCESS_COLOR = "#55D6A5"
ERROR_SOFT = "#321A22"
SUCCESS_SOFT = "#142A25"
LINK_COLOR = "#9BA8FF"

# Inputs and controls
INPUT_BG = "#0F1520"
INPUT_BORDER = BORDER_COLOR
INPUT_FOCUS_BORDER = ACCENT_COLOR
BUTTON_BG = SURFACE_ELEVATED
BUTTON_HOVER_BG = SURFACE_HOVER
BUTTON_BORDER = BORDER_COLOR
SELECTION_BG = ACCENT_COLOR
SELECTION_TEXT = "#FFFFFF"
SCROLLBAR_BG = BG_COLOR
SCROLLBAR_HANDLE = "#334158"
SCROLLBAR_HANDLE_HOVER = "#465774"
STATUS_BORDER_DIM = BORDER_COLOR


def application_stylesheet() -> str:
    """Return the shared form/window stylesheet used by Settings and Scribe."""
    return f"""
        QMainWindow, QWidget {{
            background-color: {BG_COLOR};
            color: {TEXT_COLOR};
            font-family: {FONT_FAMILY};
            font-size: 10pt;
        }}
        QLabel {{ background: transparent; color: {TEXT_COLOR}; }}
        QLabel#windowTitle {{ font-size: 22pt; font-weight: 700; }}
        QLabel#windowSubtitle {{ color: {SECONDARY_TEXT}; font-size: 10pt; }}
        QLabel#sectionTitle {{ font-size: 11pt; font-weight: 650; }}
        QLabel#fieldLabel {{ color: {SECONDARY_TEXT}; font-size: 9pt; }}
        QLabel#eyebrow {{ color: {ACCENT_COLOR}; font-size: 8pt; font-weight: 700; }}
        QFrame#card {{
            background-color: {SURFACE_COLOR};
            border: 1px solid {BORDER_COLOR};
            border-radius: 12px;
        }}
        QLineEdit, QTextEdit, QSpinBox, QComboBox {{
            background-color: {INPUT_BG};
            color: {TEXT_COLOR};
            border: 1px solid {CONTROL_BORDER};
            border-radius: 7px;
            padding: 9px 11px;
            selection-background-color: {SELECTION_BG};
            selection-color: {SELECTION_TEXT};
        }}
        QLineEdit:focus, QTextEdit:focus, QSpinBox:focus, QComboBox:focus {{
            border: 1px solid {INPUT_FOCUS_BORDER};
        }}
        QLineEdit:disabled, QTextEdit:disabled {{ color: {DIM_TEXT}; }}
        QPushButton {{
            min-height: 18px;
            background-color: {BUTTON_BG};
            color: {TEXT_COLOR};
            border: 1px solid {BUTTON_BORDER};
            border-radius: 7px;
            padding: 9px 16px;
            font-weight: 600;
        }}
        QPushButton:hover {{ background-color: {BUTTON_HOVER_BG}; }}
        QPushButton:pressed {{ background-color: {ACCENT_SOFT}; }}
        QPushButton:disabled {{ color: {DIM_TEXT}; border-color: {DIVIDER_COLOR}; }}
        QPushButton#primaryButton {{
            background-color: {ACCENT_COLOR};
            color: #FFFFFF;
            border-color: {ACCENT_COLOR};
        }}
        QPushButton#primaryButton:hover {{ background-color: {ACCENT_HOVER}; }}
        QPushButton#recordButton {{
            background-color: {RECORDING_COLOR};
            color: #FFFFFF;
            border-color: {RECORDING_COLOR};
            padding: 11px 22px;
        }}
        QPushButton#recordButton:hover {{ background-color: #FF858E; }}
        QCheckBox {{ color: {TEXT_COLOR}; spacing: 9px; padding: 3px 0; }}
        QCheckBox::indicator {{
            width: 17px;
            height: 17px;
            border: 1px solid {INPUT_BORDER};
            border-radius: 4px;
            background: {INPUT_BG};
        }}
        QCheckBox::indicator:checked {{
            background: {ACCENT_COLOR};
            border-color: {ACCENT_COLOR};
        }}
        QScrollArea {{ border: none; background: {BG_COLOR}; }}
        QScrollBar:vertical {{ background: {SCROLLBAR_BG}; width: 8px; margin: 0; }}
        QScrollBar::handle:vertical {{
            background: {SCROLLBAR_HANDLE};
            border-radius: 4px;
            min-height: 32px;
        }}
        QScrollBar::handle:vertical:hover {{ background: {SCROLLBAR_HANDLE_HOVER}; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        QToolTip {{
            background-color: {SURFACE_ELEVATED};
            color: {TEXT_COLOR};
            border: 1px solid {BORDER_COLOR};
            padding: 6px;
        }}
    """


def tray_menu_stylesheet() -> str:
    """Return the compact tray-menu styling used by the live Koe process."""
    return f"""
        QMenu {{
            background-color: {SURFACE_COLOR};
            color: {TEXT_COLOR};
            border: 1px solid {BORDER_COLOR};
            border-radius: 8px;
            padding: 7px;
            font-family: {FONT_FAMILY};
            font-size: 10pt;
        }}
        QMenu::item {{
            padding: 9px 30px 9px 14px;
            border-radius: 6px;
            margin: 1px 0;
        }}
        QMenu::item:selected {{ background-color: {ACCENT_SOFT}; color: {TEXT_COLOR}; }}
        QMenu::item:disabled {{ color: {DIM_TEXT}; }}
        QMenu::separator {{
            height: 1px;
            background-color: {DIVIDER_COLOR};
            margin: 6px 8px;
        }}
    """
