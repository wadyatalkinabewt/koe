import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from key_listener import KeyListener


def test_hotkey_parser_normalizes_windows_modifiers():
    assert KeyListener.parse_key_combination("ctrl + Shift + space") == {
        "CTRL",
        "SHIFT",
        "SPACE",
    }
    assert KeyListener.parse_key_combination("win+f8") == {"META", "F8"}


def test_callbacks_only_accept_supported_events():
    listener = KeyListener.__new__(KeyListener)
    listener.callbacks = {"on_activate": [], "on_deactivate": []}

    def callback():
        return None

    listener.add_callback("on_activate", callback)
    listener.add_callback("unknown", callback)

    assert listener.callbacks == {
        "on_activate": [callback],
        "on_deactivate": [],
    }
