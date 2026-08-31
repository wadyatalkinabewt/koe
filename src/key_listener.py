"""Windows global-hotkey listener for Koe snippets."""

from collections.abc import Callable
from threading import RLock

from utils import ConfigManager

_MODIFIER_NAMES = {
    "CTRL": "CTRL",
    "CONTROL": "CTRL",
    "SHIFT": "SHIFT",
    "ALT": "ALT",
    "META": "META",
    "WIN": "META",
    "WINDOWS": "META",
}


class KeyListener:
    """Emit activation/deactivation when the configured chord changes state."""

    def __init__(self):
        self.callbacks: dict[str, list[Callable]] = {
            "on_activate": [],
            "on_deactivate": [],
        }
        self.activation_keys: set[str] = set()
        self._pressed_native: set[object] = set()
        self._listener = None
        self._keyboard = None
        self._lock = RLock()
        self.load_activation_keys()

    @staticmethod
    def parse_key_combination(combination: str) -> set[str]:
        """Normalize a setting such as ``ctrl+shift+space`` into chord tokens."""
        tokens: set[str] = set()
        for part in str(combination or "").split("+"):
            token = part.strip().upper()
            if not token:
                continue
            tokens.add(_MODIFIER_NAMES.get(token, token))
        return tokens

    def load_activation_keys(self) -> None:
        combination = (
            ConfigManager.get_config_value("recording_options", "activation_key")
            or "ctrl+shift+space"
        )
        with self._lock:
            self.activation_keys = self.parse_key_combination(combination)
            self._pressed_native.clear()

    def update_activation_keys(self) -> None:
        self.load_activation_keys()

    def add_callback(self, event: str, callback: Callable) -> None:
        if event in self.callbacks:
            self.callbacks[event].append(callback)

    def start(self) -> None:
        """Start listening; repeated calls are deliberately harmless."""
        with self._lock:
            if self._listener is not None and self._listener.is_alive():
                return

            from pynput import keyboard

            self._keyboard = keyboard
            self._pressed_native.clear()
            self._listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self._listener.start()

    def stop(self) -> None:
        with self._lock:
            listener = self._listener
            self._listener = None
            self._pressed_native.clear()
        if listener is not None:
            listener.stop()

    def _token_for(self, key) -> str | None:
        keyboard = self._keyboard
        if keyboard is None:
            return None

        modifier_keys = {
            keyboard.Key.ctrl_l: "CTRL",
            keyboard.Key.ctrl_r: "CTRL",
            keyboard.Key.shift_l: "SHIFT",
            keyboard.Key.shift_r: "SHIFT",
            keyboard.Key.alt_l: "ALT",
            keyboard.Key.alt_r: "ALT",
            keyboard.Key.cmd_l: "META",
            keyboard.Key.cmd_r: "META",
        }
        if key in modifier_keys:
            return modifier_keys[key]

        char = getattr(key, "char", None)
        if char:
            return str(char).upper()

        name = str(getattr(key, "name", "") or "").upper()
        aliases = {
            "ESC": "ESC",
            "ESCAPE": "ESC",
            "RETURN": "ENTER",
        }
        return aliases.get(name, name or None)

    def _active_tokens(self) -> set[str]:
        return {
            token
            for key in self._pressed_native
            if (token := self._token_for(key)) is not None
        }

    def _is_active(self) -> bool:
        return bool(self.activation_keys) and self.activation_keys.issubset(
            self._active_tokens()
        )

    def _on_press(self, key) -> None:
        with self._lock:
            was_active = self._is_active()
            self._pressed_native.add(key)
            is_active = self._is_active()
        if not was_active and is_active:
            self._emit("on_activate")

    def _on_release(self, key) -> None:
        with self._lock:
            was_active = self._is_active()
            self._pressed_native.discard(key)
            is_active = self._is_active()
        if was_active and not is_active:
            self._emit("on_deactivate")

    def _emit(self, event: str) -> None:
        for callback in tuple(self.callbacks.get(event, ())):
            callback()
