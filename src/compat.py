"""Windows integration helpers for Koe's tray and desktop surfaces."""

import ctypes
import subprocess
import sys
from ctypes import wintypes
from pathlib import Path


def acquire_single_instance_lock(lock_name="KoeTranscriptionApp"):
    """Acquire Koe's named Windows mutex or exit when another instance owns it."""
    error_already_exists = 183
    mutex = ctypes.windll.kernel32.CreateMutexW(  # type: ignore[attr-defined]
        None,
        False,
        f"{lock_name}Mutex_v1",
    )
    if ctypes.windll.kernel32.GetLastError() == error_already_exists:  # type: ignore[attr-defined]
        print("[Koe] Another instance is already running. Exiting.")
        sys.exit(0)
    return mutex


def release_single_instance_lock(lock_handle) -> None:
    if lock_handle is None:
        return
    try:
        ctypes.windll.kernel32.ReleaseMutex(lock_handle)  # type: ignore[attr-defined]
        ctypes.windll.kernel32.CloseHandle(lock_handle)  # type: ignore[attr-defined]
    except Exception:
        pass


def set_app_user_model_id(app_id="Koe.Transcription.App") -> None:
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)  # type: ignore[attr-defined]
    except Exception:
        pass


def apply_window_icon(window, icon_path, app_id=None) -> None:
    """Apply Qt/native icons and optional taskbar identity metadata."""
    from PyQt5.QtGui import QIcon

    icon_path = Path(icon_path).resolve()
    if not icon_path.exists():
        return
    window.setWindowIcon(QIcon(str(icon_path)))

    try:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        user32.LoadImageW.restype = wintypes.HANDLE
        handles = []
        hwnd = int(window.winId())
        for icon_type, size in ((1, 32), (0, 16)):
            handle = user32.LoadImageW(
                None,
                str(icon_path),
                1,  # IMAGE_ICON
                size,
                size,
                0x0010,  # LR_LOADFROMFILE
            )
            if handle:
                user32.SendMessageW(hwnd, 0x0080, icon_type, handle)  # WM_SETICON
                handles.append(handle)
        window._koe_native_icon_handles = handles
    except Exception:
        pass

    if app_id:
        try:
            from win32com.propsys import propsys, pscon

            store = propsys.SHGetPropertyStoreForWindow(
                int(window.winId()),
                propsys.IID_IPropertyStore,
            )
            store.SetValue(
                pscon.PKEY_AppUserModel_ID,
                propsys.PROPVARIANTType(str(app_id)),
            )
            store.SetValue(
                pscon.PKEY_AppUserModel_RelaunchIconResource,
                propsys.PROPVARIANTType(f"{icon_path},0"),
            )
            store.Commit()
            window._koe_taskbar_property_store = store
        except Exception:
            pass


def ensure_windows_shortcut(
    shortcut_path,
    *,
    app_id,
    target_path,
    arguments,
    working_directory,
    icon_path,
):
    """Create/update a Windows shortcut registered to the supplied taskbar ID."""
    shortcut_path = Path(shortcut_path).resolve()
    shortcut_path.parent.mkdir(parents=True, exist_ok=True)
    icon_path = Path(icon_path).resolve()

    import win32com.client
    from win32com.propsys import propsys, pscon

    shortcut = win32com.client.Dispatch("WScript.Shell").CreateShortcut(
        str(shortcut_path)
    )
    shortcut.TargetPath = str(Path(target_path).resolve())
    shortcut.Arguments = str(arguments)
    shortcut.WorkingDirectory = str(Path(working_directory).resolve())
    shortcut.IconLocation = f"{icon_path},0"
    shortcut.Description = "Koe Scribe meeting recorder"
    shortcut.Save()

    store = propsys.SHGetPropertyStoreFromParsingName(
        str(shortcut_path),
        None,
        2,  # GPS_READWRITE
        propsys.IID_IPropertyStore,
    )
    store.SetValue(
        pscon.PKEY_AppUserModel_ID,
        propsys.PROPVARIANTType(str(app_id)),
    )
    store.Commit()
    return shortcut_path


def enable_dark_titlebar(window) -> None:
    try:
        enabled = ctypes.c_int(1)
        hwnd = ctypes.c_void_p(int(window.winId()))
        for attribute in (20, 19):
            result = ctypes.windll.dwmapi.DwmSetWindowAttribute(  # type: ignore[attr-defined]
                hwnd,
                attribute,
                ctypes.byref(enabled),
                ctypes.sizeof(enabled),
            )
            if result == 0:
                break
    except Exception:
        pass


def play_sound_file(file_path) -> None:
    try:
        import winsound

        winsound.PlaySound(str(file_path), winsound.SND_FILENAME)
    except Exception:
        pass


def clipboard_copy_fallback(text) -> bool:
    try:
        process = subprocess.Popen(["clip"], stdin=subprocess.PIPE)
        process.communicate(str(text).encode("utf-16le"))
        return process.returncode == 0
    except Exception:
        return False
