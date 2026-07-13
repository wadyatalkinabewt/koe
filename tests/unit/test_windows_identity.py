import gc
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@pytest.mark.skipif(sys.platform != "win32", reason="Windows taskbar identity only")
def test_scribe_shortcut_has_koe_icon_and_matching_app_id(tmp_path):
    import pythoncom
    import win32com.client
    from win32com.propsys import propsys, pscon

    from compat import ensure_windows_shortcut

    project_root = Path(__file__).parent.parent.parent
    shortcut_path = tmp_path / "Koe Scribe.lnk"
    pythonw = Path(sys.executable).with_name("pythonw.exe")
    shortcut_result = ensure_windows_shortcut(
        shortcut_path,
        app_id="Koe.Scribe.App",
        target_path=pythonw,
        arguments="-m src.meeting.app",
        working_directory=project_root,
        icon_path=project_root / "assets" / "koe-icon.ico",
    )

    shortcut = win32com.client.Dispatch("WScript.Shell").CreateShortcut(str(shortcut_path))
    store = propsys.SHGetPropertyStoreFromParsingName(
        str(shortcut_path), None, 0, propsys.IID_IPropertyStore
    )
    app_id = store.GetValue(pscon.PKEY_AppUserModel_ID).GetValue()

    assert shortcut_result == shortcut_path.resolve()
    assert Path(shortcut.TargetPath).resolve() == pythonw.resolve()
    assert shortcut.Arguments == "-m src.meeting.app"
    assert "koe-icon.ico" in shortcut.IconLocation.lower()
    assert app_id == "Koe.Scribe.App"

    store = None
    shortcut = None
    pythoncom.CoFreeUnusedLibraries()
    gc.collect()
