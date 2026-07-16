from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_installer_is_per_user_and_creates_two_direct_exe_shortcuts():
    installer = (ROOT / "packaging" / "Koe.iss").read_text(encoding="utf-8")

    assert "DefaultDirName={localappdata}\\Programs\\Koe" in installer
    assert "Name: \"{autodesktop}\\Koe Snippet\"" in installer
    assert 'Parameters: "--snippet"' in installer
    assert "Name: \"{autodesktop}\\Koe Scribe\"" in installer
    assert 'Parameters: "--scribe"' in installer
    assert "python" not in installer.lower()


def test_installer_preserves_runtime_data_and_bundles_only_openrouter():
    installer = (ROOT / "packaging" / "Koe.iss").read_text(encoding="utf-8")
    build_script = (ROOT / "packaging" / "build.ps1").read_text(encoding="utf-8")

    assert "onlyifdoesntexist uninsneveruninstall" in installer
    assert "CloseApplications=no" in installer
    assert "OPENROUTER_API_KEY" in build_script
    assert "must contain only one non-empty OPENROUTER_API_KEY" in build_script
    assert "Koe-Operator-Setup.exe" in build_script
    assert "if (-not $SkipInstaller -and (Test-Path -LiteralPath $installer))" in build_script
    assert "if (-not $KeepStandalone)" in build_script
    assert ".zip" not in (installer + build_script).lower()
