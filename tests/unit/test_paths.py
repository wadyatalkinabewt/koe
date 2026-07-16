from pathlib import Path

import paths


def test_source_runs_keep_all_runtime_state_in_the_checkout(monkeypatch):
    monkeypatch.delenv("KOE_APPDATA_DIR", raising=False)
    monkeypatch.delenv("KOE_DOCUMENTS_DIR", raising=False)
    monkeypatch.setattr(paths, "is_frozen", lambda: False)

    root = paths.source_root()
    assert paths.app_data_dir() == root
    assert paths.documents_dir() == root
    assert paths.env_path() == root / ".env"
    assert paths.config_path() == root / "config.yaml"
    assert paths.logs_dir() == root / "logs"
    assert paths.scribe_temp_dir() == root / ".scribe_temp"
    assert paths.default_snippets_dir() == root / "Snippets"
    assert paths.default_meetings_dir() == root / "Meetings"


def test_packaged_runs_keep_per_user_windows_locations(tmp_path, monkeypatch):
    monkeypatch.delenv("KOE_APPDATA_DIR", raising=False)
    monkeypatch.delenv("KOE_DOCUMENTS_DIR", raising=False)
    monkeypatch.setattr(paths, "is_frozen", lambda: True)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "LocalAppData"))
    monkeypatch.setattr(paths, "_known_documents_dir", lambda: tmp_path / "Documents")

    assert paths.app_data_dir() == tmp_path / "LocalAppData" / "Koe"
    assert paths.documents_dir() == tmp_path / "Documents" / "Koe"
    assert paths.scribe_temp_dir() == tmp_path / "LocalAppData" / "Koe" / "scribe-temp"


def test_explicit_runtime_overrides_still_win(tmp_path, monkeypatch):
    app_data = tmp_path / "custom-appdata"
    documents = tmp_path / "custom-documents"
    monkeypatch.setenv("KOE_APPDATA_DIR", str(app_data))
    monkeypatch.setenv("KOE_DOCUMENTS_DIR", str(documents))
    monkeypatch.setattr(paths, "is_frozen", lambda: False)

    assert paths.app_data_dir() == app_data.resolve()
    assert paths.documents_dir() == documents.resolve()
    assert paths.scribe_temp_dir() == app_data.resolve() / "scribe-temp"
