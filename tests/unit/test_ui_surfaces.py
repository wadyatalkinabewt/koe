import os
import sys
import wave
from array import array
from datetime import datetime
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import QPoint, QRect, QSize, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFrame,
    QInputDialog,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def reset_config_manager():
    from utils import ConfigManager

    original = ConfigManager._instance
    ConfigManager._instance = None
    yield
    ConfigManager._instance = original


def test_settings_autosaves_without_footer_or_retired_controls(qapp):
    from paths import default_meetings_dir, default_snippets_dir
    from ui.settings_window import SettingsWindow, ToggleRow

    window = SettingsWindow()
    labels = " ".join(label.text() for label in window.findChildren(QLabel)).lower()
    button_texts = [button.text() for button in window.findChildren(QPushButton)]

    assert window.findChild(QFrame, "settingsFooter") is None
    assert "Cancel" not in button_texts
    assert "Save changes" not in button_texts
    assert window.save_status_label.text() == ""
    assert window.save_status_label.isHidden()
    assert "elevenlabs" not in labels
    assert "keep koe focused" not in labels
    assert "profile" not in labels
    assert "choose where transcripts and snippets live" in labels
    assert "no-verbatim mode is always on" not in labels
    assert "koe uses one speech-to-text path" not in labels
    assert hasattr(window, "save_meeting_audio_checkbox")
    assert window.save_meeting_audio_checkbox.text() == "Save Scribe meeting audio"
    assert isinstance(window.save_meeting_audio_checkbox, ToggleRow)
    assert hasattr(window, "save_markdown_checkbox")
    assert window.save_markdown_checkbox.text() == "Save Markdown copies"
    assert isinstance(window.save_markdown_checkbox, ToggleRow)
    assert window.findChildren(QCheckBox) == []
    assert "microphone.wav and meeting-audio.wav" not in labels
    window.show()
    qapp.processEvents()
    original_size = window.size()
    window.resize(window.width(), 980)
    qapp.processEvents()
    assert window.size() == original_size
    assert window.minimumSize() == window.maximumSize() == window.size()
    assert not window.windowFlags() & Qt.WindowMaximizeButtonHint
    assert window.windowFlags() & Qt.MSWindowsFixedSizeDialogHint
    cards = window.findChildren(QFrame, "card")
    cards_by_title = {
        next(
            label.text()
            for label in card.findChildren(QLabel)
            if label.objectName() == "sectionTitle"
        ): card
        for card in cards
    }
    card_headings = {
        title: next(
            label
            for label in card.findChildren(QLabel)
            if label.objectName() == "sectionTitle" and label.text() == title
        )
        for title, card in cards_by_title.items()
    }
    assert (
        len(
            {
                heading.mapTo(cards_by_title[title], QPoint()).y()
                for title, heading in card_headings.items()
            }
        )
        == 1
    )
    profile = cards_by_title["Your Name"]
    storage = cards_by_title["Storage"]
    scribe_card = cards_by_title["Scribe"]
    assert "Transcription" not in cards_by_title
    assert "Recording" not in cards_by_title
    assert window.user_name_input.parentWidget() is profile
    assert window.save_meeting_audio_checkbox.parentWidget() is scribe_card
    assert window.save_markdown_checkbox.parentWidget() is scribe_card
    assert window.save_meeting_audio_checkbox.parentWidget() is not storage
    assert "show the snippet status card" not in labels
    assert all(
        card.sizePolicy().verticalPolicy() == QSizePolicy.Maximum
        for card in cards_by_title.values()
    )
    assert all(
        card.layout().alignment() & Qt.AlignTop for card in cards_by_title.values()
    )
    storage_description = next(
        label
        for label in storage.findChildren(QLabel)
        if label.text() == "Choose where transcripts and snippets live."
    )
    snippets_title = next(
        label
        for label in storage.findChildren(QLabel)
        if label.text() == "Snippets Folder"
    )
    storage_detail_gap = snippets_title.mapTo(storage, QPoint()).y() - (
        storage_description.mapTo(storage, QPoint()).y() + storage_description.height()
    )
    assert storage_detail_gap > 0
    assert "PDF transcripts and summaries are always saved." in {
        label.text() for label in scribe_card.findChildren(QLabel)
    }
    assert (
        window.snippets_input.mapTo(window, QPoint()).y()
        < window.meetings_input.mapTo(window, QPoint()).y()
    )
    assert window.snippets_input.placeholderText() == (
        f"Leave empty for {default_snippets_dir()}"
    )
    assert window.meetings_input.placeholderText() == (
        f"Leave empty for {default_meetings_dir()}"
    )
    assert "Documents/Koe" not in window.snippets_input.placeholderText()
    assert "Documents/Koe" not in window.meetings_input.placeholderText()
    assert "esc to close" not in labels
    assert not hasattr(window, "provider_combo")

    window.close()


def test_escape_closes_settings_without_emitting_save(qapp):
    from ui.settings_window import SettingsWindow

    window = SettingsWindow()
    saved = []
    window.settings_saved.connect(lambda: saved.append(True))
    window.show()
    qapp.processEvents()

    QTest.keyClick(window, Qt.Key_Escape)
    qapp.processEvents()

    assert not window.isVisible()
    assert saved == []


def test_settings_debounces_changes_and_saves_automatically(qapp, monkeypatch):
    from ui.settings_window import SettingsWindow
    from utils import ConfigManager

    writes = []
    monkeypatch.setattr(ConfigManager, "save_config", lambda: writes.append(True))
    window = SettingsWindow()
    closed_after_change = []
    window.settings_saved.connect(lambda: closed_after_change.append(True))
    window.show()
    qapp.processEvents()

    window.user_name_input.setText("Alex Autosave")
    assert window.save_status_label.text() == "Saving…"
    assert window.save_status_label.isVisible()
    QTest.qWait(420)

    assert writes == [True]
    assert ConfigManager.get_config_value("profile", "user_name") == "Alex Autosave"
    assert window.save_status_label.text() == "Saved"
    window.close()
    assert closed_after_change == [True]


def test_status_and_scribe_construct_with_new_copy(qapp, monkeypatch, tmp_path):
    from meeting.app import (
        MODE_IN_PERSON_GROUP,
        MODE_IN_PERSON_ONE_ON_ONE,
        MODE_ONLINE_GROUP,
        MODE_ONLINE_ONE_ON_ONE,
        ScribeWindow,
    )
    from ui.status_window import StatusWindow
    from utils import ConfigManager

    status = StatusWindow()
    status.updateStatus("recording")
    listening_width = status.width()
    listening_positions = [
        widget.mapTo(status, QPoint()).x()
        for widget in (
            status.indicator,
            status.status_label,
            status.timer_label,
            status.cancel_button,
        )
    ]
    assert status.status_label.text() == "Listening"
    assert status.cancel_button.isVisible()
    assert status.cancel_button.accessibleName() == "Cancel snippet"
    cancelled = []
    status.cancelSignal.connect(lambda: cancelled.append(True))
    QTest.mouseClick(status.cancel_button, Qt.LeftButton)
    assert cancelled == [True]
    assert not hasattr(status, "helper_label")
    assert status.width() == 207
    assert status.height() <= 54
    assert status.status_label.width() == 88
    assert (
        status.timer_label.mapTo(status, QPoint()).x()
        - (
            status.status_label.mapTo(status, QPoint()).x()
            + status.status_label.width()
        )
        == 7
    )
    assert "border: none" in status.timer_label.styleSheet().lower()
    assert "font-size: 10pt" in status.timer_label.styleSheet().lower()
    status.updateStatus("transcribing")
    assert status.status_label.text() == "Transcribing"
    assert status.cancel_button.isVisible()
    assert status.cancel_button.accessibleName() == "Dismiss transcription"
    assert status.width() == listening_width
    assert [
        widget.mapTo(status, QPoint()).x()
        for widget in (
            status.indicator,
            status.status_label,
            status.timer_label,
            status.cancel_button,
        )
    ] == listening_positions
    assert "elevenlabs" not in status.status_label.text().lower()
    dismissed = []
    status.dismissSignal.connect(lambda: dismissed.append(True))
    QTest.mouseClick(status.cancel_button, Qt.LeftButton)
    assert dismissed == [True]
    assert not status.isVisible()
    assert "#f0a7ae" in status.cancel_button.styleSheet().lower()

    scribe = ScribeWindow(MODE_ONLINE_ONE_ON_ONE)
    scribe.show()
    qapp.processEvents()
    original_size = scribe.size()
    scribe.resize(1100, 800)
    qapp.processEvents()
    assert scribe.size() == original_size == QSize(760, 590)
    assert scribe.minimumSize() == scribe.maximumSize() == scribe.size()
    assert not scribe.windowFlags() & Qt.WindowMaximizeButtonHint
    assert scribe.windowFlags() & Qt.MSWindowsFixedSizeDialogHint
    labels = " ".join(label.text() for label in scribe.findChildren(QLabel))
    assert scribe.record_button.text() == "Start"
    assert scribe.record_button.objectName() == "startButton"
    assert scribe.record_button.size().width() == 72
    assert scribe.meeting_name_input.placeholderText().startswith(
        "e.g. Invoice workflow"
    )
    assert scribe.meeting_field_label.text() == "Meeting name"
    assert scribe.participant_field_label.text() == "Participant name"
    assert scribe.participant_input.placeholderText() == "Full name works best"
    assert scribe.participant_input.isVisible()
    assert scribe.meeting_type_label.text() == "Meeting type"
    assert scribe.meeting_type_combo.accessibleName() == "Meeting type"
    assert scribe.meeting_type_combo.maximumWidth() == 360
    assert [
        scribe.meeting_type_combo.itemText(index)
        for index in range(scribe.meeting_type_combo.count())
    ] == [
        "Online — One-on-One",
        "Online — Group Meeting",
        "In Person — One-on-One",
        "In Person — Group Meeting",
    ]
    assert [
        scribe.meeting_type_combo.itemData(index)
        for index in range(scribe.meeting_type_combo.count())
    ] == [
        MODE_ONLINE_ONE_ON_ONE,
        MODE_ONLINE_GROUP,
        MODE_IN_PERSON_ONE_ON_ONE,
        MODE_IN_PERSON_GROUP,
    ]
    assert scribe.meeting_type_combo.currentData() == MODE_ONLINE_ONE_ON_ONE
    action_y = scribe.action_stack.mapTo(scribe, QPoint()).y()
    timer_y = scribe.timer_label.mapTo(scribe, QPoint()).y()
    divider_y = scribe.divider.mapTo(scribe, QPoint()).y()
    assert action_y + scribe.action_stack.height() // 2 == (
        timer_y + scribe.timer_label.height() // 2
    )
    scribe.meeting_type_combo.setCurrentIndex(
        scribe.meeting_type_combo.findData(MODE_ONLINE_GROUP)
    )
    qapp.processEvents()
    assert scribe.meeting_mode == MODE_ONLINE_GROUP
    assert not scribe.participant_input.isVisible()
    assert scribe.action_stack.mapTo(scribe, QPoint()).y() == action_y
    assert (
        ConfigManager.get_config_value("meeting_options", "last_meeting_mode")
        == MODE_ONLINE_GROUP
    )
    scribe.meeting_type_combo.setCurrentIndex(
        scribe.meeting_type_combo.findData(MODE_ONLINE_ONE_ON_ONE)
    )
    qapp.processEvents()
    assert scribe.participant_input.isVisible()
    assert scribe.action_stack.mapTo(scribe, QPoint()).y() == action_y
    assert "Meeting notes" in labels
    assert "Scribe" in labels
    assert "Capture the conversation" not in labels
    assert "Saved with the meeting" not in labels
    assert scribe.timer_label.objectName() == "scribeTimer"
    assert scribe.record_button.mapTo(scribe, QPoint()).x() < (
        scribe.timer_label.mapTo(scribe, QPoint()).x()
    )
    assert scribe.participant_input.maximumWidth() == 360
    assert "border: 1px" in scribe.styleSheet().lower()
    assert "background: transparent" in scribe.styleSheet().lower()
    assert "qcombobox#meetingtypeselector::drop-down" in scribe.styleSheet().lower()
    assert "border: none" in scribe.styleSheet().lower()

    scribe._set_record_button_state(recording=True)
    assert scribe.record_button.text() == "Stop"
    assert scribe.record_button.objectName() == "stopButton"
    assert not scribe.record_button.icon().isNull()
    assert not hasattr(scribe, "recording_indicator")
    assert scribe.record_button.size().width() == 72
    assert scribe.record_button.iconSize() == QSize(14, 10)
    scribe._set_record_button_state(recording=False)

    captured_flags = []
    monkeypatch.setattr(
        QInputDialog,
        "exec_",
        lambda dialog: captured_flags.append(dialog.windowFlags()) or QDialog.Rejected,
    )
    scribe._prompt_for_meeting_name()
    scribe._prompt_for_participant_name()
    assert len(captured_flags) == 2
    assert all(not flags & Qt.WindowContextHelpButtonHint for flags in captured_flags)

    scribe.timer_label.setText("12:34")
    scribe._show_processing()
    qapp.processEvents()
    assert scribe.working_status_label.text() == "Preparing"
    assert scribe.action_stack.currentWidget() is scribe.status_controls_widget
    assert scribe.timer_label.text() == "12:34"
    assert scribe.timer_label.isVisible()
    assert scribe.header_state_row.isVisible()
    assert not scribe.header_state_label.isVisible()
    assert scribe.divider.mapTo(scribe, QPoint()).y() == divider_y
    scribe._on_worker_status("Transcribing your audio...")
    assert scribe.working_status_label.text() == "Transcribing"
    scribe._on_worker_status("Transcribing other audio...")
    assert scribe.working_status_label.text() == "Transcribing"
    scribe._on_worker_status("Generating summary...")
    assert scribe.working_status_label.text() == "Summarising"
    assert (
        scribe._concise_error('Failed: ElevenLabs HTTP 400: {"detail":"invalid audio"}')
        == "Couldn’t process audio"
    )
    scribe._on_worker_error("No speech detected in either stream.")
    qapp.processEvents()
    assert scribe.action_stack.currentWidget() is scribe.record_controls_widget
    assert scribe.header_state_label.text() == "No speech detected"
    assert scribe.header_state_label.toolTip() == "No speech detected in either stream."
    assert scribe.header_state_row.isVisible()
    assert scribe.header_state_label.isVisible()
    no_speech_text_left = (
        scribe.header_state_label.mapTo(scribe, QPoint()).x()
        + scribe.header_state_label.contentsMargins().left()
    )
    assert no_speech_text_left == scribe.record_button.mapTo(scribe, QPoint()).x()
    assert scribe.record_button.text() == "Start"
    assert scribe.timer_label.text() == "00:00"
    assert scribe.meeting_name_input.isEnabled()
    assert scribe.participant_input.isEnabled()
    assert scribe.notes_edit.isEnabled()
    message_top = scribe.header_state_label.mapTo(scribe, QPoint()).y()
    message_bottom = message_top + scribe.header_state_label.height()
    timer_bottom = scribe.timer_label.mapTo(scribe, QPoint()).y() + scribe.timer_label.height()
    divider_top = scribe.divider.mapTo(scribe, QPoint()).y()
    assert divider_top == divider_y
    assert message_top - timer_bottom > divider_top - message_bottom
    assert divider_top - message_bottom <= 2
    assert not hasattr(scribe, "status_label")

    scribe.output_root = Path("C:/tmp/Meetings")
    scribe.meeting_started_at = datetime(2026, 7, 14, 9, 30)
    first_meeting_dir = scribe._meeting_directory_for_session("Management Meeting")
    scribe.recording_started_at = datetime(2026, 7, 14, 9, 45)
    second_meeting_dir = scribe._meeting_directory_for_session("Renamed Meeting")
    assert second_meeting_dir == first_meeting_dir

    scribe._on_worker_error("Provider request failed")
    qapp.processEvents()
    assert scribe.action_stack.currentWidget() is scribe.recovery_controls_widget
    assert scribe.recovery_button.text() == "Recovery Folder"
    assert scribe.recovery_button.height() == 34
    assert scribe.header_state_label.text() == "Couldn’t process audio"
    assert scribe.timer_label.isVisible()
    assert scribe.divider.mapTo(scribe, QPoint()).y() == divider_y
    processing_error_text_left = (
        scribe.header_state_label.mapTo(scribe, QPoint()).x()
        + scribe.header_state_label.contentsMargins().left()
    )
    assert processing_error_text_left == scribe.recovery_button.mapTo(
        scribe, QPoint()
    ).x()

    meeting_dir = tmp_path / "meeting"
    meeting_dir.mkdir()
    summary_pdf = meeting_dir / "summary.pdf"
    transcript_pdf = meeting_dir / "transcript.pdf"
    summary_pdf.write_bytes(b"summary")
    transcript_pdf.write_bytes(b"transcript")
    scribe._on_worker_done(str(meeting_dir), str(summary_pdf))
    qapp.processEvents()
    assert not scribe.action_stack.isVisible()
    assert not scribe.timer_label.isVisible()
    assert scribe.header_state_row.isVisible()
    assert not scribe.header_state_label.isVisible()
    assert scribe.divider.mapTo(scribe, QPoint()).y() == divider_y
    assert scribe.completion_options.isVisible()
    assert scribe.open_summary_button.text() == "Summary"
    assert scribe.open_transcript_button.text() == "Transcript"
    assert scribe.open_summary_button.accessibleName() == "Open summary"
    assert scribe.open_transcript_button.accessibleName() == "Open transcript"
    assert scribe.open_summary_button.objectName() == "summaryButton"
    assert scribe.open_transcript_button.objectName() == "transcriptButton"
    assert scribe.open_summary_button.height() == 34
    assert scribe.open_transcript_button.height() == 34
    assert scribe.completion_options.objectName() == "completionOptions"
    button_gap = scribe.open_transcript_button.mapTo(
        scribe.completion_options, QPoint()
    ).x() - (
        scribe.open_summary_button.mapTo(scribe.completion_options, QPoint()).x()
        + scribe.open_summary_button.width()
    )
    assert button_gap == 7
    completion_right = (
        scribe.completion_options.mapTo(scribe.centralWidget(), QPoint()).x()
        + scribe.completion_options.width()
    )
    assert scribe.centralWidget().width() - completion_right == 28
    assert scribe.open_summary_button.width() >= (
        scribe.open_summary_button.fontMetrics().horizontalAdvance("Summary") + 24
    )
    assert scribe.open_transcript_button.width() >= (
        scribe.open_transcript_button.fontMetrics().horizontalAdvance("Transcript")
        + 24
    )
    assert scribe.open_summary_button.width() < 120
    assert scribe.open_transcript_button.width() < 180
    assert scribe._done_summary_path == str(summary_pdf)
    assert scribe._done_transcript_path == str(transcript_pdf)

    from PyQt5.QtGui import QDesktopServices

    opened_paths = []
    monkeypatch.setattr(
        QDesktopServices,
        "openUrl",
        staticmethod(lambda url: opened_paths.append(url.toLocalFile()) or True),
    )
    QTest.mouseClick(scribe.open_summary_button, Qt.LeftButton)
    QTest.mouseClick(scribe.open_transcript_button, Qt.LeftButton)
    assert [Path(path) for path in opened_paths] == [summary_pdf, transcript_pdf]

    transcript_pdf.unlink()
    transcript_markdown = meeting_dir / "transcript.md"
    transcript_markdown.write_text("transcript", encoding="utf-8")
    assert (
        scribe._preferred_document_path(str(meeting_dir), "transcript")
        == str(transcript_markdown)
    )
    assert not hasattr(scribe, "close_button")

    class FakeCapture:
        def __init__(self, _temp_dir):
            pass

        @staticmethod
        def start():
            return True

    monkeypatch.setattr("meeting.app.AudioCapture", FakeCapture)
    scribe._start_recording()
    assert not scribe.meeting_type_combo.isEnabled()
    scribe.elapsed_timer.stop()
    scribe.recording_pulse_timer.stop()
    scribe.capture = None
    scribe.close()

    group_scribe = ScribeWindow(MODE_ONLINE_GROUP)
    assert group_scribe.meeting_field_label.text() == "Meeting name"
    assert group_scribe.meeting_name_input.placeholderText().startswith(
        "e.g. Invoice workflow"
    )
    assert group_scribe.meeting_type_combo.currentData() == MODE_ONLINE_GROUP
    assert not group_scribe.participant_input.isVisible()
    group_scribe.close()

    in_person_scribe = ScribeWindow(MODE_IN_PERSON_GROUP)
    assert in_person_scribe.meeting_field_label.text() == "Meeting name"
    assert in_person_scribe.meeting_name_input.placeholderText().startswith(
        "e.g. Invoice workflow"
    )
    assert in_person_scribe.meeting_type_combo.currentData() == MODE_IN_PERSON_GROUP
    assert not in_person_scribe.participant_input.isVisible()
    in_person_scribe.close()

    in_person_one_on_one = ScribeWindow(MODE_IN_PERSON_ONE_ON_ONE)
    in_person_one_on_one.show()
    qapp.processEvents()
    assert (
        in_person_one_on_one.meeting_type_combo.currentData()
        == MODE_IN_PERSON_ONE_ON_ONE
    )
    assert in_person_one_on_one.participant_input.isVisible()
    in_person_one_on_one.close()


def test_initialization_card_is_compact_and_uses_the_koe_icon(qapp):
    from ui.initialization_window import InitializationWindow

    window = InitializationWindow()
    labels = " ".join(
        label.text() for label in window.findChildren(QLabel) if label.text()
    )

    assert window.size() == QSize(184, 64)
    assert labels == "Koe Initializing…"
    assert "Getting things ready" not in labels
    assert window.icon_label.pixmap() is not None
    assert not window.icon_label.pixmap().isNull()
    window._do_close()


def test_tray_menu_uses_shared_non_terminal_theme(qapp):
    from ui import theme

    menu = QMenu()
    menu.setStyleSheet(theme.tray_menu_stylesheet())
    menu.addAction("Start Scribe")
    menu.addAction("Settings")

    stylesheet = menu.styleSheet().lower()
    assert "segoe ui" in stylesheet
    assert theme.ACCENT_SOFT.lower() in stylesheet
    assert theme.BORDER_COLOR.lower() in stylesheet


def test_tray_menu_position_opens_above_taskbar_and_stays_on_screen():
    from main import tray_menu_position

    available = QRect(0, 0, 1920, 1032)
    cursor = QPoint(1810, 1060)
    menu_size = QSize(168, 156)
    position = tray_menu_position(cursor, menu_size, available)

    assert position.y() + menu_size.height() - 1 <= available.bottom()
    assert position.y() < cursor.y()
    assert available.left() <= position.x()
    assert position.x() + menu_size.width() - 1 <= available.right()


def test_applying_settings_never_stops_an_active_snippet(monkeypatch):
    import main

    class Listener:
        reloads = 0

        def load_activation_keys(self):
            self.reloads += 1

    class ActiveThread:
        stop_calls = 0

        def stop(self):
            self.stop_calls += 1

    class FakeApp:
        _components_initialized = True
        key_listener = Listener()
        result_thread = ActiveThread()

    fake = FakeApp()
    monkeypatch.setattr(main, "_debug", lambda _message: None)
    main.KoeApp.apply_settings(fake)

    assert fake.key_listener.reloads == 1
    assert fake.result_thread.stop_calls == 0
    assert not hasattr(main.KoeApp, "restart_app")
    assert not hasattr(main.KoeApp, "stop_result_thread")


def test_only_tray_exit_is_wired_to_process_shutdown():
    import inspect

    import main

    initialize_source = inspect.getsource(main.KoeApp.initialize_components)
    tray_source = inspect.getsource(main.KoeApp.create_tray_icon)

    assert "closeApp.connect(self.exit_app)" not in initialize_source
    assert "exit_action.triggered.connect(self.exit_app)" in tray_source


def test_dismissed_transcription_never_reaches_clipboard_or_beep(monkeypatch):
    import main
    from utils import ConfigManager

    class Listener:
        starts = 0

        def start(self):
            self.starts += 1

    class StatusWindow:
        @staticmethod
        def isVisible():
            return False

    class FakeApp:
        recording_start_time = 1.0
        processing_result = False
        suppress_current_result = True
        key_listener = Listener()
        status_window = StatusWindow()
        copied = []

        def _copy_to_clipboard(self, value):
            self.copied.append(value)
            return True

    def fake_config(*keys):
        values = {
            ("misc", "noise_on_completion"): False,
        }
        return values.get(tuple(keys))

    fake = FakeApp()
    monkeypatch.setattr(ConfigManager, "get_config_value", fake_config)
    monkeypatch.setattr(main, "_debug", lambda _message: None)

    main.KoeApp.on_transcription_complete(fake, "Do not copy me")

    assert fake.copied == []
    assert fake.key_listener.starts == 1
    assert fake.suppress_current_result is False


def test_completion_sound_is_short_and_bounded():
    sound_path = Path(__file__).parent.parent.parent / "assets" / "beep.wav"
    with wave.open(str(sound_path), "rb") as sound:
        duration = sound.getnframes() / sound.getframerate()
        samples = array("h", sound.readframes(sound.getnframes()))

    assert 0.2 <= duration <= 0.8
    assert max(abs(sample) for sample in samples) / 32768 < 0.75
