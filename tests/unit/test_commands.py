import time
from types import SimpleNamespace

from PyQt5.QtCore import QCoreApplication


def test_command_server_round_trip_uses_local_channel():
    from commands import CommandServer, send_command

    app = QCoreApplication.instance() or QCoreApplication([])
    received = []
    server = CommandServer(port=0)
    server.command_received.connect(received.append)
    try:
        assert server.start()
        assert send_command("scribe", host=server.host, port=server.port)
        deadline = time.monotonic() + 2.0
        while not received and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert received == ["scribe"]
    finally:
        server.stop()


def test_shortcut_commands_dispatch_to_existing_koe_actions():
    from main import KoeApp

    calls = []
    app = SimpleNamespace(
        _components_initialized=True,
        on_activation=lambda: calls.append("snippet"),
        start_meeting_mode=lambda: calls.append("scribe"),
    )

    KoeApp.handle_command(app, "snippet")
    KoeApp.handle_command(app, "scribe")

    assert calls == ["snippet", "scribe"]
