"""Small local command channel used by Koe's desktop shortcuts."""

from __future__ import annotations

import socket
import threading

from PyQt5.QtCore import QObject, pyqtSignal


COMMAND_HOST = "127.0.0.1"
COMMAND_PORT = 9877
VALID_COMMANDS = {"snippet", "scribe", "activate"}


def send_command(
    command: str,
    timeout: float = 0.6,
    *,
    host: str = COMMAND_HOST,
    port: int = COMMAND_PORT,
) -> bool:
    if command not in VALID_COMMANDS:
        raise ValueError(f"Unsupported Koe command: {command}")
    try:
        with socket.create_connection((host, port), timeout=timeout) as client:
            client.sendall((command + "\n").encode("utf-8"))
            client.settimeout(timeout)
            return client.recv(16).strip() == b"ok"
    except OSError:
        return False


class CommandServer(QObject):
    command_received = pyqtSignal(str)

    def __init__(self, host: str = COMMAND_HOST, port: int = COMMAND_PORT) -> None:
        super().__init__()
        self.host = host
        self.port = port
        self._socket: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()

    def start(self) -> bool:
        if self._socket is not None:
            return True
        self._stopping.clear()
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind((self.host, self.port))
            server.listen(4)
            server.settimeout(0.5)
        except OSError:
            server.close()
            return False
        self._socket = server
        self.port = int(server.getsockname()[1])
        self._thread = threading.Thread(target=self._serve, name="KoeCommandServer", daemon=True)
        self._thread.start()
        return True

    def _serve(self) -> None:
        while not self._stopping.is_set():
            try:
                assert self._socket is not None
                connection, _address = self._socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with connection:
                try:
                    payload = connection.recv(64).decode("utf-8", errors="ignore").strip()
                    if payload in VALID_COMMANDS:
                        connection.sendall(b"ok\n")
                        self.command_received.emit(payload)
                    else:
                        connection.sendall(b"invalid\n")
                except OSError:
                    continue

    def stop(self) -> None:
        self._stopping.set()
        server = self._socket
        self._socket = None
        if server is not None:
            try:
                server.close()
            except OSError:
                pass
        thread = self._thread
        self._thread = None
        if thread and thread.is_alive():
            thread.join(timeout=1.0)
