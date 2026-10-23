"""Per-request HTTPS timing for snippets, without global socket patches."""

from __future__ import annotations

import time
from collections.abc import Callable

import requests
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPSConnection
from urllib3.connectionpool import HTTPSConnectionPool


def post(url: str, *, log: Callable[[str], None], **kwargs) -> requests.Response:
    """Send one request; retain certificate checks and disable transport retries.

    Connection includes DNS/TCP/TLS. Upload measures socket writes, not server
    acknowledgement; response wait includes network and provider processing.
    Each request owns its session, pool and counters, including concurrent ones.
    """
    timings = {"connect": 0.0, "upload": 0.0, "response_wait": 0.0}

    class TimedConnection(HTTPSConnection):
        def connect(self):
            started = time.perf_counter()
            try:
                return super().connect()
            finally:
                timings["connect"] += time.perf_counter() - started

        def send(self, data):
            # Avoid counting a lazy connect as upload time as well.
            if self.sock is None:
                self.connect()
            started = time.perf_counter()
            try:
                return super().send(data)
            finally:
                timings["upload"] += time.perf_counter() - started

        def getresponse(self):
            started = time.perf_counter()
            try:
                return super().getresponse()
            finally:
                timings["response_wait"] += time.perf_counter() - started

    class TimedPool(HTTPSConnectionPool):
        ConnectionCls = TimedConnection

    class TimedAdapter(HTTPAdapter):
        def init_poolmanager(self, *args, **pool_kwargs):
            super().init_poolmanager(*args, **pool_kwargs)
            self.poolmanager.pool_classes_by_scheme = {
                **self.poolmanager.pool_classes_by_scheme,
                "https": TimedPool,
            }

        def proxy_manager_for(self, *args, **proxy_kwargs):
            manager = super().proxy_manager_for(*args, **proxy_kwargs)
            manager.pool_classes_by_scheme = {
                **manager.pool_classes_by_scheme,
                "https": TimedPool,
            }
            return manager

    started = time.perf_counter()
    response = None
    try:
        with requests.Session() as session:
            session.mount("https://", TimedAdapter(max_retries=0))
            response = session.post(url, **kwargs)
            return response
    finally:
        total = time.perf_counter() - started
        status = response.status_code if response is not None else "failed"
        log(
            f"HTTPS stages: connect={timings['connect']:.2f}s, "
            f"upload={timings['upload']:.2f}s, "
            f"response_wait={timings['response_wait']:.2f}s, "
            f"total={total:.2f}s, status={status}"
        )
