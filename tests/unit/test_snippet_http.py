import pytest
import requests
from urllib3.poolmanager import pool_classes_by_scheme


@pytest.mark.parametrize("failure", [False, True])
def test_timing_is_local_and_logged_on_success_and_timeout(monkeypatch, failure):
    from providers import snippet_http

    global_pool = pool_classes_by_scheme["https"]
    adapters = []
    messages = []

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def mount(self, scheme, adapter):
            assert scheme == "https://"
            assert adapter.max_retries.total == 0
            adapters.append(adapter)

        def post(self, url, **kwargs):
            assert url == "https://example.invalid"
            assert kwargs["timeout"] == (10, 180)
            if failure:
                raise requests.ReadTimeout("test timeout")
            response = requests.Response()
            response.status_code = 200
            return response

    monkeypatch.setattr(snippet_http.requests, "Session", Session)
    for _ in range(2):
        if failure:
            with pytest.raises(requests.ReadTimeout):
                snippet_http.post("https://example.invalid", log=messages.append, timeout=(10, 180))
        else:
            assert snippet_http.post("https://example.invalid", log=messages.append, timeout=(10, 180)).status_code == 200
    assert len(messages) == 2
    assert all("response_wait=" in message for message in messages)
    assert all(("status=failed" if failure else "status=200") in message for message in messages)
    assert pool_classes_by_scheme["https"] is global_pool
    assert adapters[0].poolmanager.pool_classes_by_scheme["https"] is not adapters[1].poolmanager.pool_classes_by_scheme["https"]
