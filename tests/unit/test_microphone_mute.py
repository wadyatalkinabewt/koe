"""Mute-control checks use synthetic COM objects, never the real microphone."""

import ctypes
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import microphone_mute as mute


@pytest.mark.parametrize(
    "initially_muted,sticky,set_result,get_result,error",
    [
        (False, False, 0, 0, None),
        (True, False, 0, 0, None),
        (True, True, 0, 0, "remains muted"),
        (True, False, -2147024891, 0, "Unmute microphone failed"),
        (True, False, 0, -2147024891, "Read microphone mute state failed"),
    ],
)
def test_unmute_uses_windows_com_abi_and_checks_result(
    monkeypatch, initially_muted, sticky, set_result, get_result, error
):
    state = {"muted": initially_muted}
    writes = []

    @ctypes.WINFUNCTYPE(mute._HRESULT, ctypes.c_void_p, ctypes.POINTER(mute._BOOL))
    def get_mute(_this, output):
        output[0] = state["muted"]
        return get_result

    @ctypes.WINFUNCTYPE(mute._HRESULT, ctypes.c_void_p, mute._BOOL, ctypes.c_void_p)
    def set_mute(_this, value, context):
        writes.append((value, context))
        if not sticky and set_result >= 0:
            state["muted"] = bool(value)
        return set_result

    # Only GetMute/SetMute exist: any volume/gain operation fails this test.
    vtable = (ctypes.c_void_p * 16)()
    vtable[14] = ctypes.cast(set_mute, ctypes.c_void_p)
    vtable[15] = ctypes.cast(get_mute, ctypes.c_void_p)
    table_pointer = ctypes.cast(vtable, ctypes.POINTER(ctypes.c_void_p))
    interface = ctypes.pointer(table_pointer)
    volume = ctypes.cast(interface, ctypes.c_void_p)
    closed = []

    @contextmanager
    def fake_volume(library, device):
        assert (library, device) == ("test-portaudio", 25)
        try:
            yield volume
        finally:
            closed.append(True)

    monkeypatch.setattr(mute, "_endpoint_volume", fake_volume)
    if error:
        with pytest.raises(OSError, match=error):
            mute.ensure_microphone_unmuted("test-portaudio", 25)
    else:
        assert mute.ensure_microphone_unmuted("test-portaudio", 25) is initially_muted
        assert state["muted"] is False
    assert closed == [True]
    assert writes == ([(0, None)] if initially_muted and get_result >= 0 else [])


@pytest.mark.parametrize("initialized", [0, 1, mute._RPC_E_CHANGED_MODE])
@pytest.mark.parametrize("activation_fails", [False, True])
def test_endpoint_volume_balances_com_and_borrowed_device(
    monkeypatch, initialized, activation_fails
):
    calls = []

    def initialize(_reserved, flags):
        assert flags == 0
        return initialized

    def uninitialize():
        calls.append("uninitialize")

    def get_device(index, output):
        assert index == 25
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = 100
        return 0

    def method(pointer, slot, _restype, *_argtypes):
        if pointer.value == 100:
            assert slot == 3  # Borrowed IMMDevice must never be released.

            def activate(_this, _iid, context, _params, output):
                assert context == 23
                if activation_fails:
                    return -2147024891
                ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = 200
                return 0

            return activate
        assert pointer.value == 200 and slot == 2
        return lambda _this: calls.append("release") or 0

    monkeypatch.setattr(mute.ctypes, "WinDLL", lambda _name: SimpleNamespace(
        CoInitializeEx=initialize, CoUninitialize=uninitialize
    ))
    monkeypatch.setattr(mute.ctypes, "CDLL", lambda _name: SimpleNamespace(
        PaWasapi_GetIMMDevice=get_device
    ))
    monkeypatch.setattr(mute, "_com_method", method)
    if activation_fails:
        with pytest.raises(OSError, match="Open microphone mute control"):
            with mute._endpoint_volume("test-portaudio", 25):
                pytest.fail("activation should fail")
    else:
        with pytest.raises(RuntimeError, match="test body"):
            with mute._endpoint_volume("test-portaudio", 25) as volume:
                assert volume.value == 200
                raise RuntimeError("test body")
    expected = [] if activation_fails else ["release"]
    if initialized >= 0:
        expected.append("uninitialize")
    assert calls == expected


@pytest.mark.parametrize("host,name,expected", [
    ("Windows WASAPI", "Headset", 0),
    ("Windows WDM-KS", " Headset ", 1),
    ("MME", "Missing mic", None),
])
def test_only_matching_capture_endpoint_is_unmuted(monkeypatch, host, name, expected):
    import result_thread as recording

    devices = [
        {"name": name, "hostapi": 0, "max_input_channels": 1},
        {"name": "Headset", "hostapi": 1, "max_input_channels": 1},
        {"name": "Other mic", "hostapi": 1, "max_input_channels": 1},
    ]
    apis = [{"name": host}, {"name": "Windows WASAPI"}]
    monkeypatch.setattr(recording.sd, "query_devices", lambda index=None:
                        devices if index is None else devices[index])
    monkeypatch.setattr(recording.sd, "query_hostapis", lambda index: apis[index])
    calls = []
    monkeypatch.setattr(recording, "ensure_microphone_unmuted",
                        lambda library, index: calls.append(index) or True)
    if expected is None:
        with pytest.raises(RuntimeError, match="uniquely identify"):
            recording._unmute_input_device(0)
    else:
        recording._unmute_input_device(0)
    assert calls == ([] if expected is None else [expected])


def test_ambiguous_legacy_mics_do_not_unmute_any_endpoint(monkeypatch):
    import result_thread as recording

    devices = [
        {"name": "Same name", "max_input_channels": 1},
        {"name": "Same name", "max_input_channels": 1},
        {"name": "Same name", "max_input_channels": 1},
    ]
    monkeypatch.setattr(recording.sd, "query_devices", lambda index=None:
                        devices if index is None else devices[index])
    monkeypatch.setattr(recording, "_device_host_api", lambda index:
                        "MME" if index == 0 else "Windows WASAPI")
    monkeypatch.setattr(recording, "ensure_microphone_unmuted",
                        lambda *_args: pytest.fail("must not guess a microphone"))
    with pytest.raises(RuntimeError, match="uniquely identify"):
        recording._unmute_input_device(0)
