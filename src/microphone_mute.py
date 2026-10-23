"""Windows endpoint mute control for the exact PortAudio WASAPI input."""

import ctypes
from contextlib import contextmanager
from uuid import UUID


class _GUID(ctypes.Structure):
    _fields_ = [
        ("data1", ctypes.c_uint32),
        ("data2", ctypes.c_uint16),
        ("data3", ctypes.c_uint16),
        ("data4", ctypes.c_ubyte * 8),
    ]


_ENDPOINT_VOLUME_IID = _GUID.from_buffer_copy(
    UUID("5CDF2C82-841E-4546-9722-0CF74078229A").bytes_le
)
_HRESULT = ctypes.c_int32
_BOOL = ctypes.c_int32
_RPC_E_CHANGED_MODE = -2147417850


def _check_hresult(result, operation):
    if result < 0:
        raise OSError(f"{operation} failed (HRESULT 0x{result & 0xffffffff:08X})")


def _com_method(pointer, slot, restype, *argtypes):
    vtable = ctypes.cast(
        pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
    ).contents
    return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtable[slot])


@contextmanager
def _endpoint_volume(portaudio_library, device_index):
    """Borrow PortAudio's IMMDevice; own and release only the activated volume."""
    ole32 = ctypes.WinDLL("ole32")
    ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    # Use an integer result so RPC_E_CHANGED_MODE can use the existing apartment.
    ole32.CoInitializeEx.restype = _HRESULT
    ole32.CoUninitialize.argtypes = []
    ole32.CoUninitialize.restype = None
    initialized = ole32.CoInitializeEx(None, 0)  # COINIT_MULTITHREADED
    if initialized != _RPC_E_CHANGED_MODE:
        _check_hresult(initialized, "Initialize microphone mute control")

    volume = ctypes.c_void_p()
    try:
        library = ctypes.CDLL(portaudio_library)
        get_device = library.PaWasapi_GetIMMDevice
        get_device.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
        get_device.restype = ctypes.c_int
        device = ctypes.c_void_p()
        result = get_device(int(device_index), ctypes.byref(device))
        if result != 0 or not device.value:
            raise OSError(f"Resolve microphone endpoint failed (PortAudio {result})")

        # IMMDevice::Activate (IUnknown has three preceding vtable entries).
        activate = _com_method(
            device, 3, _HRESULT, ctypes.POINTER(_GUID), ctypes.c_uint32,
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
        )
        _check_hresult(
            activate(device, ctypes.byref(_ENDPOINT_VOLUME_IID), 23, None,
                     ctypes.byref(volume)),  # CLSCTX_ALL
            "Open microphone mute control",
        )
        if not volume.value:
            raise OSError("Windows returned no microphone mute control")
        yield volume
    finally:
        try:
            if volume.value:
                _com_method(volume, 2, ctypes.c_uint32)(volume)  # Release
        finally:
            if initialized >= 0:
                ole32.CoUninitialize()


def _get_mute(volume):
    muted = _BOOL()
    _check_hresult(
        _com_method(volume, 15, _HRESULT, ctypes.POINTER(_BOOL))(
            volume, ctypes.byref(muted)
        ),  # IAudioEndpointVolume::GetMute
        "Read microphone mute state",
    )
    return bool(muted.value)


def ensure_microphone_unmuted(portaudio_library, device_index):
    """Clear Windows mute once; return whether it was changed. Never change gain."""
    with _endpoint_volume(portaudio_library, device_index) as volume:
        if not _get_mute(volume):
            return False
        _check_hresult(
            _com_method(volume, 14, _HRESULT, _BOOL, ctypes.c_void_p)(
                volume, False, None
            ),  # IAudioEndpointVolume::SetMute
            "Unmute microphone",
        )
        if _get_mute(volume):
            raise OSError("Microphone remains muted in Windows")
        return True
