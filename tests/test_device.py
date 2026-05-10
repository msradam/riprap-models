from riprap_models.device import DeviceInfo, get_device


def test_get_device_returns_known_kind():
    info = get_device()
    assert isinstance(info, DeviceInfo)
    assert info.kind in ("cuda", "mps", "cpu")
    assert info.dtype in ("float16", "bfloat16", "float32")
    assert info.label  # non-empty


def test_force_cpu():
    info = get_device(prefer="cpu")
    assert info.kind == "cpu"
    assert info.dtype == "float32"
