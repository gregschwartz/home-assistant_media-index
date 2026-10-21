import importlib.util
from pathlib import Path

# Load by path: package __init__ imports homeassistant, which unit tests avoid.
_p = Path(__file__).parents[2] / "custom_components/media_index/scan_policy.py"
_s = importlib.util.spec_from_file_location("scan_policy", _p)
sp = importlib.util.module_from_spec(_s)
_s.loader.exec_module(sp)


def test_never_blocks_when_libmediainfo_missing_even_if_option_false() -> None:
    # musl HAOS has no pymediainfo wheel; images must still index.
    assert sp.scan_blocked(libmediainfo_available=False, scan_without=False) is False


def test_never_blocks_when_available() -> None:
    assert sp.scan_blocked(libmediainfo_available=True, scan_without=False) is False
