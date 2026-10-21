"""Scan gating policy, kept HA-free so it is unit-testable."""


def scan_blocked(libmediainfo_available: bool, scan_without: bool) -> bool:
    """Whether a scan must be refused.

    Never: pymediainfo ships no musllinux wheel, so on HAOS (Alpine) libmediainfo
    is usually absent. Blocking would stop image indexing; videos merely lose
    pymediainfo metadata (mutagen fallback remains).
    """
    return False
