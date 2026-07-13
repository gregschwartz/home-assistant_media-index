"""Unit tests for CacheManager error tracking and face data (bad-image flow).

Uses a real SQLite database in a pytest tmp_path, matching the conventions
in test_cache_manager.py.
"""

import os
import pytest_asyncio

import importlib.util, os as _os
def _load(name):
    path = _os.path.join(_os.path.dirname(__file__), '..', '..', 'custom_components', 'media_index', f'{name}.py')
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
CacheManager = _load('cache_manager').CacheManager


@pytest_asyncio.fixture
async def cache(tmp_path):
    """Return an initialised CacheManager backed by a temp SQLite file."""
    db_path = str(tmp_path / "test_media_index.db")
    mgr = CacheManager(db_path)
    ok = await mgr.async_setup()
    assert ok, "CacheManager.async_setup() returned False"
    yield mgr
    await mgr.close()


def _file_data(path: str) -> dict:
    return {
        "path": path,
        "filename": os.path.basename(path),
        "folder": os.path.dirname(path),
        "file_type": "image",
        "file_size": 1_000_000,
        "modified_time": "2023-06-23T10:00:00",
        "created_time": "2023-06-23T10:00:00",
        "width": 4000,
        "height": 3000,
        "orientation": "normal",
        "is_favorited": 0,
        "rating": 0,
    }


class TestErrorTracking:

    async def test_record_error_increments_count(self, cache):
        path = "/media/photo/Test/photo.jpg"
        await cache.add_file(_file_data(path))

        result = await cache.record_file_error(path, "load_failed")
        assert result["error_count"] == 1
        assert result["last_error_type"] == "load_failed"

        result = await cache.record_file_error(path, "decode_error")
        assert result["error_count"] == 2
        assert result["last_error_type"] == "decode_error"

    async def test_record_error_unknown_file(self, cache):
        result = await cache.record_file_error("/nonexistent/file.jpg")
        assert result["error"] == "file_not_found"

    async def test_get_file_error_info(self, cache):
        path = "/media/photo/Test/photo.jpg"
        await cache.add_file(_file_data(path))

        info = await cache.get_file_error_info(path)
        assert info["error_count"] == 0
        assert info["last_error_date"] is None

        await cache.record_file_error(path, "load_failed")
        info = await cache.get_file_error_info(path)
        assert info["error_count"] == 1
        assert info["last_error_type"] == "load_failed"
        assert info["last_error_date"] is not None

    async def test_get_file_error_info_unknown_file(self, cache):
        assert await cache.get_file_error_info("/nonexistent/file.jpg") is None


class TestFaceData:

    async def test_update_and_get_crop_info(self, cache):
        path = "/media/photo/Test/portrait.jpg"
        await cache.add_file(_file_data(path))

        ok = await cache.update_face_data(
            file_path=path,
            face_count=1,
            face_center_x=0.45,
            face_center_y=0.3,
            css_position="45.0% 30.0%",
            faces_json='{"face_count": 1}',
        )
        assert ok

        info = await cache.get_crop_info(path)
        assert info["face_count"] == 1
        assert info["css_position"] == "45.0% 30.0%"
        assert info["face_center_x"] == 0.45

    async def test_crop_info_defaults_to_center(self, cache):
        path = "/media/photo/Test/landscape.jpg"
        await cache.add_file(_file_data(path))

        info = await cache.get_crop_info(path)
        assert info["css_position"] == "50% 50%"
        assert info["face_count"] == 0

    async def test_crop_info_unknown_file(self, cache):
        assert await cache.get_crop_info("/nonexistent/file.jpg") is None
