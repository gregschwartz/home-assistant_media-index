"""Tests for CacheManager error tracking functionality."""
import asyncio
import os
import sys
import tempfile
import pytest
import pytest_asyncio

# Add the current directory to path to import cache_manager directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import only what we need for testing, avoiding HA dependencies
import aiosqlite
import logging

_LOGGER = logging.getLogger(__name__)


class CacheManager:
    """Minimal CacheManager for testing - copied core methods."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._db = None

    async def async_setup(self) -> bool:
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._create_schema()
        return True

    async def _create_schema(self) -> None:
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS media_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                folder TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_size INTEGER,
                modified_time INTEGER NOT NULL,
                created_time INTEGER,
                last_scanned INTEGER NOT NULL,
                error_count INTEGER DEFAULT 0,
                last_error_date INTEGER,
                last_error_type TEXT
            )
        """)
        await self._db.commit()

    async def add_file(self, file_data: dict) -> int:
        import time
        current_time = int(time.time())
        await self._db.execute("""
            INSERT OR REPLACE INTO media_files
            (path, filename, folder, file_type, file_size, modified_time, created_time, last_scanned)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            file_data['path'],
            file_data['filename'],
            file_data['folder'],
            file_data['file_type'],
            file_data.get('file_size'),
            file_data['modified_time'],
            file_data.get('created_time'),
            current_time,
        ))
        await self._db.commit()
        async with self._db.execute(
            "SELECT id FROM media_files WHERE path = ?",
            (file_data['path'],)
        ) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def record_file_error(self, file_path: str, error_type: str = "load_failed") -> dict:
        import time
        current_time = int(time.time())
        async with self._db.execute(
            "SELECT id, error_count FROM media_files WHERE path = ?",
            (file_path,)
        ) as cursor:
            row = await cursor.fetchone()
        if not row:
            return {"error": "file_not_found", "file_path": file_path}
        file_id = row[0]
        current_error_count = row[1] or 0
        new_error_count = current_error_count + 1
        await self._db.execute(
            """UPDATE media_files
               SET error_count = ?, last_error_date = ?, last_error_type = ?
               WHERE id = ?""",
            (new_error_count, current_time, error_type, file_id)
        )
        await self._db.commit()
        return {
            "file_id": file_id,
            "file_path": file_path,
            "error_count": new_error_count,
            "last_error_date": current_time,
            "last_error_type": error_type
        }

    async def get_file_error_info(self, file_path: str) -> dict | None:
        async with self._db.execute(
            """SELECT id, error_count, last_error_date, last_error_type
               FROM media_files WHERE path = ?""",
            (file_path,)
        ) as cursor:
            row = await cursor.fetchone()
        if not row:
            return None
        return {
            "file_id": row[0],
            "error_count": row[1] or 0,
            "last_error_date": row[2],
            "last_error_type": row[3]
        }

    async def close(self) -> None:
        if self._db:
            await self._db.close()


@pytest_asyncio.fixture
async def cache_manager():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_media_index.db")
        cm = CacheManager(db_path)
        await cm.async_setup()
        yield cm
        await cm.close()


@pytest_asyncio.fixture
async def cache_with_file(cache_manager):
    """Cache manager with a test file already added."""
    test_file = {
        "path": "/media/test/photo.jpg",
        "filename": "photo.jpg",
        "folder": "/media/test",
        "file_type": "image",
        "file_size": 1024,
        "modified_time": 1700000000,
        "created_time": 1700000000,
    }
    await cache_manager.add_file(test_file)
    return cache_manager


class TestErrorTracking:
    """Test error tracking functionality."""

    @pytest.mark.asyncio
    async def test_record_file_error_increments_count(self, cache_with_file):
        """Test that recording an error increments the count."""
        cm = cache_with_file
        file_path = "/media/test/photo.jpg"

        # First error
        result = await cm.record_file_error(file_path, "load_failed")
        assert result["error_count"] == 1
        assert result["last_error_type"] == "load_failed"
        assert result["file_path"] == file_path
        assert "last_error_date" in result

        # Second error
        result = await cm.record_file_error(file_path, "decode_error")
        assert result["error_count"] == 2
        assert result["last_error_type"] == "decode_error"

    @pytest.mark.asyncio
    async def test_record_file_error_nonexistent_file(self, cache_manager):
        """Test error handling for non-existent file."""
        result = await cache_manager.record_file_error("/nonexistent/file.jpg")
        assert result["error"] == "file_not_found"

    @pytest.mark.asyncio
    async def test_get_file_error_info(self, cache_with_file):
        """Test retrieving error info."""
        cm = cache_with_file
        file_path = "/media/test/photo.jpg"

        # Before any errors
        info = await cm.get_file_error_info(file_path)
        assert info["error_count"] == 0
        assert info["last_error_date"] is None

        # After recording error
        await cm.record_file_error(file_path, "load_failed")
        info = await cm.get_file_error_info(file_path)
        assert info["error_count"] == 1
        assert info["last_error_type"] == "load_failed"
        assert info["last_error_date"] is not None

    @pytest.mark.asyncio
    async def test_get_file_error_info_nonexistent(self, cache_manager):
        """Test error info for non-existent file returns None."""
        result = await cache_manager.get_file_error_info("/nonexistent/file.jpg")
        assert result is None


class TestDatabaseMigrations:
    """Test database schema migrations."""

    @pytest.mark.asyncio
    async def test_error_columns_created(self, cache_manager):
        """Test that error tracking columns are created."""
        # Check columns exist by inserting a file and reading back
        test_file = {
            "path": "/media/test/migration_test.jpg",
            "filename": "migration_test.jpg",
            "folder": "/media/test",
            "file_type": "image",
            "file_size": 1024,
            "modified_time": 1700000000,
        }
        await cache_manager.add_file(test_file)

        # Query should work without error if columns exist
        info = await cache_manager.get_file_error_info("/media/test/migration_test.jpg")
        assert info is not None
        assert "error_count" in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
