"""Tests for face detection module."""
import json
import os
import sys
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Mock cv2 before importing face_detector
sys.modules['cv2'] = MagicMock()
sys.modules['numpy'] = MagicMock()

# Now import the module (it will use mocked cv2)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'custom_components', 'media_index'))
from face_detector import (
    FaceInfo,
    FaceDetectionResult,
    FaceDetector,
    detect_faces_in_image,
    faces_to_json,
    OPENCV_AVAILABLE,
)


class TestFaceInfo:
    """Test FaceInfo dataclass."""

    def test_face_info_creation(self):
        """Test creating a FaceInfo object."""
        face = FaceInfo(x=100, y=50, width=200, height=250, center_x=0.5, center_y=0.3)
        assert face.x == 100
        assert face.y == 50
        assert face.width == 200
        assert face.height == 250
        assert face.center_x == 0.5
        assert face.center_y == 0.3


class TestFaceDetectionResult:
    """Test FaceDetectionResult dataclass."""

    def test_result_to_dict_no_faces(self):
        """Test converting result with no faces to dict."""
        result = FaceDetectionResult(
            face_count=0,
            primary_face=None,
            all_faces=[],
            css_position="50% 50%",
            image_width=1920,
            image_height=1080,
        )
        d = result.to_dict()
        assert d["face_count"] == 0
        assert d["primary_face"] is None
        assert d["all_faces"] == []
        assert d["css_position"] == "50% 50%"

    def test_result_to_dict_with_faces(self):
        """Test converting result with faces to dict."""
        face1 = FaceInfo(x=100, y=50, width=200, height=250, center_x=0.25, center_y=0.2)
        face2 = FaceInfo(x=500, y=100, width=150, height=180, center_x=0.6, center_y=0.3)

        result = FaceDetectionResult(
            face_count=2,
            primary_face=face1,
            all_faces=[face1, face2],
            css_position="42.5% 25%",
            image_width=1920,
            image_height=1080,
        )
        d = result.to_dict()
        assert d["face_count"] == 2
        assert d["primary_face"]["center_x"] == 0.25
        assert len(d["all_faces"]) == 2


class TestCSSPositionCalculation:
    """Test CSS position calculation logic."""

    def test_no_faces_returns_center(self):
        """Test that no faces returns center position."""
        detector = FaceDetector.__new__(FaceDetector)  # Create without __init__
        result = detector._calculate_css_position([], 1920, 1080)
        assert result == "50% 50%"

    def test_single_face_uses_face_center(self):
        """Test single face uses face center for CSS position."""
        detector = FaceDetector.__new__(FaceDetector)
        face = FaceInfo(x=100, y=50, width=200, height=250, center_x=0.45, center_y=0.28)
        result = detector._calculate_css_position([face], 1920, 1080)
        assert result == "45.0% 28.0%"

    def test_multiple_faces_uses_bounding_box(self):
        """Test multiple faces uses bounding box center."""
        detector = FaceDetector.__new__(FaceDetector)
        face1 = FaceInfo(x=100, y=100, width=100, height=100, center_x=0.15, center_y=0.15)
        face2 = FaceInfo(x=500, y=200, width=100, height=100, center_x=0.55, center_y=0.25)
        # Bounding box: x=100 to 600, y=100 to 300
        # Center: (350, 200) on 1000x1000 image -> 35%, 20%
        result = detector._calculate_css_position([face1, face2], 1000, 1000)
        assert "35.0%" in result
        assert "20.0%" in result


class TestFacesToJson:
    """Test JSON serialization."""

    def test_none_result(self):
        """Test None result returns None."""
        assert faces_to_json(None) is None

    def test_valid_result(self):
        """Test valid result returns JSON string."""
        result = FaceDetectionResult(
            face_count=1,
            primary_face=FaceInfo(x=100, y=50, width=200, height=250, center_x=0.5, center_y=0.3),
            all_faces=[FaceInfo(x=100, y=50, width=200, height=250, center_x=0.5, center_y=0.3)],
            css_position="50% 30%",
            image_width=800,
            image_height=600,
        )
        json_str = faces_to_json(result)
        assert json_str is not None

        parsed = json.loads(json_str)
        assert parsed["face_count"] == 1
        assert parsed["css_position"] == "50% 30%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
