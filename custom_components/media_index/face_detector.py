"""Face detection module using OpenCV."""
import json
import logging
from dataclasses import dataclass
from pathlib import Path

_LOGGER = logging.getLogger(__name__)

# Try to import opencv, but handle gracefully if not available
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    _LOGGER.warning("opencv-python not installed. Face detection will be unavailable.")


@dataclass
class FaceInfo:
    """Information about a detected face."""
    x: int  # Top-left x coordinate
    y: int  # Top-left y coordinate
    width: int
    height: int
    center_x: float  # Normalized center x (0.0-1.0)
    center_y: float  # Normalized center y (0.0-1.0)


@dataclass
class FaceDetectionResult:
    """Result of face detection on an image."""
    face_count: int
    primary_face: FaceInfo | None  # Largest face
    all_faces: list[FaceInfo]
    css_position: str  # CSS background-position value
    image_width: int
    image_height: int

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "face_count": self.face_count,
            "primary_face": {
                "x": self.primary_face.x,
                "y": self.primary_face.y,
                "width": self.primary_face.width,
                "height": self.primary_face.height,
                "center_x": self.primary_face.center_x,
                "center_y": self.primary_face.center_y,
            } if self.primary_face else None,
            "all_faces": [
                {
                    "x": f.x,
                    "y": f.y,
                    "width": f.width,
                    "height": f.height,
                    "center_x": f.center_x,
                    "center_y": f.center_y,
                }
                for f in self.all_faces
            ],
            "css_position": self.css_position,
            "image_width": self.image_width,
            "image_height": self.image_height,
        }


class FaceDetector:
    """Detects faces in images using OpenCV's Haar cascade classifier."""

    def __init__(self):
        if not OPENCV_AVAILABLE:
            raise RuntimeError("opencv-python is required for face detection")

        # Load the pre-trained face detection cascade
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load cascade classifier from {cascade_path}")

    def detect_faces(self, image_path: str | Path) -> FaceDetectionResult:
        """Detect faces in an image file.

        Args:
            image_path: Path to the image file

        Returns:
            FaceDetectionResult with detected faces and CSS positioning
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        height, width = img.shape[:2]

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces with varying scale factors for better detection
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        all_faces = []
        for (x, y, w, h) in faces:
            center_x = (x + w / 2) / width
            center_y = (y + h / 2) / height
            all_faces.append(FaceInfo(
                x=int(x),
                y=int(y),
                width=int(w),
                height=int(h),
                center_x=round(center_x, 4),
                center_y=round(center_y, 4),
            ))

        # Sort by face size (largest first) to find primary face
        all_faces.sort(key=lambda f: f.width * f.height, reverse=True)

        primary_face = all_faces[0] if all_faces else None
        css_position = self._calculate_css_position(all_faces, width, height)

        return FaceDetectionResult(
            face_count=len(all_faces),
            primary_face=primary_face,
            all_faces=all_faces,
            css_position=css_position,
            image_width=width,
            image_height=height,
        )

    def _calculate_css_position(
        self,
        faces: list[FaceInfo],
        img_width: int,
        img_height: int
    ) -> str:
        """Calculate optimal CSS background-position based on detected faces.

        Args:
            faces: List of detected faces
            img_width: Image width in pixels
            img_height: Image height in pixels

        Returns:
            CSS background-position string (e.g., "45% 30%")
        """
        if not faces:
            return "50% 50%"  # Default to center

        if len(faces) == 1:
            # Single face: position based on face center
            face = faces[0]
            x_percent = round(face.center_x * 100, 1)
            y_percent = round(face.center_y * 100, 1)
            return f"{x_percent}% {y_percent}%"

        # Multiple faces: find bounding box center
        min_x = min(f.x for f in faces)
        min_y = min(f.y for f in faces)
        max_x = max(f.x + f.width for f in faces)
        max_y = max(f.y + f.height for f in faces)

        center_x = (min_x + max_x) / 2 / img_width
        center_y = (min_y + max_y) / 2 / img_height

        x_percent = round(center_x * 100, 1)
        y_percent = round(center_y * 100, 1)

        return f"{x_percent}% {y_percent}%"


def detect_faces_in_image(image_path: str | Path) -> FaceDetectionResult | None:
    """Convenience function to detect faces in an image.

    Args:
        image_path: Path to the image file

    Returns:
        FaceDetectionResult or None if detection fails
    """
    if not OPENCV_AVAILABLE:
        _LOGGER.debug("OpenCV not available, skipping face detection")
        return None

    try:
        detector = FaceDetector()
        return detector.detect_faces(image_path)
    except Exception as e:
        _LOGGER.warning("Face detection failed for %s: %s", image_path, e)
        return None


def faces_to_json(result: FaceDetectionResult | None) -> str | None:
    """Convert FaceDetectionResult to JSON string for database storage."""
    if result is None:
        return None
    return json.dumps(result.to_dict())
