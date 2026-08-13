"""Pytest global config: add project root to sys.path, stub cv2 if missing, and provide shared fixtures."""

import sys
import types
from pathlib import Path

import pytest

# Add project root to sys.path / 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Stub cv2 module when OpenCV is not installed / 缺失 cv2 时注入最小桩模块
try:
    import cv2  # noqa: F401  try importing real cv2
except ImportError:
    from unittest.mock import MagicMock

    _cv2_stub = types.ModuleType("cv2")

    # OpenCV constant stubs (values match the official constants)
    _CONSTS = {
        "CAP_PROP_FRAME_WIDTH": 3,
        "CAP_PROP_FRAME_HEIGHT": 4,
        "CAP_PROP_FPS": 5,
        "CAP_PROP_FRAME_COUNT": 7,
        "CAP_PROP_BUFFERSIZE": 38,
        "FILLED": -1,
        "LINE_AA": 4,
        "FONT_HERSHEY_SIMPLEX": 0,
    }
    for _name, _val in _CONSTS.items():
        setattr(_cv2_stub, _name, _val)

    # MagicMock placeholders for all functions / classes
    _cv2_stub.VideoCapture = MagicMock()
    _cv2_stub.VideoWriter = MagicMock()
    _cv2_stub.VideoWriter_fourcc = MagicMock(return_value=0)
    _cv2_stub.imshow = MagicMock()
    _cv2_stub.waitKey = MagicMock(return_value=255)
    _cv2_stub.destroyAllWindows = MagicMock()
    _cv2_stub.imwrite = MagicMock()
    _cv2_stub.rectangle = MagicMock()
    _cv2_stub.putText = MagicMock()
    _cv2_stub.getTextSize = MagicMock(return_value=((10, 10), 5))
    _cv2_stub.addWeighted = MagicMock()
    _cv2_stub.flip = MagicMock()

    sys.modules["cv2"] = _cv2_stub


@pytest.fixture
def project_root() -> Path:
    """Return the project root path / 返回项目根目录路径"""
    return PROJECT_ROOT


@pytest.fixture
def make_detection():
    """Factory fixture to build detector.Detection instances / 工厂 fixture 构造 Detection 实例"""
    from detector import Detection

    def _create(
        x1: float = 0.0,
        y1: float = 0.0,
        x2: float = 100.0,
        y2: float = 100.0,
        confidence: float = 0.9,
        class_id: int = 0,
        class_name: str = "person",
    ) -> Detection:
        return Detection(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            confidence=confidence,
            class_id=class_id,
            class_name=class_name,
        )

    return _create
