"""
pytest 全局配置
=================
1. 将项目根目录加入 sys.path, 使测试可直接 import 项目模块
2. 当环境中缺失 opencv-python (cv2) 时, 注入一个最小桩模块,
   保证依赖 cv2 的源码模块 (camera_capture / visualizer / main)
   能被正常导入, 从而测试其非硬件相关逻辑
3. 提供公共 fixtures
"""

import sys
import types
from pathlib import Path

import pytest


# ============================================================
# 1. 项目根目录加入 sys.path
# ============================================================
# conftest.py 所在目录为 tests/, 其上一级即项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 2. cv2 桩模块 (仅在真实 cv2 缺失时启用)
# ============================================================
# 让测试在 "无 OpenCV" 环境下也能导入 camera_capture.py / main.py
try:
    import cv2  # noqa: F401  试图导入真实 cv2
except ImportError:
    from unittest.mock import MagicMock

    _cv2_stub = types.ModuleType("cv2")

    # OpenCV VideoCapture / VideoWriter 属性常量桩 (取值与官方一致, 仅作占位)
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

    # 用 MagicMock 占位所有可能的函数 / 类, 调用时不报错
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


# ============================================================
# 3. 公共 fixtures
# ============================================================
@pytest.fixture
def project_root() -> Path:
    """返回项目根目录路径"""
    return PROJECT_ROOT


@pytest.fixture
def make_detection():
    """
    工厂 fixture: 快速构造 detector.Detection 实例

    用法:
        det = make_detection(x1=10, y1=20, x2=110, y2=220,
                             confidence=0.9, class_id=0, class_name="person")
    """
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
