"""
config.py 模块单元测试
=======================
覆盖: CameraConfig / ModelConfig / DisplayConfig / COCO_CLASSES

注意: config.py 仅依赖标准库 (platform / dataclasses / pathlib),
不依赖 cv2 / openvino, 因此可在任意环境运行。
"""

import platform
from pathlib import Path

import pytest

from config import (
    CameraConfig,
    ModelConfig,
    DisplayConfig,
    COCO_CLASSES,
    _detect_api_backend,
)


# ============================================================
# CameraConfig
# ============================================================
class TestCameraConfig:
    """摄像头配置测试"""

    def test_defaults(self):
        """测试 CameraConfig 默认值"""
        cam = CameraConfig()
        assert cam.device_index == 0
        assert cam.width == 1920
        assert cam.height == 1080
        assert cam.fps == 30
        assert cam.buffer_size == 1

    def test_post_init_auto_detect_backend(self):
        """
        测试 __post_init__ 自动检测平台后端 (api_preference 不为 0)

        当 api_preference 保持默认 0 时, __post_init__ 会调用
        _detect_api_backend() 覆盖为当前平台对应的后端常量。
        """
        cam = CameraConfig()
        # 在已知平台 (Windows/Linux/macOS) 上应被覆盖为非 0
        assert cam.api_preference == _detect_api_backend()
        assert cam.api_preference != 0

    @pytest.mark.parametrize(
        "system, expected",
        [
            ("Windows", 700),    # CAP_MSMF
            ("Linux", 200),      # CAP_V4L2
            ("Darwin", 1200),    # CAP_AVFOUNDATION
            ("UnknownOS", 0),    # CAP_ANY
        ],
    )
    def test_detect_api_backend_mapping(self, monkeypatch, system, expected):
        """测试 _detect_api_backend 在各平台返回的常量值"""
        monkeypatch.setattr(platform, "system", lambda: system)
        assert _detect_api_backend() == expected

    def test_manual_api_preference_not_overridden(self):
        """
        当用户手动指定 api_preference (非 0) 时, __post_init__ 不应覆盖
        """
        cam = CameraConfig(api_preference=700)
        assert cam.api_preference == 700


# ============================================================
# ModelConfig
# ============================================================
class TestModelConfig:
    """模型配置测试"""

    def test_defaults(self):
        """测试 ModelConfig 默认值"""
        m = ModelConfig()
        assert m.model_name == "yolo26s.pt"
        assert m.backend == "openvino"
        assert m.export_format == "openvino"
        assert m.imgsz == 640
        assert m.int8 is True
        assert m.half is False
        assert m.nms is True
        assert m.calib_data == "coco128.yaml"
        assert m.calib_fraction == 0.1
        assert m.inference_device == "intel:gpu"
        assert m.conf_threshold == 0.5
        assert m.iou_threshold == 0.5
        assert m.models_dir == Path("models")
        assert m.exported_dir == Path("models/exported")

    @pytest.mark.parametrize(
        "int8, half, precision",
        [
            (True, False, "int8"),
            (False, True, "fp16"),
            (False, False, "fp32"),
        ],
    )
    def test_exported_path_by_precision(self, int8, half, precision):
        """
        测试 exported_path 根据精度 (int8/fp16/fp32) 生成路径
        """
        m = ModelConfig(backend="openvino", int8=int8, half=half)
        expected = Path("models/exported") / f"yolo26s_{precision}_openvino_model"
        assert m.exported_path == expected

    @pytest.mark.parametrize(
        "backend, suffix",
        [
            ("openvino", "yolo26s_int8_openvino_model"),
            ("tensorrt", "yolo26s_int8_engine"),
        ],
    )
    def test_exported_path_by_backend(self, backend, suffix):
        """
        测试 exported_path 根据 backend (openvino/tensorrt) 生成不同路径
        """
        m = ModelConfig(backend=backend, int8=True, half=False)
        assert m.exported_path == Path("models/exported") / suffix

    def test_exported_path_cuda_returns_model_path(self):
        """
        CUDA 后端无需导出, exported_path 直接返回原始 PyTorch 模型路径
        """
        m = ModelConfig(backend="cuda")
        assert m.exported_path == m.model_path

    def test_needs_export(self):
        """测试 needs_export 属性: openvino/tensorrt 需导出, cuda 不需要"""
        assert ModelConfig(backend="openvino").needs_export is True
        assert ModelConfig(backend="tensorrt").needs_export is True
        assert ModelConfig(backend="cuda").needs_export is False

    def test_model_path(self):
        """测试 model_path 属性: models_dir / model_name"""
        m = ModelConfig(model_name="yolo26m.pt")
        assert m.model_path == Path("models") / "yolo26m.pt"


# ============================================================
# DisplayConfig
# ============================================================
class TestDisplayConfig:
    """显示配置测试"""

    def test_defaults(self):
        """测试 DisplayConfig 默认值"""
        d = DisplayConfig()
        assert d.show_window is True
        assert d.window_name == "DJI Vision System"
        assert d.show_fps is True
        assert d.show_confidence is True
        assert d.show_class_name is True
        assert d.box_thickness == 2
        assert d.font_scale == 0.6
        assert d.mirror is False
        assert d.save_output is False
        assert d.output_path == Path("output/result.mp4")
        assert d.output_fps == 30
        assert d.log_detections is False
        assert d.log_path == Path("output/detections.csv")


# ============================================================
# COCO_CLASSES
# ============================================================
class TestCocoClasses:
    """COCO 类别表测试"""

    def test_has_80_classes(self):
        """测试 COCO_CLASSES 有 80 个类别"""
        assert len(COCO_CLASSES) == 80

    def test_known_indices(self):
        """校验若干已知类别的索引位置"""
        assert COCO_CLASSES[0] == "person"
        assert COCO_CLASSES[2] == "car"
        assert COCO_CLASSES[7] == "truck"

    def test_no_duplicates(self):
        """类别名称不应重复"""
        assert len(COCO_CLASSES) == len(set(COCO_CLASSES))
