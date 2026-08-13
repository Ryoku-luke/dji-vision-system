"""Unit tests for detector.py — Detection dataclass, DetectionResult, and detector init."""

from pathlib import Path

import pytest

from config import MODEL
from detector import Detection, DetectionResult, OpenVINODetector


class TestDetection:
    """Single detection dataclass tests / 单个检测结果数据类测试"""

    def test_attributes(self, make_detection):
        """Test Detection attributes (bbox / width / height / center)"""
        det = make_detection(x1=10, y1=20, x2=110, y2=220,
                             confidence=0.9, class_id=0, class_name="person")

        # Basic fields / 基础字段
        assert det.x1 == 10
        assert det.y1 == 20
        assert det.x2 == 110
        assert det.y2 == 220
        assert det.confidence == 0.9
        assert det.class_id == 0
        assert det.class_name == "person"

        # Computed properties / 计算属性
        assert det.bbox == (10, 20, 110, 220)
        assert det.width == 100          # x2 - x1
        assert det.height == 200         # y2 - y1
        assert det.center == (60.0, 120.0)  # ((x1+x2)/2, (y1+y2)/2)

    @pytest.mark.parametrize(
        "x1, y1, x2, y2, exp_w, exp_h, exp_center",
        [
            (0, 0, 50, 50, 50, 50, (25.0, 25.0)),
            (100, 200, 300, 400, 200, 200, (200.0, 300.0)),
            (0, 0, 0, 0, 0, 0, (0.0, 0.0)),
        ],
    )
    def test_geometry(self, make_detection, x1, y1, x2, y2,
                      exp_w, exp_h, exp_center):
        """Parametrized bbox geometry checks / 参数化验证边界框几何计算"""
        det = make_detection(x1=x1, y1=y1, x2=x2, y2=y2)
        assert det.width == exp_w
        assert det.height == exp_h
        assert det.center == exp_center


class TestDetectionResult:
    """Per-frame detection result tests / 单帧检测结果测试"""

    def _make_result(self, detections, inference_time_ms=10.0):
        """Helper to build a DetectionResult / 构造 DetectionResult 辅助方法"""
        return DetectionResult(
            detections=detections,
            inference_time_ms=inference_time_ms,
            frame_shape=(1080, 1920),
        )

    def test_num_objects(self, make_detection):
        """Test num_objects property / 测试 num_objects 属性"""
        dets = [make_detection(class_id=i) for i in range(3)]
        result = self._make_result(dets)
        assert result.num_objects == 3

        # Empty list / 空列表
        empty = self._make_result([])
        assert empty.num_objects == 0

    def test_fps_normal(self, make_detection):
        """Test fps: inference_time_ms=10 -> fps=100"""
        result = self._make_result([make_detection()], inference_time_ms=10.0)
        assert result.fps == pytest.approx(100.0)

    @pytest.mark.parametrize("time_ms", [0.0, -1.0])
    def test_fps_zero_when_no_time(self, make_detection, time_ms):
        """fps is 0 when inference_time_ms is 0 or negative."""
        result = self._make_result([make_detection()], inference_time_ms=time_ms)
        assert result.fps == 0.0

    def test_filter_by_class(self, make_detection):
        """Test filter_by_class method / 测试 filter_by_class 方法"""
        dets = [
            make_detection(class_id=0, class_name="person"),
            make_detection(class_id=2, class_name="car"),
            make_detection(class_id=7, class_name="truck"),
            make_detection(class_id=0, class_name="person"),
        ]
        result = self._make_result(dets)

        # Keep only person (class_id=0) / 仅保留 person
        persons = result.filter_by_class([0])
        assert len(persons) == 2
        assert all(d.class_id == 0 for d in persons)

        # Keep person + car / 保留 person + car
        filtered = result.filter_by_class([0, 2])
        assert len(filtered) == 3

        # Non-existent class -> empty list / 不存在的类别 -> 空列表
        assert result.filter_by_class([99]) == []

    def test_filter_by_confidence(self, make_detection):
        """Test filter_by_confidence method / 测试 filter_by_confidence 方法"""
        dets = [
            make_detection(confidence=0.3),
            make_detection(confidence=0.6),
            make_detection(confidence=0.9),
        ]
        result = self._make_result(dets)

        # Threshold 0.5: keep 0.6 and 0.9 / 阈值 0.5: 保留 0.6 与 0.9
        filtered = result.filter_by_confidence(0.5)
        assert len(filtered) == 2
        assert all(d.confidence >= 0.5 for d in filtered)

        # Threshold 0.0: keep all (boundary, >= 0) / 阈值 0.0: 全部保留
        assert len(result.filter_by_confidence(0.0)) == 3

        # Threshold 1.0: filter all / 阈值 1.0: 全部过滤
        assert len(result.filter_by_confidence(1.0)) == 0


class TestOpenVINODetectorInit:
    """Regression tests for detector init parameter handling / 初始化参数处理回归测试"""

    def test_defaults_from_config(self):
        """With no args, use config.MODEL defaults / 未传参时使用 config.MODEL 默认值"""
        det = OpenVINODetector()
        assert det.model_path == MODEL.exported_path
        assert det.device == MODEL.inference_device
        assert det.conf_threshold == MODEL.conf_threshold
        assert det.iou_threshold == MODEL.iou_threshold
        assert det.classes is None
        assert det.is_loaded is False
        assert det.model is None

    def test_conf_threshold_zero_not_overridden(self):
        """conf_threshold=0.0 is kept, not overridden by the default."""
        det = OpenVINODetector(conf_threshold=0.0)
        assert det.conf_threshold == 0.0
        assert det.conf_threshold != MODEL.conf_threshold

    def test_iou_threshold_zero_not_overridden(self):
        """iou_threshold=0.0 is kept, not overridden by the default."""
        det = OpenVINODetector(iou_threshold=0.0)
        assert det.iou_threshold == 0.0
        assert det.iou_threshold != MODEL.iou_threshold

    def test_device_empty_string_not_overridden(self):
        """device="" is kept, not overridden by the default."""
        det = OpenVINODetector(device="")
        assert det.device == ""
        assert det.device != MODEL.inference_device

    def test_explicit_model_path_kept(self):
        """An explicitly given model_path is kept / 显式指定的 model_path 被保留"""
        custom = Path("/tmp/custom_model")
        det = OpenVINODetector(model_path=custom)
        assert det.model_path == custom

    def test_explicit_classes_kept(self):
        """An explicitly given classes list is kept / 显式指定的 classes 列表被保留"""
        classes = [0, 2, 7]
        det = OpenVINODetector(classes=classes)
        assert det.classes == classes

    def test_none_values_fall_back_to_config(self):
        """Explicit None falls back to config defaults / 显式传 None 时回退到 config 默认值"""
        det = OpenVINODetector(
            model_path=None,
            device=None,
            conf_threshold=None,
            iou_threshold=None,
            classes=None,
        )
        assert det.model_path == MODEL.exported_path
        assert det.device == MODEL.inference_device
        assert det.conf_threshold == MODEL.conf_threshold
        assert det.iou_threshold == MODEL.iou_threshold
        assert det.classes is None
