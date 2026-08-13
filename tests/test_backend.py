"""Unit tests for backend switching, multi-backend config, and device fallback logic."""

import platform
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from config import (
    CameraConfig,
    ModelConfig,
    _detect_api_backend,
)
from detector import OpenVINODetector


class TestCrossPlatform:
    """Cross-platform backend detection tests / 跨平台后端检测测试"""

    @pytest.mark.parametrize(
        "system, expected",
        [
            ("Windows", 700),    # CAP_MSMF (Windows Media Foundation)
            ("Linux", 200),      # CAP_V4L2 (Video4Linux2)
            ("Darwin", 1200),    # CAP_AVFOUNDATION (macOS)
            ("FreeBSD", 0),      # unknown platform -> CAP_ANY (auto)
            ("", 0),             # empty string -> CAP_ANY
        ],
    )
    def test_detect_api_backend(self, system, expected):
        """Test _detect_api_backend() returns the correct constant per platform."""
        with patch("platform.system", return_value=system):
            assert _detect_api_backend() == expected

    @pytest.mark.parametrize(
        "system, expected",
        [
            ("Windows", 700),
            ("Linux", 200),
            ("Darwin", 1200),
            ("UnknownOS", 0),
        ],
    )
    def test_camera_config_post_init_auto_detect(self, system, expected):
        """CameraConfig.__post_init__ auto-sets api_preference per platform."""
        with patch("platform.system", return_value=system):
            cam = CameraConfig()
            assert cam.api_preference == expected

    @pytest.mark.parametrize("manual_value", [700, 200, 1200, 1])
    def test_camera_config_manual_preference_not_overridden(self, manual_value):
        """A manually set api_preference (non-zero) is not overridden."""
        with patch("platform.system", return_value="Linux"):
            cam = CameraConfig(api_preference=manual_value)
            assert cam.api_preference == manual_value

    def test_camera_config_zero_triggers_auto_detect(self):
        """An explicit api_preference=0 still triggers auto-detection."""
        with patch("platform.system", return_value="Windows"):
            cam = CameraConfig(api_preference=0)
            assert cam.api_preference == 700

    def test_patch_isolation(self):
        """Patch scope is isolated; platform.system restores after the with block."""
        original = platform.system()
        with patch("platform.system", return_value="Windows"):
            assert _detect_api_backend() == 700
        # Restored to the real platform after the with block / 退出 with 后恢复真实平台
        assert platform.system() == original


class TestBackendConfig:
    """Multi-backend model config tests / 多后端模型配置测试"""

    def test_default_backend_is_openvino(self):
        """ModelConfig.backend defaults to "openvino" / 测试默认 backend"""
        m = ModelConfig()
        assert m.backend == "openvino"

    @pytest.mark.parametrize(
        "backend, expected_path",
        [
            ("openvino", Path("models/exported/yolo26s_int8_openvino_model")),
            ("cuda", Path("models/yolo26s.pt")),
            ("tensorrt", Path("models/exported/yolo26s_int8_engine")),
        ],
    )
    def test_exported_path_by_backend(self, backend, expected_path):
        """Test exported_path format per backend / 测试各后端下路径格式"""
        m = ModelConfig(backend=backend, int8=True, half=False)
        assert m.exported_path == expected_path

    @pytest.mark.parametrize(
        "backend, expected",
        [
            ("openvino", True),
            ("tensorrt", True),
            ("cuda", False),
        ],
    )
    def test_needs_export(self, backend, expected):
        """Test needs_export per backend / 测试各后端是否需要导出"""
        m = ModelConfig(backend=backend)
        assert m.needs_export is expected

    @pytest.mark.parametrize(
        "int8, half, precision",
        [
            (True, False, "int8"),
            (False, True, "fp16"),
            (False, False, "fp32"),
        ],
    )
    def test_exported_path_precision_combinations(self, int8, half, precision):
        """Test OpenVINO path generation per precision combination."""
        m = ModelConfig(backend="openvino", int8=int8, half=half)
        expected = Path("models/exported") / f"yolo26s_{precision}_openvino_model"
        assert m.exported_path == expected

    @pytest.mark.parametrize(
        "int8, half, precision",
        [
            (True, False, "int8"),
            (False, True, "fp16"),
            (False, False, "fp32"),
        ],
    )
    def test_tensorrt_precision_paths(self, int8, half, precision):
        """Test TensorRT path generation per precision / 测试 TensorRT 各精度路径"""
        m = ModelConfig(backend="tensorrt", int8=int8, half=half)
        expected = Path("models/exported") / f"yolo26s_{precision}_engine"
        assert m.exported_path == expected

    def test_int8_takes_priority_over_half(self):
        """When int8=True and half=True, precision is int8 (highest priority)."""
        m = ModelConfig(backend="openvino", int8=True, half=True)
        expected = Path("models/exported/yolo26s_int8_openvino_model")
        assert m.exported_path == expected

    def test_cuda_exported_path_equals_model_path(self):
        """CUDA exported_path equals model_path (uses the raw model directly)."""
        m = ModelConfig(backend="cuda")
        assert m.exported_path == m.model_path
        assert m.exported_path == Path("models/yolo26s.pt")


class TestDeviceFallback:
    """Device fallback logic tests / 设备回退逻辑测试"""

    @pytest.mark.parametrize(
        "backend, expected_device",
        [
            ("openvino", "intel:cpu"),      # OpenVINO falls back to Intel CPU
            ("cuda", "cpu"),                # CUDA falls back to CPU
            ("tensorrt", "cpu"),            # TensorRT falls back to CPU
            ("unknown_backend", "cpu"),     # unknown backend falls back to CPU
            ("", "cpu"),                    # empty string falls back to CPU
        ],
    )
    def test_get_fallback_device(self, backend, expected_device):
        """Test _get_fallback_device() per backend / 测试各后端的回退设备"""
        # Build a mock MODEL with only the backend attribute needed
        mock_model = MagicMock()
        mock_model.backend = backend

        with patch("detector.MODEL", mock_model):
            detector = OpenVINODetector()
            assert detector._get_fallback_device() == expected_device

    def test_fallback_does_not_modify_global_model(self):
        """The mock does not pollute the global MODEL singleton."""
        from config import MODEL as global_model
        original_backend = global_model.backend

        mock_model = MagicMock()
        mock_model.backend = "cuda"
        with patch("detector.MODEL", mock_model):
            detector = OpenVINODetector()
            # During patch, detector sees the mock; fallback should be "cpu"
            assert detector._get_fallback_device() == "cpu"

        # After patch exits, global MODEL keeps its original value
        assert global_model.backend == original_backend


class TestBackendDependencyCheck:
    """Backend dependency check tests / 后端依赖检查测试"""

    def test_openvino_backend_success(self):
        """OpenVINO backend, openvino installed -> True"""
        mock_model = MagicMock()
        mock_model.backend = "openvino"
        with patch("detector.MODEL", mock_model):
            with patch("builtins.__import__") as mock_import:
                def custom_import(name, *args, **kwargs):
                    if name == "openvino":
                        return MagicMock()
                    return __builtins__.__import__(name, *args, **kwargs) if hasattr(__builtins__, '__import__') else MagicMock()
                mock_import.side_effect = custom_import
                detector = OpenVINODetector()
                assert detector._check_backend_dependencies() is True

    def test_cuda_backend_no_torch(self):
        """CUDA backend, PyTorch not installed -> False"""
        mock_model = MagicMock()
        mock_model.backend = "cuda"
        with patch("detector.MODEL", mock_model):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'torch'")):
                detector = OpenVINODetector()
                assert detector._check_backend_dependencies() is False

    def test_cuda_backend_torch_no_cuda(self):
        """CUDA backend, PyTorch installed but CUDA unavailable -> False"""
        mock_model = MagicMock()
        mock_model.backend = "cuda"

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch("detector.MODEL", mock_model):
            with patch.dict("sys.modules", {"torch": mock_torch}):
                detector = OpenVINODetector()
                assert detector._check_backend_dependencies() is False

    def test_cuda_backend_success(self):
        """CUDA backend, PyTorch+CUDA available -> True"""
        mock_model = MagicMock()
        mock_model.backend = "cuda"

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch("detector.MODEL", mock_model):
            with patch.dict("sys.modules", {"torch": mock_torch}):
                detector = OpenVINODetector()
                assert detector._check_backend_dependencies() is True

    def test_tensorrt_backend_no_tensorrt(self):
        """TensorRT backend, torch+CUDA available but TensorRT missing -> False"""
        mock_model = MagicMock()
        mock_model.backend = "tensorrt"

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "tensorrt":
                raise ImportError("No module named 'tensorrt'")
            return original_import(name, *args, **kwargs)

        with patch("detector.MODEL", mock_model):
            with patch.dict("sys.modules", {"torch": mock_torch}):
                with patch("builtins.__import__", side_effect=mock_import):
                    detector = OpenVINODetector()
                    assert detector._check_backend_dependencies() is False
