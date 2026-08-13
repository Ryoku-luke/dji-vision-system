"""
跨平台后端切换 & 多后端推理逻辑测试 (H-09)
=============================================
覆盖:
  1. 跨平台后端检测 (_detect_api_backend / CameraConfig.__post_init__)
  2. 多后端模型配置 (ModelConfig.backend / exported_path / needs_export / 精度组合)
  3. 设备回退逻辑 (OpenVINODetector._get_fallback_device)

注意:
  - 不依赖真实 GPU / 摄像头 / OpenVINO / ultralytics 安装
  - 使用 unittest.mock.patch 模拟平台与后端环境
  - 使用 pytest.mark.parametrize 参数化测试
  - 与现有 65 个测试不冲突 (独立新增, 不修改已有测试)

导入安全说明:
  detector.py 在模块顶层仅导入 numpy 和 config, 不导入 ultralytics
  (ultralytics 在 load() 方法内才延迟导入), 因此可直接安全导入。
  本文件中所有涉及 detector 的测试均不调用 load(), 不会触发 ultralytics 依赖。
"""

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


# ============================================================
# 1. 跨平台后端检测测试
# ============================================================
class TestCrossPlatform:
    """
    跨平台后端检测测试 (H-01 / H-09)

    验证 _detect_api_backend() 和 CameraConfig.__post_init__ 在不同
    操作系统平台下返回 / 设置正确的 OpenCV VideoCapture 后端常量:
      Windows -> 700  (CAP_MSMF)
      Linux   -> 200  (CAP_V4L2)
      Darwin  -> 1200 (CAP_AVFOUNDATION)
      其他     -> 0   (CAP_ANY)
    """

    @pytest.mark.parametrize(
        "system, expected",
        [
            ("Windows", 700),    # CAP_MSMF (Windows Media Foundation)
            ("Linux", 200),      # CAP_V4L2 (Video4Linux2)
            ("Darwin", 1200),    # CAP_AVFOUNDATION (macOS)
            ("FreeBSD", 0),      # 未知平台 -> CAP_ANY (自动选择)
            ("", 0),             # 空字符串 -> CAP_ANY
        ],
    )
    def test_detect_api_backend(self, system, expected):
        """
        测试 _detect_api_backend() 在不同平台返回正确的后端常量。
        使用 unittest.mock.patch 模拟 platform.system() 返回值。
        """
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
        """
        测试 CameraConfig.__post_init__ 在不同平台下自动设置 api_preference。
        当 api_preference 保持默认 0 时, __post_init__ 调用
        _detect_api_backend() 将其覆盖为当前平台对应的后端常量。
        """
        with patch("platform.system", return_value=system):
            cam = CameraConfig()
            assert cam.api_preference == expected

    @pytest.mark.parametrize("manual_value", [700, 200, 1200, 1])
    def test_camera_config_manual_preference_not_overridden(self, manual_value):
        """
        测试 CameraConfig 手动指定 api_preference (非 0) 时不被自动覆盖。
        __post_init__ 仅在 api_preference == 0 时才调用 _detect_api_backend(),
        非 0 值应原样保留。
        """
        with patch("platform.system", return_value="Linux"):
            cam = CameraConfig(api_preference=manual_value)
            assert cam.api_preference == manual_value

    def test_camera_config_zero_triggers_auto_detect(self):
        """
        测试 api_preference 显式传 0 时仍触发自动检测。
        0 是 "自动检测" 的信号值, 即使显式传入 0 也会被
        _detect_api_backend() 的返回值覆盖。
        """
        with patch("platform.system", return_value="Windows"):
            cam = CameraConfig(api_preference=0)
            assert cam.api_preference == 700

    def test_patch_isolation(self):
        """
        测试 patch 作用域隔离: 退出 with 块后 platform.system 恢复原值。
        确保测试不会污染全局状态, 不影响其他测试。
        """
        original = platform.system()
        with patch("platform.system", return_value="Windows"):
            assert _detect_api_backend() == 700
        # 退出 with 块后应恢复为真实平台
        assert platform.system() == original


# ============================================================
# 2. 多后端配置测试
# ============================================================
class TestBackendConfig:
    """
    多后端模型配置测试 (H-02 / H-09)

    验证 ModelConfig 在不同推理后端 (openvino / cuda / tensorrt) 下:
      - 默认 backend 为 "openvino"
      - exported_path 根据后端和精度生成正确路径
      - needs_export 属性正确判断是否需要导出步骤
    """

    def test_default_backend_is_openvino(self):
        """测试 ModelConfig 的 backend 字段默认为 "openvino" """
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
        """
        测试 ModelConfig.exported_path 在不同 backend 下的路径格式:
          - openvino: models/exported/yolo26s_int8_openvino_model
          - cuda:     models/yolo26s.pt (直接使用原始模型, 无需导出)
          - tensorrt: models/exported/yolo26s_int8_engine
        """
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
        """
        测试 ModelConfig.needs_export 属性:
          - openvino -> True  (需要导出为 OpenVINO IR)
          - tensorrt -> True  (需要导出为 TensorRT 引擎)
          - cuda     -> False (直接使用 PyTorch 模型, 无需导出)
        """
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
        """
        测试不同精度组合 (int8/fp16/fp32) 下的 OpenVINO 路径生成。
        int8 优先级最高, 其次 half (fp16), 最后 fp32。
        """
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
        """测试 TensorRT 后端在不同精度下的路径生成"""
        m = ModelConfig(backend="tensorrt", int8=int8, half=half)
        expected = Path("models/exported") / f"yolo26s_{precision}_engine"
        assert m.exported_path == expected

    def test_int8_takes_priority_over_half(self):
        """
        测试 int8=True 且 half=True 时, 精度取 int8 (int8 优先级最高)。
        两者同时为 True 时不应生成 fp16 路径。
        """
        m = ModelConfig(backend="openvino", int8=True, half=True)
        expected = Path("models/exported/yolo26s_int8_openvino_model")
        assert m.exported_path == expected

    def test_cuda_exported_path_equals_model_path(self):
        """
        测试 CUDA 后端的 exported_path 等于 model_path (直接使用原始模型)。
        CUDA 无需导出, 路径直接指向 .pt 文件。
        """
        m = ModelConfig(backend="cuda")
        assert m.exported_path == m.model_path
        assert m.exported_path == Path("models/yolo26s.pt")


# ============================================================
# 3. 设备回退逻辑测试
# ============================================================
class TestDeviceFallback:
    """
    设备回退逻辑测试 (OpenVINODetector._get_fallback_device)

    验证推理设备验证失败时的回退逻辑:
      - openvino  -> "intel:cpu"
      - cuda      -> "cpu"
      - tensorrt  -> "cpu"
      - 未知后端   -> "cpu"

    实现方式:
      - detector.py 在模块顶层仅导入 numpy 和 config, 不导入 ultralytics
        (ultralytics 在 load() 方法内才延迟导入), 因此可直接安全导入。
      - _get_fallback_device() 读取全局 MODEL.backend 决定回退设备,
        使用 unittest.mock.patch 替换 detector.MODEL 来模拟不同后端,
        避免依赖真实 OpenVINO / ultralytics 安装。
    """

    @pytest.mark.parametrize(
        "backend, expected_device",
        [
            ("openvino", "intel:cpu"),      # OpenVINO 回退到 Intel CPU
            ("cuda", "cpu"),                # CUDA 回退到 CPU
            ("tensorrt", "cpu"),            # TensorRT 回退到 CPU
            ("unknown_backend", "cpu"),     # 未知后端回退到 CPU
            ("", "cpu"),                    # 空字符串回退到 CPU
        ],
    )
    def test_get_fallback_device(self, backend, expected_device):
        """
        测试 OpenVINODetector._get_fallback_device() 在不同后端下的回退设备。

        使用 MagicMock 替换 detector.MODEL, 设置不同的 backend 属性,
        避免依赖真实 OpenVINO / ultralytics 安装。
        """
        # 构造 mock MODEL, 仅需 backend 属性供 _get_fallback_device 判断
        # (构造函数访问的其他属性由 MagicMock 自动生成, 不影响测试)
        mock_model = MagicMock()
        mock_model.backend = backend

        with patch("detector.MODEL", mock_model):
            detector = OpenVINODetector()
            assert detector._get_fallback_device() == expected_device

    def test_fallback_does_not_modify_global_model(self):
        """
        测试 mock 不会污染全局 MODEL 单例。
        退出 patch 上下文后, 全局 config.MODEL.backend 应保持原值。
        """
        from config import MODEL as global_model
        original_backend = global_model.backend

        mock_model = MagicMock()
        mock_model.backend = "cuda"
        with patch("detector.MODEL", mock_model):
            detector = OpenVINODetector()
            # patch 期间 detector 看到的是 mock, 回退设备应为 "cpu"
            assert detector._get_fallback_device() == "cpu"

        # 退出 patch 后, 全局 MODEL 应保持原值 (默认 "openvino")
        assert global_model.backend == original_backend


# ============================================================
# 4. 后端依赖检查测试 (L-14)
# ============================================================
class TestBackendDependencyCheck:
    """
    后端依赖检查测试 (OpenVINODetector._check_backend_dependencies)

    验证:
      - openvino 后端: openvino 可导入 -> True, 不可导入 -> False
      - cuda 后端: torch.cuda 可用 -> True, 不可用 -> False, torch 未安装 -> False
      - tensorrt 后端: torch+cuda+tensorrt 均可用 -> True, tensorrt 缺失 -> False
    """

    def test_openvino_backend_success(self):
        """OpenVINO 后端, openvino 已安装 -> True"""
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
        """CUDA 后端, PyTorch 未安装 -> False"""
        mock_model = MagicMock()
        mock_model.backend = "cuda"
        with patch("detector.MODEL", mock_model):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'torch'")):
                detector = OpenVINODetector()
                assert detector._check_backend_dependencies() is False

    def test_cuda_backend_torch_no_cuda(self):
        """CUDA 后端, PyTorch 已安装但 CUDA 不可用 -> False"""
        mock_model = MagicMock()
        mock_model.backend = "cuda"

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch("detector.MODEL", mock_model):
            with patch.dict("sys.modules", {"torch": mock_torch}):
                detector = OpenVINODetector()
                assert detector._check_backend_dependencies() is False

    def test_cuda_backend_success(self):
        """CUDA 后端, PyTorch+CUDA 可用 -> True"""
        mock_model = MagicMock()
        mock_model.backend = "cuda"

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True

        with patch("detector.MODEL", mock_model):
            with patch.dict("sys.modules", {"torch": mock_torch}):
                detector = OpenVINODetector()
                assert detector._check_backend_dependencies() is True

    def test_tensorrt_backend_no_tensorrt(self):
        """TensorRT 后端, torch+CUDA 可用但 TensorRT 未安装 -> False"""
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
