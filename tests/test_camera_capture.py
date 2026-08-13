"""
camera_capture.py 模块单元测试
================================
覆盖:
  - CameraCapture 初始化参数的 "is not None" 判断 (BUG-08 回归测试)
  - source 参数 (视频文件模式) 的初始化行为
  - 上下文管理器接口 (__enter__ / __exit__)

注意:
  - 不打开真实摄像头, 不读取真实视频文件
  - 涉及 open/close 的用例使用 mock 替换, 避免硬件依赖
"""

from unittest.mock import patch

import pytest

from config import CAMERA
from camera_capture import CameraCapture


# ============================================================
# BUG-08 回归测试: 零值参数不应被覆盖
# ============================================================
class TestCameraCaptureInitZeroValues:
    """
    BUG-08 回归测试
    -----------------
    历史问题: 初始化时使用 `if width:` 真值判断, 导致传入 0 等合法
    "零值" 参数会被误判为未设置, 从而被 config 默认值 (1920/1080/30) 覆盖。

    修复: 改用 `is not None` 判断。此处验证 0 值参数被正确保留。
    """

    def test_width_zero_not_overridden(self):
        """width=0 时不会被覆盖为默认值 1920 (BUG-08)"""
        cam = CameraCapture(width=0)
        assert cam.width == 0
        assert cam.width != CAMERA.width

    def test_height_zero_not_overridden(self):
        """height=0 时不会被覆盖为默认值 1080 (BUG-08)"""
        cam = CameraCapture(height=0)
        assert cam.height == 0
        assert cam.height != CAMERA.height

    def test_fps_zero_not_overridden(self):
        """fps=0 时不会被覆盖为默认值 30 (BUG-08)"""
        cam = CameraCapture(fps=0)
        assert cam.fps == 0
        assert cam.fps != CAMERA.fps

    def test_all_zero_not_overridden(self):
        """width/height/fps 同时为 0 时均被保留 (BUG-08)"""
        cam = CameraCapture(width=0, height=0, fps=0)
        assert (cam.width, cam.height, cam.fps) == (0, 0, 0)

    def test_none_falls_back_to_config(self):
        """显式传 None 时回退到 config 默认值"""
        cam = CameraCapture(width=None, height=None, fps=None)
        assert cam.width == CAMERA.width
        assert cam.height == CAMERA.height
        assert cam.fps == CAMERA.fps

    def test_device_index_zero_not_overridden(self):
        """device_index=0 时被保留 (0 是合法的默认设备索引)"""
        cam = CameraCapture(device_index=0)
        assert cam.device_index == 0


# ============================================================
# source 参数 (视频文件模式)
# ============================================================
class TestCameraCaptureSource:
    """source 参数 / 视频文件模式测试"""

    def test_source_sets_file_mode(self):
        """指定 source 后进入视频文件模式 (_is_file=True)"""
        cam = CameraCapture(source="video.mp4")
        assert cam.source == "video.mp4"
        assert cam._is_file is True

    def test_source_file_mode_resets_capture_params(self):
        """
        视频文件模式下, 摄像头采集参数被置零:
        device_index=-1, width/height/fps=0 (实际值在 open 时从文件读取)
        """
        cam = CameraCapture(source="test.mp4")
        assert cam.device_index == -1
        assert cam.width == 0
        assert cam.height == 0
        assert cam.fps == 0

    def test_source_ignores_device_index(self):
        """视频文件模式下, 即使传入 device_index 也会被忽略"""
        cam = CameraCapture(source="video.mp4", device_index=5)
        assert cam.device_index == -1

    def test_no_source_is_camera_mode(self):
        """未指定 source 时为摄像头模式 (_is_file=False)"""
        cam = CameraCapture()
        assert cam.source is None
        assert cam._is_file is False
        assert cam.device_index == CAMERA.device_index

    def test_no_source_uses_custom_params(self):
        """摄像头模式下, 自定义参数被保留"""
        cam = CameraCapture(device_index=1, width=1280, height=720, fps=60)
        assert cam.device_index == 1
        assert cam.width == 1280
        assert cam.height == 720
        assert cam.fps == 60

    def test_initial_state_not_running(self):
        """构造后未运行, cap 为 None"""
        cam = CameraCapture()
        assert cam.is_running is False
        assert cam.cap is None


# ============================================================
# 上下文管理器接口
# ============================================================
class TestCameraCaptureContextManager:
    """上下文管理器 (with 语句) 接口测试"""

    def test_context_manager_calls_open_and_close(self):
        """
        with 语句进入时调用 open(), 退出时调用 close()
        使用 mock 替换 open/close, 避免触碰真实摄像头 / cv2
        """
        cam = CameraCapture()
        with patch.object(cam, "open", return_value=True) as mock_open, \
                patch.object(cam, "close") as mock_close:
            with cam as c:
                # __enter__ 应返回采集器自身
                assert c is cam
                mock_open.assert_called_once()
            # 退出时应调用 close
            mock_close.assert_called_once()

    def test_context_manager_open_failure_propagates(self):
        """open() 返回 False 时, with 仍正常退出并调用 close"""
        cam = CameraCapture()
        with patch.object(cam, "open", return_value=False), \
                patch.object(cam, "close") as mock_close:
            with cam as c:
                assert c is cam
            mock_close.assert_called_once()

    def test_context_manager_exception_still_closes(self):
        """with 块内抛异常时, close() 仍被调用 (异常正常传播)"""
        cam = CameraCapture()
        with patch.object(cam, "open", return_value=True), \
                patch.object(cam, "close") as mock_close:
            with pytest.raises(RuntimeError, match="boom"):
                with cam:
                    raise RuntimeError("boom")
            mock_close.assert_called_once()
