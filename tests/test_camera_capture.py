"""Unit tests for camera_capture.py — init params, source mode, and context manager."""

from unittest.mock import patch

import pytest

from config import CAMERA
from camera_capture import CameraCapture


class TestCameraCaptureInitZeroValues:
    """Regression tests: zero-value params should not be overridden / 零值参数不应被覆盖回归测试"""

    def test_width_zero_not_overridden(self):
        """width=0 is kept, not overridden by the default 1920."""
        cam = CameraCapture(width=0)
        assert cam.width == 0
        assert cam.width != CAMERA.width

    def test_height_zero_not_overridden(self):
        """height=0 is kept, not overridden by the default 1080."""
        cam = CameraCapture(height=0)
        assert cam.height == 0
        assert cam.height != CAMERA.height

    def test_fps_zero_not_overridden(self):
        """fps=0 is kept, not overridden by the default 30."""
        cam = CameraCapture(fps=0)
        assert cam.fps == 0
        assert cam.fps != CAMERA.fps

    def test_all_zero_not_overridden(self):
        """width/height/fps all zero are kept / 同时为 0 时均被保留"""
        cam = CameraCapture(width=0, height=0, fps=0)
        assert (cam.width, cam.height, cam.fps) == (0, 0, 0)

    def test_none_falls_back_to_config(self):
        """Explicit None falls back to config defaults / 显式传 None 时回退到 config 默认值"""
        cam = CameraCapture(width=None, height=None, fps=None)
        assert cam.width == CAMERA.width
        assert cam.height == CAMERA.height
        assert cam.fps == CAMERA.fps

    def test_device_index_zero_not_overridden(self):
        """device_index=0 is kept (0 is a valid default device index)."""
        cam = CameraCapture(device_index=0)
        assert cam.device_index == 0


class TestCameraCaptureSource:
    """Source param / video file mode tests / source 参数 / 视频文件模式测试"""

    def test_source_sets_file_mode(self):
        """Specifying source enters video file mode (_is_file=True)."""
        cam = CameraCapture(source="video.mp4")
        assert cam.source == "video.mp4"
        assert cam._is_file is True

    def test_source_file_mode_resets_capture_params(self):
        """In file mode, capture params are zeroed (actual values read on open)."""
        cam = CameraCapture(source="test.mp4")
        assert cam.device_index == -1
        assert cam.width == 0
        assert cam.height == 0
        assert cam.fps == 0

    def test_source_ignores_device_index(self):
        """In file mode, a given device_index is ignored / 视频文件模式下 device_index 被忽略"""
        cam = CameraCapture(source="video.mp4", device_index=5)
        assert cam.device_index == -1

    def test_no_source_is_camera_mode(self):
        """Without source, camera mode is active (_is_file=False)."""
        cam = CameraCapture()
        assert cam.source is None
        assert cam._is_file is False
        assert cam.device_index == CAMERA.device_index

    def test_no_source_uses_custom_params(self):
        """In camera mode, custom params are kept / 摄像头模式下自定义参数被保留"""
        cam = CameraCapture(device_index=1, width=1280, height=720, fps=60)
        assert cam.device_index == 1
        assert cam.width == 1280
        assert cam.height == 720
        assert cam.fps == 60

    def test_initial_state_not_running(self):
        """After construction, not running and cap is None / 构造后未运行, cap 为 None"""
        cam = CameraCapture()
        assert cam.is_running is False
        assert cam.cap is None


class TestCameraCaptureContextManager:
    """Context manager (with statement) tests / 上下文管理器接口测试"""

    def test_context_manager_calls_open_and_close(self):
        """with calls open() on enter and close() on exit (mocked to avoid real hardware)."""
        cam = CameraCapture()
        with patch.object(cam, "open", return_value=True) as mock_open, \
                patch.object(cam, "close") as mock_close:
            with cam as c:
                # __enter__ should return the capture itself / __enter__ 返回采集器自身
                assert c is cam
                mock_open.assert_called_once()
            # close() should be called on exit / 退出时应调用 close
            mock_close.assert_called_once()

    def test_context_manager_open_failure_propagates(self):
        """When open() returns False, with still exits and calls close."""
        cam = CameraCapture()
        with patch.object(cam, "open", return_value=False), \
                patch.object(cam, "close") as mock_close:
            with cam as c:
                assert c is cam
            mock_close.assert_called_once()

    def test_context_manager_exception_still_closes(self):
        """On exception inside with, close() is still called (exception propagates)."""
        cam = CameraCapture()
        with patch.object(cam, "open", return_value=True), \
                patch.object(cam, "close") as mock_close:
            with pytest.raises(RuntimeError, match="boom"):
                with cam:
                    raise RuntimeError("boom")
            mock_close.assert_called_once()
