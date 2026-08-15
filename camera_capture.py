"""
UVC camera capture and video file reader module.
UVC 摄像头采集与视频文件读取模块

Captures frames from DJI Osmo Action cameras (UVC webcam mode) via OpenCV.
"""

import time
import logging
from typing import Any

import cv2
import numpy as np

from config import CAMERA
from messages import t

logger = logging.getLogger(__name__)


class CameraCapture:
    """UVC camera capture / video file reader. UVC 摄像头采集器 / 视频文件读取器"""

    def __init__(
        self,
        device_index: int | None = None,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
        source: str | None = None,
    ):
        """Initialize the capture device."""
        self.source = source
        self._is_file = source is not None

        if self._is_file:
            self.device_index = -1
            self.width = 0
            self.height = 0
            self.fps = 0
        else:
            self.device_index = device_index if device_index is not None else CAMERA.device_index
            self.width = width if width is not None else CAMERA.width
            self.height = height if height is not None else CAMERA.height
            self.fps = fps if fps is not None else CAMERA.fps

        self.cap: Any = None
        self._is_running = False
        self._read_failed = False  # Throttle consecutive read-failure warnings

    def open(self) -> bool:
        """Open the camera device or video file."""
        if self._is_file:
            return self._open_file()
        else:
            return self._open_camera()

    def _open_camera(self) -> bool:
        """Open the UVC camera device."""
        logger.info(t("camera_opening", index=self.device_index))

        self.cap = cv2.VideoCapture(self.device_index, CAMERA.api_preference)

        if not self.cap.isOpened():
            logger.error(t("camera_open_fail", index=self.device_index))
            logger.error(t("camera_check"))
            logger.error(t("camera_check_1"))
            logger.error(t("camera_check_2"))
            logger.error(t("camera_check_3"))
            return False

        # Set capture parameters
        # 设置采集参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Buffer size of 1 frame to minimize latency
        # 缓冲区设为 1 帧, 降低延迟
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA.buffer_size)

        # Verify actual params (some cameras ignore custom settings)
        # 验证实际参数 (部分相机不支持所有参数)
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        # Some cameras report FPS as 0.0
        # 某些相机不报告 FPS, 实际值为 0.0
        fps_display = f"{actual_fps:.0f}" if actual_fps > 0 else "N/A"
        logger.info(t("camera_opened", w=actual_w, h=actual_h, fps=fps_display))

        if actual_w != self.width or actual_h != self.height:
            logger.warning(t("res_mismatch", w=self.width, h=self.height, aw=actual_w, ah=actual_h))

        # Update to actual values for downstream modules
        # 更新为实际值, 供后续模块使用
        self.width = actual_w
        self.height = actual_h
        if actual_fps > 0:
            self.fps = int(actual_fps)

        self._is_running = True
        return True

    def _open_file(self) -> bool:
        """Open a video file."""
        from pathlib import Path

        assert self.source is not None  # _is_file guarantees source is set
        file_path = Path(self.source)
        if not file_path.exists():
            logger.error(t("video_not_found", path=self.source))
            return False

        logger.info(t("video_opening", path=self.source))
        self.cap = cv2.VideoCapture(str(file_path))

        if not self.cap.isOpened():
            logger.error(t("video_open_fail", path=self.source))
            return False

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Use explicit > 0 check instead of or to avoid falsy semantics mismatch
        # 使用显式 > 0 判断替代 or, 避免 falsy 语义不一致
        file_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.fps = file_fps if file_fps > 0 else 30

        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(t("video_opened", w=self.width, h=self.height, fps=self.fps, frames=frame_count))

        self._is_running = True
        return True

    def read(self) -> np.ndarray | None:
        """Read a single frame. Returns BGR image or None on failure."""
        if not self._is_running or self.cap is None:
            logger.error(t("camera_not_open"))
            return None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            # Throttle warnings: only log first failure, not every consecutive one
            # 限制警告频率: 仅首次失败记录, 非每次都记录
            if not self._read_failed:
                logger.warning(t("frame_read_fail"))
                self._read_failed = True
            return None

        self._read_failed = False
        return frame

    def read_batch(self, count: int = 1) -> list[np.ndarray]:
        """Read multiple consecutive frames (for warmup or batch processing)."""
        frames = []
        for _ in range(count):
            frame = self.read()
            if frame is not None:
                frames.append(frame)
        return frames

    def warmup(self, frames: int = 5):
        """Warm up: read and discard frames to stabilize auto-exposure/white balance."""
        logger.info(t("camera_warmup", frames=frames))
        for i in range(frames):
            self.read()
            time.sleep(0.05)
        logger.info(t("camera_warmup_done"))

    def list_devices(self) -> list[int]:
        """List available camera device indices by probing each one."""
        available = []
        for i in range(5):  # Check up to 5 devices
            cap = cv2.VideoCapture(i, CAMERA.api_preference)
            try:
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        available.append(i)
            except Exception:
                pass  # Skip devices that throw on read
            finally:
                cap.release()
        logger.info(t("camera_devices", devices=available))
        return available

    @property
    def is_running(self) -> bool:
        return self._is_running

    @property
    def is_file(self) -> bool:
        """Whether the source is a video file rather than a live camera."""
        return self._is_file

    def close(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._is_running = False
        logger.info(t("camera_closed"))

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()
