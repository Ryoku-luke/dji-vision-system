"""
UVC 摄像头采集模块
====================
负责从 DJI Osmo Action 相机 (UVC 网络摄像头模式) 采集视频帧
基于 OpenCV VideoCapture 实现

关键点:
  - DJI 相机在 UVC 模式下被系统识别为标准 USB 摄像头
  - Windows 上使用 MSMF (Media Foundation) 后端, 兼容性最好
  - 设置 buffer_size=1 可最大限度降低取流延迟
"""

import time
import logging
import cv2
import numpy as np

from config import CAMERA

logger = logging.getLogger(__name__)


class CameraCapture:
    """UVC 摄像头采集器"""

    def __init__(
        self,
        device_index: int = None,
        width: int = None,
        height: int = None,
        fps: int = None,
    ):
        self.device_index = device_index if device_index is not None else CAMERA.device_index
        self.width = width or CAMERA.width
        self.height = height or CAMERA.height
        self.fps = fps or CAMERA.fps

        self.cap = None
        self._is_running = False

    def open(self) -> bool:
        """
        打开摄像头设备

        Returns:
            True 如果成功打开
        """
        logger.info(f"正在打开 UVC 摄像头 (设备索引: {self.device_index})...")

        # 使用 MSMF 后端 (Windows 推荐)
        self.cap = cv2.VideoCapture(self.device_index, CAMERA.api_preference)

        if not self.cap.isOpened():
            logger.error(f"无法打开摄像头设备 {self.device_index}")
            logger.error("请检查:")
            logger.error("  1. DJI 相机已通过 Type-C 线连接电脑")
            logger.error("  2. 相机屏幕上已选择 '网络摄像头' 模式")
            logger.error("  3. 没有其他程序正在占用该摄像头")
            return False

        # 设置采集参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        # 设置缓冲区为 1 帧, 降低延迟
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA.buffer_size)

        # 验证实际参数 (部分相机不支持所有参数, 实际值可能不同)
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        logger.info(f"摄像头已打开: {actual_w}x{actual_h} @ {actual_fps:.0f}fps")

        if actual_w != self.width or actual_h != self.height:
            logger.warning(f"请求 {self.width}x{self.height}, 实际 {actual_w}x{actual_h}"
                          f" (部分相机不支持自定义分辨率)")

        self._is_running = True
        return True

    def read(self) -> np.ndarray | None:
        """
        读取一帧画面

        Returns:
            BGR 格式的图像帧, 或 None (读取失败时)
        """
        if not self._is_running or self.cap is None:
            logger.error("摄像头未打开, 请先调用 open()")
            return None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            logger.warning("读取帧失败, 可能是摄像头断开连接")
            return None

        return frame

    def read_batch(self, count: int = 1) -> list[np.ndarray]:
        """
        连续读取多帧 (用于预热或批量处理)

        Args:
            count: 读取帧数

        Returns:
            图像帧列表
        """
        frames = []
        for _ in range(count):
            frame = self.read()
            if frame is not None:
                frames.append(frame)
        return frames

    def warmup(self, frames: int = 5):
        """
        预热: 读取若干帧丢弃, 让相机自动曝光/白平衡稳定

        Args:
            frames: 预热帧数
        """
        logger.info(f"摄像头预热中 ({frames} 帧)...")
        for i in range(frames):
            self.read()
            time.sleep(0.05)
        logger.info("预热完成")

    def list_devices(self) -> list[int]:
        """
        列出可用的摄像头设备索引 (逐个尝试打开)

        Returns:
            可用设备索引列表
        """
        available = []
        for i in range(5):  # 最多检查 5 个设备
            cap = cv2.VideoCapture(i, CAMERA.api_preference)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available.append(i)
            cap.release()
        logger.info(f"可用摄像头设备: {available}")
        return available

    @property
    def is_running(self) -> bool:
        return self._is_running

    def close(self):
        """释放摄像头资源"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._is_running = False
        logger.info("摄像头已关闭")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()
