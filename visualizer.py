"""
可视化模块
====================
负责在视频帧上绘制检测结果 (边界框、标签、FPS 等)
"""

import time
import logging
from collections import deque

import cv2
import numpy as np

from config import DISPLAY
from detector import Detection, DetectionResult

logger = logging.getLogger(__name__)


# 类别颜色调色板 (BGR 格式, 每个类别一个固定颜色)
COLOR_PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (128, 0, 128), (0, 128, 128), (64, 0, 0), (0, 64, 0),
    (64, 64, 0), (64, 0, 64), (0, 64, 64), (192, 0, 0),
    (0, 192, 0), (0, 0, 192), (192, 192, 0), (192, 0, 192),
]


def get_color(class_id: int) -> tuple[int, int, int]:
    """获取类别对应的颜色"""
    return COLOR_PALETTE[class_id % len(COLOR_PALETTE)]


class FPSCounter:
    """FPS 计数器 (滑动窗口平均)"""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times = deque(maxlen=window_size)
        self._last_time = None

    def update(self) -> float:
        """
        记录一帧的时间戳, 返回当前平均 FPS

        Returns:
            滑动窗口平均 FPS
        """
        now = time.perf_counter()
        if self._last_time is not None:
            self.frame_times.append(now - self._last_time)
        self._last_time = now

        if len(self.frame_times) < 2:
            return 0.0

        avg_interval = sum(self.frame_times) / len(self.frame_times)
        if avg_interval <= 0:
            return 0.0
        return 1.0 / avg_interval

    def reset(self):
        self.frame_times.clear()
        self._last_time = None


class Visualizer:
    """检测结果可视化器"""

    def __init__(self):
        self.fps_counter = FPSCounter(window_size=30)

    def draw(
        self,
        frame: np.ndarray,
        result: DetectionResult,
        extra_info: dict = None,
        mirror: bool = False,
    ) -> np.ndarray:
        """
        在帧上绘制检测结果

        Args:
            frame: 原始 BGR 图像
            result: 检测结果
            extra_info: 额外显示信息 (如 {"系统CPU": "45%"})
            mirror: 是否水平镜像画面

        Returns:
            绘制后的 BGR 图像
        """
        output = frame.copy()

        # 画面水平镜像 (相机倒装时使用)
        if mirror:
            output = cv2.flip(output, 1)

        # 绘制检测框
        for det in result.detections:
            self._draw_detection(output, det)

        # 更新 FPS
        current_fps = self.fps_counter.update()

        # 绘制信息面板
        if DISPLAY.show_fps or extra_info:
            self._draw_info_panel(
                output,
                fps=current_fps,
                num_objects=result.num_objects,
                inference_ms=result.inference_time_ms,
                extra_info=extra_info or {},
            )

        return output

    def _draw_detection(self, frame: np.ndarray, det: Detection):
        """绘制单个检测结果"""
        color = get_color(det.class_id)
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)

        # 绘制边界框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, DISPLAY.box_thickness)

        # 构建标签文本
        parts = []
        if DISPLAY.show_class_name:
            parts.append(det.class_name)
        if DISPLAY.show_confidence:
            parts.append(f"{det.confidence:.2f}")
        label = " ".join(parts)

        if label:
            # 计算标签背景尺寸
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, DISPLAY.font_scale, 1
            )

            # 绘制标签背景
            label_y = max(y1, th + 4)
            cv2.rectangle(
                frame,
                (x1, label_y - th - 4),
                (x1 + tw + 4, label_y),
                color,
                cv2.FILLED,
            )

            # 绘制标签文字
            cv2.putText(
                frame,
                label,
                (x1 + 2, label_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                DISPLAY.font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def _draw_info_panel(
        self,
        frame: np.ndarray,
        fps: float,
        num_objects: int,
        inference_ms: float,
        extra_info: dict,
    ):
        """绘制左上角信息面板"""
        lines = []

        if DISPLAY.show_fps:
            lines.append(f"FPS: {fps:.1f}")
            lines.append(f"Inference: {inference_ms:.1f}ms")
            lines.append(f"Objects: {num_objects}")

        for key, value in extra_info.items():
            lines.append(f"{key}: {value}")

        if not lines:
            return

        # 计算面板尺寸
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        padding = 8
        line_height = 20

        max_width = 0
        for line in lines:
            (tw, _), _ = cv2.getTextSize(line, font, scale, thickness)
            max_width = max(max_width, tw)

        panel_w = max_width + padding * 2
        panel_h = len(lines) * line_height + padding

        # 绘制半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), cv2.FILLED)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # 绘制文字
        for i, line in enumerate(lines):
            y = padding + (i + 1) * line_height - 6
            cv2.putText(
                frame,
                line,
                (padding, y),
                font,
                scale,
                (0, 255, 0),
                thickness,
                cv2.LINE_AA,
            )

    def reset_fps(self):
        """重置 FPS 计数器 (例如暂停后恢复时)"""
        self.fps_counter.reset()


class VideoWriter:
    """视频输出写入器"""

    def __init__(self, output_path, fps: int = 30, resolution: tuple = None):
        self.output_path = str(output_path)
        self.fps = fps
        self.resolution = resolution
        self.writer = None

    def open(self, resolution: tuple = None):
        """打开视频写入器"""
        res = resolution or self.resolution
        if res is None:
            raise ValueError("请指定视频分辨率")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, res
        )

        if not self.writer.isOpened():
            logger.error(f"无法创建视频文件: {self.output_path}")
            return False

        logger.info(f"视频输出: {self.output_path} ({res[0]}x{res[1]} @ {self.fps}fps)")
        return True

    def write(self, frame: np.ndarray):
        """写入一帧"""
        if self.writer is not None:
            self.writer.write(frame)

    def close(self):
        """释放资源"""
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
