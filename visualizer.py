"""Visualization: draw detection results on video frames / 可视化: 在视频帧上绘制检测结果."""

import time
import logging
from collections import deque
from typing import Any

import cv2
import numpy as np

from config import DISPLAY
from detector import Detection, DetectionResult
from messages import t

logger = logging.getLogger(__name__)

# Class color palette (BGR), one fixed color per class / 类别颜色调色板 (BGR)
COLOR_PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (128, 0, 128), (0, 128, 128), (64, 0, 0), (0, 64, 0),
    (64, 64, 0), (64, 0, 64), (0, 64, 64), (192, 0, 0),
    (0, 192, 0), (0, 0, 192), (192, 192, 0), (192, 0, 192),
]


def get_color(class_id: int) -> tuple[int, int, int]:
    """Return a fixed BGR color for a class ID / 为类别 ID 返回固定 BGR 颜色."""
    return COLOR_PALETTE[class_id % len(COLOR_PALETTE)]


class FPSCounter:
    """FPS counter (sliding-window average) / FPS 计数器 (滑动窗口平均)."""

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.frame_times: deque[float] = deque(maxlen=window_size)
        self._last_time: float | None = None

    def update(self) -> float:
        """Record a frame timestamp, return current average FPS / 记录时间戳, 返回平均 FPS."""
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
        """Clear the frame time history / 清空帧时间历史记录."""
        self.frame_times.clear()
        self._last_time = None


class Visualizer:
    """Detection result visualizer / 检测结果可视化器."""

    def __init__(self):
        self.fps_counter = FPSCounter(window_size=30)

    def draw(
        self,
        frame: np.ndarray,
        result: DetectionResult,
        extra_info: dict | None = None,
        mirror: bool = False,
    ) -> np.ndarray:
        """Draw detection results on the frame / 在帧上绘制检测结果."""
        output = frame.copy()

        # Horizontal mirror (for inverted camera mount) / 画面水平镜像 (相机倒装时使用)
        if mirror:
            output = cv2.flip(output, 1)

        # Draw detection boxes / 绘制检测框
        for det in result.detections:
            self._draw_detection(output, det)

        # Update FPS / 更新 FPS
        current_fps = self.fps_counter.update()

        # Draw info panel / 绘制信息面板
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
        """Draw a single detection / 绘制单个检测结果."""
        color = get_color(det.class_id)
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)

        # Bounding box / 边界框
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, DISPLAY.box_thickness)

        # Build label text / 构建标签文本
        parts = []
        if DISPLAY.show_class_name:
            parts.append(det.class_name)
        if DISPLAY.show_confidence:
            parts.append(f"{det.confidence:.2f}")
        label = " ".join(parts)

        if label:
            # Label background size / 标签背景尺寸
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, DISPLAY.font_scale, 1
            )

            # Label background / 标签背景
            label_y = max(y1, th + 4)
            cv2.rectangle(
                frame,
                (x1, label_y - th - 4),
                (x1 + tw + 4, label_y),
                color,
                cv2.FILLED,
            )

            # Label text / 标签文字
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
    ):  # pylint: disable=too-many-locals
        """Draw the top-left info panel / 绘制左上角信息面板."""
        lines = []

        # Panel labels stay in English (drawn on video) / 面板标签保持英文 (绘制在视频上)
        if DISPLAY.show_fps:
            lines.append(f"FPS: {fps:.1f}")
            lines.append(f"Inference: {inference_ms:.1f}ms")
            lines.append(f"Objects: {num_objects}")

        for key, value in extra_info.items():
            lines.append(f"{key}: {value}")

        if not lines:
            return

        # Panel size / 面板尺寸
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

        # Clamp panel size to frame dimensions (prevents ROI out-of-bounds)
        # 将面板尺寸限制在帧范围内 (防止 ROI 越界)
        frame_h, frame_w = frame.shape[:2]
        panel_w = min(panel_w, frame_w)
        panel_h = min(panel_h, frame_h)

        # Semi-transparent background (only copy the panel region, not the full frame)
        # 半透明背景 (仅拷贝面板区域, 避免整帧拷贝)
        roi = frame[0:panel_h, 0:panel_w]
        overlay = roi.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), cv2.FILLED)
        cv2.addWeighted(overlay, 0.6, roi, 0.4, 0, roi)

        # Draw text / 绘制文字
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
        """Reset the FPS counter / 重置 FPS 计数器."""
        self.fps_counter.reset()


class VideoWriter:
    """Video output writer / 视频输出写入器."""

    def __init__(self, output_path, fps: int = 30, resolution: tuple | None = None):
        self.output_path = str(output_path)
        self.fps = fps
        self.resolution = resolution
        self.writer: Any = None

    def open(self, resolution: tuple | None = None):
        """Open the video writer / 打开视频写入器."""
        res = resolution or self.resolution
        if res is None:
            raise ValueError("Resolution required / 请指定视频分辨率")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        self.writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, res
        )

        if not self.writer.isOpened():
            logger.error(t("video_out_fail", path=self.output_path))
            return False

        logger.info(t("video_out_created", path=self.output_path, w=res[0], h=res[1], fps=self.fps))
        return True

    def write(self, frame: np.ndarray):
        """Write a frame / 写入一帧."""
        if self.writer is not None:
            self.writer.write(frame)

    def close(self):
        """Release resources / 释放资源."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
