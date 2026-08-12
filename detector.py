"""
目标检测模块
====================
基于 OpenVINO + YOLO 的目标检测推理引擎

核心流程:
  1. 加载导出后的 OpenVINO 模型
  2. 接收 BGR 图像帧
  3. 预处理 (resize + normalize) -> 推理 -> 后处理 (NMS)
  4. 返回检测结果 (边界框、类别、置信度)

支持设备:
  - "intel:gpu"  -> Arc 集成 GPU (推荐, 155H 实测最快)
  - "intel:npu"  -> NPU (155H 兼容性有限)
  - "intel:cpu"  -> 纯 CPU
"""

import time
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from config import MODEL, CLASSES

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """单个检测结果"""
    x1: float          # 边界框左上角 x
    y1: float          # 边界框左上角 y
    x2: float          # 边界框右下角 x
    y2: float          # 边界框右下角 y
    confidence: float  # 置信度
    class_id: int      # 类别 ID
    class_name: str    # 类别名称

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass
class DetectionResult:
    """单帧检测结果"""
    detections: list[Detection]
    inference_time_ms: float
    frame_shape: tuple[int, int]  # (height, width)

    @property
    def num_objects(self) -> int:
        return len(self.detections)

    @property
    def fps(self) -> float:
        """纯推理 FPS (不含取流和后处理时间)"""
        if self.inference_time_ms <= 0:
            return 0.0
        return 1000.0 / self.inference_time_ms

    def filter_by_class(self, class_ids: list[int]) -> list[Detection]:
        """按类别过滤检测结果"""
        return [d for d in self.detections if d.class_id in class_ids]

    def filter_by_confidence(self, threshold: float) -> list[Detection]:
        """按置信度过滤检测结果"""
        return [d for d in self.detections if d.confidence >= threshold]


class OpenVINODetector:
    """OpenVINO YOLO 目标检测器"""

    def __init__(
        self,
        model_path: Path = None,
        device: str = None,
        conf_threshold: float = None,
        iou_threshold: float = None,
    ):
        self.model_path = model_path if model_path is not None else MODEL.exported_path
        self.device = device if device is not None else MODEL.inference_device
        self.conf_threshold = conf_threshold if conf_threshold is not None else MODEL.conf_threshold
        self.iou_threshold = iou_threshold if iou_threshold is not None else MODEL.iou_threshold

        self.model = None
        self._is_loaded = False

    def load(self) -> bool:
        """
        加载 OpenVINO 模型

        Returns:
            True 如果加载成功
        """
        from ultralytics import YOLO

        if not self.model_path.exists():
            logger.error(f"模型文件不存在: {self.model_path}")
            logger.error("请先运行: python export_model.py")
            return False

        logger.info(f"加载模型: {self.model_path}")
        logger.info(f"推理设备: {self.device}")

        self.model = YOLO(str(self.model_path))

        # 验证设备可用性
        try:
            # 尝试在目标设备上进行一次空推理, 验证设备可用
            dummy = np.zeros((MODEL.imgsz, MODEL.imgsz, 3), dtype=np.uint8)
            self.model.predict(dummy, device=self.device, verbose=False)
            logger.info("模型加载成功, 设备验证通过")
        except Exception as e:
            logger.warning(f"设备 {self.device} 验证失败: {e}")
            logger.warning("回退到 CPU 模式")
            self.device = "intel:cpu"
            try:
                self.model.predict(dummy, device=self.device, verbose=False)
                logger.info("已回退到 CPU 模式")
            except Exception as e2:
                logger.error(f"CPU 模式验证也失败: {e2}")
                return False

        self._is_loaded = True
        return True

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        对单帧图像进行目标检测

        Args:
            frame: BGR 格式图像 (OpenCV 默认格式)

        Returns:
            DetectionResult 包含所有检测结果和推理耗时
        """
        if not self._is_loaded:
            logger.error("模型未加载, 请先调用 load()")
            return DetectionResult([], 0.0, frame.shape[:2])

        # 执行推理
        start_time = time.perf_counter()

        results = self.model.predict(
            frame,
            device=self.device,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=MODEL.imgsz,
            verbose=False,
        )

        inference_ms = (time.perf_counter() - start_time) * 1000

        # 解析结果
        detections = self._parse_results(results[0], frame.shape[:2])

        return DetectionResult(
            detections=detections,
            inference_time_ms=inference_ms,
            frame_shape=frame.shape[:2],
        )

    def detect_batch(self, frames: list[np.ndarray]) -> list[DetectionResult]:
        """
        批量检测多帧 (利用 OpenVINO 批量推理优化)

        Args:
            frames: BGR 图像帧列表

        Returns:
            检测结果列表 (与输入帧一一对应)
        """
        if not frames:
            return []

        if not self._is_loaded:
            logger.error("模型未加载, 请先调用 load()")
            return [DetectionResult([], 0.0, f.shape[:2]) for f in frames]

        start_time = time.perf_counter()

        results = self.model.predict(
            frames,
            device=self.device,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=MODEL.imgsz,
            verbose=False,
        )

        total_ms = (time.perf_counter() - start_time) * 1000
        avg_ms = total_ms / len(frames)

        return [
            DetectionResult(
                detections=self._parse_results(r, f.shape[:2]),
                inference_time_ms=avg_ms,
                frame_shape=f.shape[:2],
            )
            for r, f in zip(results, frames)
        ]

    def _parse_results(self, result, frame_shape: tuple[int, int]) -> list[Detection]:
        """解析 Ultralytics 推理结果为 Detection 列表"""
        detections = []

        if result.boxes is None:
            return detections

        boxes = result.boxes
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            conf = float(boxes.conf[i].cpu().numpy())
            cls_id = int(boxes.cls[i].cpu().numpy())

            class_name = CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id)

            detections.append(Detection(
                x1=float(xyxy[0]),
                y1=float(xyxy[1]),
                x2=float(xyxy[2]),
                y2=float(xyxy[3]),
                confidence=conf,
                class_id=cls_id,
                class_name=class_name,
            ))

        return detections

    def warmup(self, iterations: int = 3):
        """
        预热推理引擎: 跑几轮空推理, 让 OpenVINO 完成内核编译和缓存

        Args:
            iterations: 预热轮数
        """
        if not self._is_loaded:
            return

        logger.info(f"推理引擎预热中 ({iterations} 轮)...")
        dummy = np.zeros((MODEL.imgsz, MODEL.imgsz, 3), dtype=np.uint8)

        for i in range(iterations):
            self.model.predict(dummy, device=self.device, verbose=False)

        logger.info("预热完成, 推理引擎就绪")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model = None
        self._is_loaded = False
