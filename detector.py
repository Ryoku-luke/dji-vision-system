"""
Multi-backend YOLO object detection module (OpenVINO / CUDA / TensorRT).
多后端 YOLO 目标检测模块 (OpenVINO / CUDA / TensorRT)
"""

import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from config import MODEL, CLASSES
from messages import t

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single detection result."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

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
    """Detection results for a single frame."""
    detections: list[Detection]
    inference_time_ms: float
    frame_shape: tuple[int, int]  # (height, width)

    @property
    def num_objects(self) -> int:
        return len(self.detections)

    @property
    def fps(self) -> float:
        """Pure inference FPS (excludes capture and post-processing)."""
        if self.inference_time_ms <= 0:
            return 0.0
        return 1000.0 / self.inference_time_ms

    def filter_by_class(self, class_ids: list[int]) -> list[Detection]:
        return [d for d in self.detections if d.class_id in class_ids]

    def filter_by_confidence(self, threshold: float) -> list[Detection]:
        return [d for d in self.detections if d.confidence >= threshold]


class OpenVINODetector:
    """YOLO detector supporting multiple backends (OpenVINO / CUDA / TensorRT).
    支持多后端的 YOLO 目标检测器"""

    def __init__(
        self,
        model_path: Path | None = None,
        device: str | None = None,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
        classes: list[int] | None = None,
    ):
        self.model_path = model_path if model_path is not None else MODEL.exported_path
        self.device = device if device is not None else MODEL.inference_device
        self.conf_threshold = conf_threshold if conf_threshold is not None else MODEL.conf_threshold
        self.iou_threshold = iou_threshold if iou_threshold is not None else MODEL.iou_threshold
        # None = detect all classes, list = only specified classes
        # None = 检测所有类别, list = 仅检测指定类别
        self.classes = classes

        self.model: Any = None
        self._is_loaded = False
        self._predict_kwargs: dict[str, Any] = {}  # Cached kwargs for predict()

    def load(self) -> bool:
        """Load the model and verify the inference device."""
        # Check backend dependencies before loading
        # 加载前检查后端依赖是否安装
        if not self._check_backend_dependencies():
            return False

        from ultralytics import YOLO

        if not self.model_path.exists():
            logger.error(t("model_not_found"))
            if MODEL.needs_export:
                logger.error(t("model_export_hint"))
                logger.error(t("model_download_hint"))
            else:
                logger.error(t("model_download_first"))
                logger.error(t("model_export_or_download"))
            return False

        logger.info(t("loading_model", path=self.model_path))
        logger.info(t("infer_backend_info", backend=MODEL.backend, device=self.device))

        self.model = YOLO(str(self.model_path))

        # Verify device availability with a dummy inference
        # 用空推理验证设备可用性
        try:
            dummy = np.zeros((MODEL.imgsz, MODEL.imgsz, 3), dtype=np.uint8)
            self.model.predict(dummy, device=self.device, verbose=False)
            logger.info(t("model_loaded"))
        except Exception as e:
            logger.warning(t("device_verify_fail", device=self.device, error=e))
            # Select fallback device based on current backend
            # 根据当前后端选择回退设备
            fallback = self._get_fallback_device()
            if fallback is None:
                logger.error(t("no_fallback"))
                return False
            logger.warning(t("device_fallback", device=fallback))
            self.device = fallback
            try:
                self.model.predict(dummy, device=self.device, verbose=False)
                logger.info(t("device_fallback_ok", device=fallback))
            except Exception as e2:
                logger.error(t("device_fallback_fail", device=fallback, error=e2))
                return False

        self._is_loaded = True

        # Cache predict kwargs to avoid rebuilding every frame
        # 缓存 predict 参数, 避免每帧重建字典
        self._predict_kwargs = {
            "device": self.device,
            "conf": self.conf_threshold,
            "iou": self.iou_threshold,
            "imgsz": MODEL.imgsz,
            "verbose": False,
        }
        if self.classes is not None:
            self._predict_kwargs["classes"] = self.classes
        # Enable FP16 for CUDA backend (PyTorch runtime inference)
        # CUDA 后端启用 FP16 (PyTorch 运行时推理)
        if MODEL.backend == "cuda" and MODEL.half:
            self._predict_kwargs["half"] = True

        return True

    def _get_fallback_device(self) -> str:
        """Pick a fallback device based on the current backend."""
        if MODEL.backend == "openvino":
            return "intel:cpu"
        elif MODEL.backend in ("cuda", "tensorrt"):
            return "cpu"
        else:
            return "cpu"

    def _check_backend_dependencies(self) -> bool:
        """Check that required dependencies for the current backend are installed."""
        if MODEL.backend in ("cuda", "tensorrt"):
            try:
                import torch
                if not torch.cuda.is_available():
                    # Distinguish: CPU-only PyTorch vs. GPU not detected
                    # 区分: CPU 版 PyTorch vs. 未检测到 GPU
                    cuda_version = getattr(torch.version, "cuda", None)
                    if cuda_version is None:
                        logger.error(t("cuda_torch_cpu_only"))
                    else:
                        logger.error(t("cuda_unavailable", backend=MODEL.backend))
                    logger.error(t("cuda_check_1"))
                    logger.error(t("cuda_check_2"))
                    logger.error(t("cuda_install_cmd"))
                    logger.error(t("backend_switch_hint"))
                    return False
            except ImportError:
                logger.error(t("torch_not_installed"))
                logger.error(t("install_optional_deps"))
                logger.error(t("backend_switch_hint"))
                return False

            if MODEL.backend == "tensorrt":
                try:
                    import tensorrt  # noqa: F401
                except ImportError:
                    logger.error(t("tensorrt_not_installed"))
                    logger.error(t("tensorrt_install_hint"))
                    logger.error(t("backend_switch_cuda"))
                    return False

        if MODEL.backend == "openvino":
            try:
                import openvino  # noqa: F401
            except ImportError:
                logger.error(t("openvino_not_installed"))
                logger.error(t("install_openvino"))
                return False

        return True

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run detection on a single BGR frame."""
        if not self._is_loaded:
            logger.error(t("model_not_loaded"))
            return DetectionResult([], 0.0, frame.shape[:2])

        start_time = time.perf_counter()

        # Catch inference errors so a single frame failure doesn't crash
        # 捕获推理异常, 避免单帧失败导致程序崩溃
        try:
            results = self.model.predict(frame, **self._predict_kwargs)
        except Exception as e:
            logger.error(t("infer_failed", error=e))
            return DetectionResult([], 0.0, frame.shape[:2])

        inference_ms = (time.perf_counter() - start_time) * 1000

        detections = self._parse_results(results[0], frame.shape[:2])

        return DetectionResult(
            detections=detections,
            inference_time_ms=inference_ms,
            frame_shape=frame.shape[:2],
        )

    def detect_batch(self, frames: list[np.ndarray]) -> list[DetectionResult]:
        """Run batch detection on multiple BGR frames."""
        if not frames:
            return []

        if not self._is_loaded:
            logger.error(t("model_not_loaded"))
            return [DetectionResult([], 0.0, f.shape[:2]) for f in frames]

        start_time = time.perf_counter()

        # Batch inference also wraps in try/except
        # 批量推理同样添加异常捕获
        try:
            results = self.model.predict(frames, **self._predict_kwargs)
        except Exception as e:
            logger.error(t("infer_batch_failed", error=e))
            return [DetectionResult([], 0.0, f.shape[:2]) for f in frames]

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
        """Parse Ultralytics results into a list of Detection objects."""
        detections: list[Detection] = []

        if result.boxes is None:
            return detections

        boxes = result.boxes
        n = len(boxes)
        if n == 0:
            return detections

        # Batch GPU->CPU transfer: one call per tensor instead of per-box
        # 批量 GPU->CPU 传输: 每个 tensor 一次调用, 而非逐检测
        all_xyxy = boxes.xyxy.cpu().numpy()
        all_conf = boxes.conf.cpu().numpy()
        all_cls = boxes.cls.cpu().numpy()

        for i in range(n):
            cls_id = int(all_cls[i])
            class_name = CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id)
            detections.append(Detection(
                x1=float(all_xyxy[i][0]),
                y1=float(all_xyxy[i][1]),
                x2=float(all_xyxy[i][2]),
                y2=float(all_xyxy[i][3]),
                confidence=float(all_conf[i]),
                class_id=cls_id,
                class_name=class_name,
            ))

        return detections

    def warmup(self, iterations: int = 3):
        """Warm up the inference engine with dummy runs."""
        if not self._is_loaded:
            return

        logger.info(t("model_warmup", iterations=iterations))
        dummy = np.zeros((MODEL.imgsz, MODEL.imgsz, 3), dtype=np.uint8)

        for i in range(iterations):
            self.model.predict(dummy, **self._predict_kwargs)

        logger.info(t("model_warmup_done"))

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model = None
        self._is_loaded = False
