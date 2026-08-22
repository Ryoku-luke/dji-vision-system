"""
Global configuration for DJI Action Camera Vision System.
All tunable parameters live here — edit this file only.
"""

import platform
from dataclasses import dataclass
from pathlib import Path

# Language: "zh" (Chinese) or "en" (English)
LANGUAGE = "zh"

# Project root (this file's directory); ensures paths resolve regardless of CWD.
# 项目根目录 (本文件所在目录); 确保路径与 CWD 无关, 修复 BUG-03。
PROJECT_ROOT = Path(__file__).resolve().parent


def _detect_api_backend() -> int:
    """Pick the OpenCV VideoCapture backend for the current OS."""
    system = platform.system()
    if system == "Windows":
        return 700   # CAP_MSMF
    elif system == "Linux":
        return 200   # CAP_V4L2
    elif system == "Darwin":
        return 1200  # CAP_AVFOUNDATION
    else:
        return 0     # CAP_ANY


@dataclass
class CameraConfig:
    """UVC camera capture settings (cross-platform)."""
    device_index: int = 0
    width: int = 1920
    height: int = 1080
    fps: int = 30
    buffer_size: int = 1
    api_preference: int = 0  # 0 = auto-detect at runtime

    def __post_init__(self):
        if self.api_preference == 0:
            self.api_preference = _detect_api_backend()


@dataclass
class ModelConfig:
    """YOLO model and inference backend settings."""
    model_name: str = "yolo26s.pt"

    # Backend: "openvino" (Intel), "cuda" (NVIDIA), "tensorrt" (NVIDIA, fastest)
    backend: str = "openvino"

    # Export parameters (OpenVINO / TensorRT)
    export_format: str = "openvino"
    imgsz: int = 640
    int8: bool = True
    half: bool = False
    nms: bool = True

    # INT8 calibration
    calib_data: str = "coco128.yaml"
    calib_fraction: float = 0.1

    # Inference device
    # OpenVINO: intel:gpu / intel:cpu | CUDA: 0 / cpu | TensorRT: 0
    inference_device: str = "intel:gpu"

    # Detection thresholds
    conf_threshold: float = 0.5
    iou_threshold: float = 0.5

    # Paths (absolute, based on PROJECT_ROOT; immune to caller's CWD — BUG-03)
    # 路径基于 PROJECT_ROOT 的绝对路径, 不受调用方 CWD 影响 (BUG-03)
    models_dir: Path = PROJECT_ROOT / "models"
    exported_dir: Path = PROJECT_ROOT / "models" / "exported"

    @property
    def model_path(self) -> Path:
        """Path to the original PyTorch model."""
        return self.models_dir / self.model_name

    @property
    def exported_path(self) -> Path:
        """Path to the exported model (auto-generated from backend + precision)."""
        precision = "int8" if self.int8 else ("fp16" if self.half else "fp32")
        stem = Path(self.model_name).stem
        if self.backend == "openvino":
            return self.exported_dir / f"{stem}_{precision}_openvino_model"
        elif self.backend == "tensorrt":
            return self.exported_dir / f"{stem}_{precision}_engine"
        else:
            return self.model_path

    @property
    def needs_export(self) -> bool:
        """Whether this backend requires a model export step."""
        return self.backend in ("openvino", "tensorrt")


@dataclass
class DisplayConfig:
    """Display and output settings."""
    show_window: bool = True
    window_name: str = "DJI Vision System"
    show_fps: bool = True
    show_confidence: bool = True
    show_class_name: bool = True
    box_thickness: int = 2
    font_scale: float = 0.6
    mirror: bool = False
    save_output: bool = False
    output_path: Path = Path("output/result.mp4")
    output_fps: int = 30
    log_detections: bool = False
    log_path: Path = Path("output/detections.csv")


# COCO 80 classes (YOLO default)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Global instances
CAMERA = CameraConfig()
MODEL = ModelConfig()
DISPLAY = DisplayConfig()
CLASSES = COCO_CLASSES
