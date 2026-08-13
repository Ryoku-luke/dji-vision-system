"""
全局配置文件
====================
DJI 运动相机 + Intel Core Ultra 7 155H 视觉识别系统
所有可调参数集中在此文件管理
"""

from dataclasses import dataclass
from pathlib import Path


# ============================================================
# 1. 摄像头配置 (DJI Osmo Action UVC 模式)
# ============================================================
@dataclass
class CameraConfig:
    """UVC 摄像头采集配置"""
    device_index: int = 0                # UVC 设备索引 (通常 0 表示第一个)
    width: int = 1920                    # 采集分辨率宽度 (1080P)
    height: int = 1080                   # 采集分辨率高度
    fps: int = 30                        # 采集帧率 (UVC 模式下 25 或 30)
    buffer_size: int = 1                 # OpenCV 缓冲帧数 (1 = 最低延迟)
    api_preference: int = 700            # CAP_MSMF (Windows Media Foundation)


# ============================================================
# 2. 模型配置
# ============================================================
@dataclass
class ModelConfig:
    """YOLO 模型配置"""
    # --- 模型选择 ---
    # 可选: "yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt"
    # 155H 实测推荐: yolo26s (97fps INT8) 或 yolo26m (63fps INT8)
    model_name: str = "yolo26s.pt"

    # --- OpenVINO 导出参数 ---
    export_format: str = "openvino"
    imgsz: int = 640                     # 模型输入尺寸
    int8: bool = True                    # 启用 INT8 量化 (155H Arc GPU 上快 2~3 倍)
    half: bool = False                   # FP16 量化 (与 INT8 二选一)
    nms: bool = True                     # 导出时内嵌 NMS

    # --- INT8 量化校准数据集 ---
    # 首次导出 INT8 模型需要校准数据, 使用 COCO128 子集
    calib_data: str = "coco128.yaml"
    calib_fraction: float = 0.1          # 使用 10% 数据校准 (减少导出时间)

    # --- 推理设备 ---
    # "intel:gpu"  -> Arc 集成 GPU (推荐, 实测最快)
    # "intel:npu"  -> NPU (155H 上兼容性有限, 不推荐)
    # "intel:cpu"  -> 纯 CPU (最慢但最稳定)
    inference_device: str = "intel:gpu"

    # --- 检测参数 ---
    conf_threshold: float = 0.5          # 置信度阈值
    iou_threshold: float = 0.5           # NMS IoU 阈值

    # --- 路径 ---
    models_dir: Path = Path("models")
    exported_dir: Path = Path("models/exported")

    @property
    def model_path(self) -> Path:
        """原始 PyTorch 模型路径"""
        return self.models_dir / self.model_name

    @property
    def exported_path(self) -> Path:
        """导出后的 OpenVINO 模型目录路径"""
        precision = "int8" if self.int8 else ("fp16" if self.half else "fp32")
        stem = Path(self.model_name).stem
        return self.exported_dir / f"{stem}_{precision}_openvino_model"


# ============================================================
# 3. 显示与输出配置
# ============================================================
@dataclass
class DisplayConfig:
    """显示与输出配置"""
    show_window: bool = True             # 是否显示实时画面窗口
    window_name: str = "DJI Vision System"
    show_fps: bool = True                # 显示 FPS
    show_confidence: bool = True         # 显示置信度数值
    show_class_name: bool = True         # 显示类别名称
    box_thickness: int = 2               # 边界框线宽
    font_scale: float = 0.6             # 字体大小
    mirror: bool = False                # 画面水平镜像 (相机倒装时使用)
    save_output: bool = False            # 是否保存输出视频
    output_path: Path = Path("output/result.mp4")
    output_fps: int = 30
    # 检测结果日志
    log_detections: bool = False         # 是否记录检测结果到 CSV
    log_path: Path = Path("output/detections.csv")


# ============================================================
# 4. 类别配置
# ============================================================
# COCO 80 类 (YOLO 默认)
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


# ============================================================
# 5. 全局配置实例
# ============================================================
CAMERA = CameraConfig()
MODEL = ModelConfig()
DISPLAY = DisplayConfig()
CLASSES = COCO_CLASSES
