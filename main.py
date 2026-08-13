"""
DJI Action Camera Vision System - main entry point.
DJI 运动相机视觉识别系统主程序 (OpenVINO / CUDA + YOLO).

Flow: load model -> open camera -> loop (capture -> infer -> draw -> show).
按 q 退出, s 截图, r 重置 FPS.
"""

import sys
import time
import logging
import argparse
from pathlib import Path

import cv2
import numpy as np

# Add project root to path / 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

import config
from config import CAMERA, MODEL, DISPLAY, CLASSES
from camera_capture import CameraCapture
from detector import OpenVINODetector, DetectionResult
from visualizer import Visualizer, VideoWriter
from messages import t

# Backend display names / 后端显示名
_BACKEND_NAMES = {"openvino": "OpenVINO", "cuda": "CUDA", "tensorrt": "TensorRT"}

# Logging setup / 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DJI-Vision")


class VisionSystem:
    """Main controller for the vision system / 视觉识别系统主控制器."""

    def __init__(
        self,
        device: str = None,
        confidence: float = None,
        iou_threshold: float = None,
        classes: list[int] = None,
        show_display: bool = True,
        save_output: bool = False,
        mirror: bool = False,
        log_detections: bool = False,
        input_source: str = None,
    ):
        self.device = device
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.classes = classes
        self.show_display = show_display
        self.save_output = save_output
        self.mirror = mirror
        self.log_detections = log_detections
        self.input_source = input_source

        self.camera = None
        self.detector = None
        self.visualizer = Visualizer()
        self.video_writer = None
        self._detection_logger = None

        self._running = False
        self._frame_count = 0
        self._start_time = None
        self._cached_sysinfo = None

    def initialize(self) -> bool:
        """Initialize all components / 初始化所有组件."""
        logger.info("=" * 60)
        logger.info(f"  {t('system_title')}")
        backend_display = _BACKEND_NAMES.get(MODEL.backend, MODEL.backend.upper())
        logger.info(t("info_backend", backend=MODEL.backend.upper(), display=backend_display))
        logger.info("=" * 60)

        # 1. Load inference model / 加载推理模型
        logger.info(f"\n[1/3] {t('init_model')}...")
        self.detector = OpenVINODetector(
            device=self.device,
            conf_threshold=self.confidence,
            iou_threshold=self.iou_threshold,
            classes=self.classes,
        )
        if not self.detector.load():
            return False
        self.detector.warmup(iterations=3)

        # 2. Open camera or video file / 打开摄像头或视频文件
        logger.info(f"\n[2/3] {t('init_camera')}...")
        self.camera = CameraCapture(source=self.input_source)
        if not self.camera.open():
            return False
        if not self.camera.is_file:
            self.camera.warmup(frames=5)

        # 3. Init video writer and detection log (optional) / 初始化视频写入器和检测日志
        if self.save_output:
            logger.info(f"\n[3/3] {t('init_output')}...")
            DISPLAY.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.video_writer = VideoWriter(
                output_path=DISPLAY.output_path,
                fps=DISPLAY.output_fps,
            )
            # Use actual camera resolution / 使用摄像头实际分辨率
            actual_width = self.camera.width
            actual_height = self.camera.height
            if not self.video_writer.open((actual_width, actual_height)):
                logger.warning(t("video_out_skip"))
                self.video_writer = None

        if self.log_detections:
            self._detection_logger = DetectionLogger(DISPLAY.log_path)
            if not self._detection_logger.open():
                logger.warning(t("log_init_skip"))
                self._detection_logger = None

        logger.info(f"\n{t('init_done')}!")
        precision = "INT8" if MODEL.int8 else ("FP16" if MODEL.half else "FP32")
        logger.info(t("info_model", name=MODEL.model_name, precision=precision))
        logger.info(f"  {t('device_label')}: {self.detector.device}")
        source_type = t("source_video_file") if self.camera.is_file else t("source_uvc_camera")
        logger.info(t("info_source", type=source_type, w=self.camera.width, h=self.camera.height, fps=self.camera.fps))
        logger.info(t("info_thresholds", conf=self.detector.conf_threshold, iou=self.detector.iou_threshold))
        if self.classes is not None:
            class_names = [CLASSES[i] if i < len(CLASSES) else str(i) for i in self.classes]
            logger.info(t("info_class_filter", classes=class_names))
        if self.mirror:
            logger.info(t("info_mirror"))
        if self.log_detections and self._detection_logger:
            logger.info(t("info_detection_log", path=DISPLAY.log_path))
        logger.info(t("hotkey_hint"))

        return True

    def run(self):
        """Main loop / 主循环."""
        if not self.detector or not self.camera:
            logger.error(t("not_initialized"))
            return

        self._running = True
        self._frame_count = 0
        self._start_time = time.perf_counter()
        _consecutive_drops = 0

        try:
            while self._running:
                # 1. Capture frame / 取流
                frame = self.camera.read()
                if frame is None:
                    _consecutive_drops += 1
                    # End of video file: exit / 视频文件结束: 自动退出
                    if self.camera.is_file:
                        logger.info(t("video_finished"))
                        break
                    # Camera dropped >30 frames (~1s): likely disconnected / 摄像头连续丢帧超过 30 次: 可能断开
                    if _consecutive_drops > 30:
                        logger.error(t("camera_dropped"))
                        break
                    logger.warning(t("frame_drop", count=_consecutive_drops))
                    time.sleep(0.01)
                    continue
                _consecutive_drops = 0

                # 2. Inference / 推理
                result = self.detector.detect(frame)

                # 3. Visualization / 可视化
                if self.show_display or self.save_output:
                    extra_info = self._get_system_info()
                    output_frame = self.visualizer.draw(
                        frame, result, extra_info=extra_info, mirror=self.mirror
                    )
                else:
                    output_frame = frame

                # 3.5 Detection log / 检测日志
                if self._detection_logger:
                    self._detection_logger.log(self._frame_count, result)

                # 4. Output / 输出
                if self.show_display:
                    cv2.imshow(DISPLAY.window_name, output_frame)

                if self.video_writer:
                    self.video_writer.write(output_frame)

                # 5. Keyboard interaction / 键盘交互
                if self.show_display:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info(t("user_quit"))
                        break
                    elif key == ord("s"):
                        self._save_screenshot(output_frame)
                    elif key == ord("r"):
                        self.visualizer.reset_fps()
                        logger.info(t("fps_reset"))
                else:
                    # Headless mode: throttle to avoid 100% CPU spin
                    # 无显示模式: 限速以避免 CPU 占满
                    time.sleep(0.001)

                self._frame_count += 1

                # Print stats every 100 frames / 每 100 帧打印一次统计
                if self._frame_count % 100 == 0:
                    elapsed = time.perf_counter() - self._start_time
                    avg_fps = self._frame_count / elapsed
                    logger.info(
                        t(
                            "stats_line",
                            count=self._frame_count,
                            fps=f"{avg_fps:.1f}",
                            ms=f"{result.inference_time_ms:.1f}",
                            objects=result.num_objects,
                        )
                    )

        except KeyboardInterrupt:
            logger.info(f"\n{t('interrupted')}...")

        finally:
            self.shutdown()

    def _get_system_info(self) -> dict:
        """Get system resource usage (throttled to every 30 frames)."""
        # Throttle psutil calls to avoid per-frame overhead
        # 限制 psutil 调用频率, 避免逐帧开销
        if self._frame_count % 30 != 0 and self._cached_sysinfo is not None:
            return self._cached_sysinfo
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent
            self._cached_sysinfo = {"CPU": f"{cpu:.0f}%", "MEM": f"{mem:.0f}%"}
            return self._cached_sysinfo
        except Exception:
            return {}

    def _save_screenshot(self, frame: np.ndarray):
        """Save a screenshot / 保存截图."""
        screenshot_dir = Path("output/screenshots")
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        filename = screenshot_dir / f"screenshot_{int(time.time())}.jpg"
        cv2.imwrite(str(filename), frame)
        logger.info(t("screenshot_saved", path=filename))

    def shutdown(self):
        """Release all resources / 关闭所有资源."""
        logger.info(f"\n{t('system_closing')}...")

        if self._frame_count > 0 and self._start_time:
            elapsed = time.perf_counter() - self._start_time
            avg_fps = self._frame_count / elapsed
            logger.info(
                t(
                    "total_stats",
                    count=self._frame_count,
                    elapsed=f"{elapsed:.1f}",
                    fps=f"{avg_fps:.1f}",
                )
            )

        if self.camera:
            self.camera.close()

        if self.video_writer:
            self.video_writer.close()

        if self._detection_logger:
            self._detection_logger.close()

        if self.show_display:
            cv2.destroyAllWindows()

        self._running = False
        logger.info(t("system_ready"))

    def stop(self):
        """Stop from outside / 外部调用停止."""
        self._running = False


class DetectionLogger:
    """CSV logger for detections / 检测结果 CSV 日志记录器."""

    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self._file = None
        self._writer = None

    def open(self) -> bool:
        """Open log file and write CSV header / 打开日志文件, 写入 CSV 表头."""
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            import csv
            self._file = open(self.log_path, "w", newline="", encoding="utf-8")
            self._writer = csv.writer(self._file)
            self._writer.writerow([
                "timestamp", "frame", "class_id", "class_name",
                "confidence", "x1", "y1", "x2", "y2",
            ])
            logger.info(t("log_created", path=self.log_path))
            return True
        except Exception as e:
            logger.error(t("log_create_fail", error=e))
            return False

    def log(self, frame_number: int, result: DetectionResult):
        """Log one frame of detections / 记录一帧的检测结果."""
        if self._writer is None:
            return
        timestamp = time.strftime("%H:%M:%S")
        for det in result.detections:
            self._writer.writerow([
                timestamp, frame_number, det.class_id, det.class_name,
                f"{det.confidence:.4f}",
                f"{det.x1:.1f}", f"{det.y1:.1f}",
                f"{det.x2:.1f}", f"{det.y2:.1f}",
            ])

    def close(self):
        """Close the log file / 关闭日志文件."""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None
            logger.info(t("log_closed"))


def parse_classes(class_str: str) -> list[int]:
    """Parse class filter argument / 解析类别过滤参数 (e.g. "person,car,0")."""
    if not class_str:
        return None

    result = []
    for item in class_str.split(","):
        item = item.strip()
        if not item:
            continue
        if item.isdigit():
            cls_id = int(item)
            if 0 <= cls_id < len(CLASSES):
                result.append(cls_id)
            else:
                logger.warning(t("class_id_oor", id=cls_id, max=len(CLASSES) - 1))
        else:
            # Look up by name / 按名称查找
            if item in CLASSES:
                result.append(CLASSES.index(item))
            else:
                logger.warning(t("class_unknown", name=item))

    return result if result else None


def main():
    parser = argparse.ArgumentParser(
        description="DJI Action Camera Vision System / DJI 运动相机视觉识别系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples / 示例:
  python main.py                           # Default config (UVC camera) / 默认配置启动
  python main.py --device intel:cpu        # Use CPU inference / 使用 CPU 推理
  python main.py --confidence 0.7          # Raise confidence threshold / 提高置信度阈值
  python main.py --no-display --save        # No display, save video / 无显示, 保存视频
  python main.py --list-cameras            # List available cameras / 列出可用摄像头
  python main.py --input video.mp4         # Infer from video file / 从视频文件推理
  python main.py --classes person,car      # Detect persons and cars only / 仅检测行人和车辆
  python main.py --flip                    # Horizontal mirror / 水平镜像画面
  python main.py --log-detections          # Log detections to CSV / 记录检测结果到 CSV
  python main.py --lang en                 # English log messages / 英文日志
        """,
    )
    parser.add_argument("--device", type=str, default=None,
                        help="Inference device / 推理设备 (OpenVINO: intel:gpu/intel:cpu | CUDA: 0/cpu)")
    parser.add_argument("--confidence", type=float, default=None,
                        help="Confidence threshold / 置信度阈值 (default: 0.5)")
    parser.add_argument("--iou", type=float, default=None,
                        help="NMS IoU threshold / NMS IoU 阈值 (default: 0.5)")
    parser.add_argument("--classes", type=str, default=None,
                        help="Filter classes, comma-separated / 仅检测指定类别, 逗号分隔 (e.g. person,car,0)")
    parser.add_argument("--input", type=str, default=None,
                        help="Video file path / 视频文件路径 (read from file instead of camera)")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable live display window / 不显示实时画面窗口")
    parser.add_argument("--save", action="store_true",
                        help="Save output video to output/result.mp4 / 保存输出视频")
    parser.add_argument("--flip", action="store_true",
                        help="Horizontal mirror / 画面水平镜像 (相机倒装时使用)")
    parser.add_argument("--log-detections", action="store_true",
                        help="Log detections to CSV / 记录检测结果到 CSV (output/detections.csv)")
    parser.add_argument("--list-cameras", action="store_true",
                        help="List available cameras / 列出可用的摄像头设备")
    parser.add_argument("--lang", type=str, default=None, choices=["zh", "en"],
                        help="Language for log messages (zh/en) / 日志语言 (zh/en)")
    args = parser.parse_args()

    # Apply language override / 应用语言设置
    if args.lang:
        config.LANGUAGE = args.lang

    # List cameras / 列出摄像头
    if args.list_cameras:
        cam = CameraCapture()
        cam.list_devices()
        return

    # Check model exists / 检查模型是否存在
    if not MODEL.exported_path.exists():
        logger.error(t("model_not_found_path", path=MODEL.exported_path))
        if MODEL.needs_export:
            logger.error(t("model_export_hint"))
            logger.error(t("model_download_hint"))
        else:
            logger.error(t("model_pt_download_hint"))
        sys.exit(1)

    # Parse class filter / 解析类别过滤
    class_ids = parse_classes(args.classes) if args.classes else None

    # Start system / 启动系统
    system = VisionSystem(
        device=args.device,
        confidence=args.confidence,
        iou_threshold=args.iou,
        classes=class_ids,
        show_display=not args.no_display,
        save_output=args.save,
        mirror=args.flip,
        log_detections=args.log_detections,
        input_source=args.input,
    )

    if not system.initialize():
        logger.error(t("init_failed"))
        sys.exit(1)

    system.run()


if __name__ == "__main__":
    main()
