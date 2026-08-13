"""
DJI 运动相机视觉识别系统 - 主程序
=====================================
Intel Core Ultra 7 155H + OpenVINO + YOLO

启动流程:
  1. 加载 OpenVINO 模型
  2. 打开 UVC 摄像头
  3. 循环: 取流 -> 推理 -> 可视化 -> 显示
  4. 按 q 退出, 按 s 截图, 按 r 重置 FPS

用法:
    python main.py                           # 使用 config.py 默认配置
    python main.py --no-display              # 不显示窗口 (仅推理, 用于性能测试)
    python main.py --save                    # 保存输出视频
    python main.py --device intel:cpu        # 指定推理设备
    python main.py --confidence 0.7          # 调整置信度阈值
"""

import sys
import time
import logging
import argparse
from pathlib import Path

import cv2
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config import CAMERA, MODEL, DISPLAY, CLASSES
from camera_capture import CameraCapture
from detector import OpenVINODetector, DetectionResult
from visualizer import Visualizer, VideoWriter

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DJI-Vision")


class VisionSystem:
    """视觉识别系统主控制器"""

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

    def initialize(self) -> bool:
        """初始化所有组件"""
        logger.info("=" * 60)
        logger.info("  DJI 运动相机视觉识别系统")
        logger.info(f"  后端: {MODEL.backend.upper()} | YOLO + {'OpenVINO' if MODEL.backend == 'openvino' else 'CUDA' if MODEL.backend == 'cuda' else 'TensorRT'}")
        logger.info("=" * 60)

        # 1. 加载推理模型
        logger.info("\n[1/3] 加载推理模型...")
        self.detector = OpenVINODetector(
            device=self.device,
            conf_threshold=self.confidence,
            iou_threshold=self.iou_threshold,
            classes=self.classes,
        )
        if not self.detector.load():
            return False
        self.detector.warmup(iterations=3)

        # 2. 打开摄像头或视频文件
        logger.info("\n[2/3] 打开视频源...")
        self.camera = CameraCapture(source=self.input_source)
        if not self.camera.open():
            return False
        if not self.camera._is_file:
            self.camera.warmup(frames=5)

        # 3. 初始化视频写入器和检测日志 (可选)
        if self.save_output:
            logger.info("\n[3/3] 初始化视频输出...")
            DISPLAY.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.video_writer = VideoWriter(
                output_path=DISPLAY.output_path,
                fps=DISPLAY.output_fps,
            )
            # BUG-09 修复: 使用摄像头实际分辨率
            actual_width = self.camera.width
            actual_height = self.camera.height
            if not self.video_writer.open((actual_width, actual_height)):
                logger.warning("视频输出初始化失败, 将忽略保存功能")
                self.video_writer = None

        if self.log_detections:
            self._detection_logger = DetectionLogger(DISPLAY.log_path)
            if not self._detection_logger.open():
                logger.warning("检测日志初始化失败, 将忽略日志记录")
                self._detection_logger = None

        logger.info("\n系统初始化完成!")
        logger.info(f"  模型: {MODEL.model_name} ({'INT8' if MODEL.int8 else 'FP16' if MODEL.half else 'FP32'})")
        logger.info(f"  设备: {self.detector.device}")
        # BUG-09 修复: 显示实际分辨率而非配置分辨率
        source_type = "视频文件" if self.camera._is_file else "UVC 摄像头"
        logger.info(f"  视频源: {source_type} ({self.camera.width}x{self.camera.height} @ {self.camera.fps}fps)")
        logger.info(f"  置信度: {self.detector.conf_threshold} | IoU: {self.detector.iou_threshold}")
        if self.classes is not None:
            class_names = [CLASSES[i] if i < len(CLASSES) else str(i) for i in self.classes]
            logger.info(f"  类别过滤: {class_names}")
        if self.mirror:
            logger.info("  画面镜像: 已启用")
        if self.log_detections and self._detection_logger:
            logger.info(f"  检测日志: {DISPLAY.log_path}")
        logger.info("\n按 q 退出 | s 截图 | r 重置FPS计数\n")

        return True

    def run(self):
        """主循环"""
        if not self.detector or not self.camera:
            logger.error("系统未初始化, 请先调用 initialize()")
            return

        self._running = True
        self._frame_count = 0
        self._start_time = time.perf_counter()
        _consecutive_drops = 0

        try:
            while self._running:
                # --- 1. 取流 ---
                frame = self.camera.read()
                if frame is None:
                    _consecutive_drops += 1
                    # 视频文件结束: 自动退出
                    if self.camera._is_file:
                        logger.info("视频文件播放完毕")
                        break
                    # 摄像头连续丢帧超过 30 次 (约 1 秒): 可能断开
                    if _consecutive_drops > 30:
                        logger.error("摄像头连续丢帧, 可能已断开连接")
                        break
                    logger.warning(f"丢帧 ({_consecutive_drops}), 等待下一帧...")
                    time.sleep(0.01)
                    continue
                _consecutive_drops = 0

                # --- 2. 推理 ---
                result = self.detector.detect(frame)

                # --- 3. 可视化 ---
                if self.show_display or self.save_output:
                    # 获取系统资源信息
                    extra_info = self._get_system_info()

                    output_frame = self.visualizer.draw(
                        frame, result, extra_info=extra_info, mirror=self.mirror
                    )
                else:
                    output_frame = frame

                # --- 3.5 检测日志 ---
                if self._detection_logger:
                    self._detection_logger.log(self._frame_count, result)

                # --- 4. 输出 ---
                if self.show_display:
                    cv2.imshow(DISPLAY.window_name, output_frame)

                if self.video_writer:
                    self.video_writer.write(output_frame)

                # --- 5. 键盘交互 ---
                if self.show_display:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        logger.info("用户退出")
                        break
                    elif key == ord("s"):
                        self._save_screenshot(output_frame)
                    elif key == ord("r"):
                        self.visualizer.reset_fps()
                        logger.info("FPS 计数器已重置")

                self._frame_count += 1

                # 每 100 帧打印一次统计
                if self._frame_count % 100 == 0:
                    elapsed = time.perf_counter() - self._start_time
                    avg_fps = self._frame_count / elapsed
                    logger.info(
                        f"已处理 {self._frame_count} 帧, "
                        f"平均 {avg_fps:.1f} FPS, "
                        f"推理 {result.inference_time_ms:.1f} ms/帧, "
                        f"检测到 {result.num_objects} 个目标"
                    )

        except KeyboardInterrupt:
            logger.info("\n收到中断信号, 正在退出...")

        finally:
            self.shutdown()

    def _get_system_info(self) -> dict:
        """获取系统资源使用信息 (用于显示)"""
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent
            return {"CPU": f"{cpu:.0f}%", "MEM": f"{mem:.0f}%"}
        except Exception:
            return {}

    def _save_screenshot(self, frame: np.ndarray):
        """保存截图"""
        screenshot_dir = Path("output/screenshots")
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        filename = screenshot_dir / f"screenshot_{int(time.time())}.jpg"
        cv2.imwrite(str(filename), frame)
        logger.info(f"截图已保存: {filename}")

    def shutdown(self):
        """关闭所有资源"""
        logger.info("\n正在关闭系统...")

        if self._frame_count > 0 and self._start_time:
            elapsed = time.perf_counter() - self._start_time
            avg_fps = self._frame_count / elapsed
            logger.info(f"总计: {self._frame_count} 帧, {elapsed:.1f}s, 平均 {avg_fps:.1f} FPS")

        if self.camera:
            self.camera.close()

        if self.video_writer:
            self.video_writer.close()

        if self._detection_logger:
            self._detection_logger.close()

        if self.show_display:
            cv2.destroyAllWindows()

        self._running = False
        logger.info("系统已关闭")

    def stop(self):
        """外部调用停止"""
        self._running = False


class DetectionLogger:
    """检测结果 CSV 日志记录器"""

    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self._file = None
        self._writer = None

    def open(self) -> bool:
        """打开日志文件, 写入 CSV 表头"""
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            import csv
            self._file = open(self.log_path, "w", newline="", encoding="utf-8")
            self._writer = csv.writer(self._file)
            self._writer.writerow([
                "timestamp", "frame", "class_id", "class_name",
                "confidence", "x1", "y1", "x2", "y2",
            ])
            logger.info(f"检测日志已创建: {self.log_path}")
            return True
        except Exception as e:
            logger.error(f"无法创建检测日志: {e}")
            return False

    def log(self, frame_number: int, result: DetectionResult):
        """记录一帧的检测结果"""
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
        """关闭日志文件"""
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None
            logger.info("检测日志已关闭")


def parse_classes(class_str: str) -> list[int]:
    """
    解析类别过滤参数

    Args:
        class_str: 逗号分隔的类别名称或 ID (如 "person,car,0,truck")

    Returns:
        类别 ID 列表
    """
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
                logger.warning(f"类别 ID {cls_id} 超出范围 (0-{len(CLASSES)-1}), 已忽略")
        else:
            # 按名称查找
            if item in CLASSES:
                result.append(CLASSES.index(item))
            else:
                logger.warning(f"未知类别名称: {item}, 已忽略")

    return result if result else None


def main():
    parser = argparse.ArgumentParser(
        description="DJI 运动相机视觉识别系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py                           # 默认配置启动 (UVC 摄像头)
  python main.py --device intel:cpu        # 使用 CPU 推理
  python main.py --confidence 0.7          # 提高置信度阈值
  python main.py --no-display --save       # 无显示, 保存视频
  python main.py --list-cameras            # 列出可用摄像头
  python main.py --input video.mp4         # 从视频文件推理
  python main.py --classes person,car      # 仅检测行人和车辆
  python main.py --flip                    # 水平镜像画面
  python main.py --log-detections          # 记录检测结果到 CSV
        """,
    )
    parser.add_argument("--device", type=str, default=None,
                        help="推理设备 (OpenVINO: intel:gpu/intel:cpu | CUDA: 0/cpu)")
    parser.add_argument("--confidence", type=float, default=None,
                        help="置信度阈值 (默认: 0.5)")
    parser.add_argument("--iou", type=float, default=None,
                        help="NMS IoU 阈值 (默认: 0.5)")
    parser.add_argument("--classes", type=str, default=None,
                        help="仅检测指定类别, 逗号分隔 (如 person,car,0)")
    parser.add_argument("--input", type=str, default=None,
                        help="视频文件路径 (指定后从文件读取, 而非摄像头)")
    parser.add_argument("--no-display", action="store_true",
                        help="不显示实时画面窗口")
    parser.add_argument("--save", action="store_true",
                        help="保存输出视频到 output/result.mp4")
    parser.add_argument("--flip", action="store_true",
                        help="画面水平镜像 (相机倒装时使用)")
    parser.add_argument("--log-detections", action="store_true",
                        help="记录检测结果到 CSV 文件 (output/detections.csv)")
    parser.add_argument("--list-cameras", action="store_true",
                        help="列出可用的摄像头设备")
    args = parser.parse_args()

    # 列出摄像头
    if args.list_cameras:
        cam = CameraCapture()
        cam.list_devices()
        return

    # 检查模型是否存在
    if not MODEL.exported_path.exists():
        logger.error(f"未找到导出的模型: {MODEL.exported_path}")
        logger.error("请先运行模型导出: python export_model.py")
        logger.error(f"  或指定其他模型: python export_model.py --model yolo26n.pt")
        sys.exit(1)

    # 解析类别过滤
    class_ids = parse_classes(args.classes) if args.classes else None

    # 启动系统
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
        logger.error("系统初始化失败, 请检查错误信息")
        sys.exit(1)

    system.run()


if __name__ == "__main__":
    main()
