"""
Export YOLO PyTorch models to OpenVINO INT8 format.
导出 YOLO 模型为 OpenVINO INT8 格式。

First run downloads weights and performs INT8 calibration.

Usage / 用法:
    python export_model.py                    # Use config.py defaults
    python export_model.py --model yolo26m    # Specify model
    python export_model.py --device cpu       # Use CPU (no GPU)
"""

import argparse
import sys
import shutil
import logging
from pathlib import Path

# Add project root to path / 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config import MODEL

logger = logging.getLogger(__name__)


def export_model(model_name: str = None, export_device: str = "cpu"):
    """Export a YOLO model to OpenVINO format. 导出 YOLO 模型为 OpenVINO 格式。

    Args:
        model_name: Model filename (e.g. "yolo26s.pt"); uses config default if None.
        export_device: Device for export ("cpu" or "0" for GPU).
    """
    from ultralytics import YOLO

    # Determine model name / 确定模型名称
    name = model_name or MODEL.model_name
    model_path = MODEL.models_dir / name

    print("=" * 60)
    print(f"  Model export / 模型导出: {name}")
    print(f"  Target format / 目标格式: OpenVINO ({'INT8' if MODEL.int8 else 'FP16' if MODEL.half else 'FP32'})")
    print(f"  Input size / 输入尺寸: {MODEL.imgsz}x{MODEL.imgsz}")
    print("=" * 60)

    # Check if model file exists; download if missing / 检查模型文件是否存在，不存在则自动下载
    if not model_path.exists():
        print(f"\n[1/3] Downloading model / 下载模型: {name}")
        MODEL.models_dir.mkdir(parents=True, exist_ok=True)
        # YOLO constructor auto-downloads to the current working directory
        yolo = YOLO(name)
        # Move to models dir (download path is relative to CWD)
        downloaded = Path(name)
        if downloaded.exists():
            shutil.move(str(downloaded), str(model_path))
        else:
            # Some versions download to ~/.config/ultralytics or elsewhere
            logger.warning(
                f"Downloaded model not found in CWD: {name}; please check the model path. "
                f"/ 未在当前目录找到下载的模型文件 {name}, 请检查模型路径"
            )
        print(f"      Model saved to / 模型已保存到: {model_path}")
    else:
        print(f"\n[1/3] Model already exists / 模型已存在: {model_path}")
        yolo = YOLO(str(model_path))

    # Build export kwargs / 构建导出参数
    export_kwargs = {
        "format": MODEL.export_format,
        "imgsz": MODEL.imgsz,
        "nms": MODEL.nms,
        "device": export_device,
    }

    if MODEL.int8:
        export_kwargs["int8"] = True
        export_kwargs["data"] = MODEL.calib_data
        export_kwargs["fraction"] = MODEL.calib_fraction
        print(f"\n[2/3] Exporting OpenVINO INT8 (calib data: {MODEL.calib_data})")
        print(f"      导出 OpenVINO INT8 (校准数据: {MODEL.calib_data})")
        print(f"      Calibration fraction / 校准比例: {MODEL.calib_fraction*100:.0f}%")
        print(f"      Note: first INT8 export downloads the calibration dataset, please be patient.")
        print(f"      注意: 首次 INT8 导出需要下载校准数据集, 请耐心等待...")
    elif MODEL.half:
        export_kwargs["half"] = True
        print(f"\n[2/3] Exporting OpenVINO FP16 / 导出 OpenVINO FP16")
    else:
        print(f"\n[2/3] Exporting OpenVINO FP32 / 导出 OpenVINO FP32")

    # Run export / 执行导出
    exported_path = yolo.export(**export_kwargs)
    exported_path = Path(exported_path)
    print(f"      Export done / 导出完成: {exported_path}")

    # Move to target directory / 重命名到目标目录
    MODEL.exported_dir.mkdir(parents=True, exist_ok=True)
    target = MODEL.exported_path

    if exported_path != target:
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(exported_path), str(target))

    print(f"\n[3/3] Model ready / 模型已就绪: {target}")
    print("Export complete! You can now run main.py to start inference.")
    print("导出完成! 现在可以运行 main.py 启动推理:")
    print(f"  python main.py")

    # Print model size info / 打印模型大小信息
    if target.is_dir():
        total_size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
        print(f"\n  Model size / 模型大小: {total_size / 1024 / 1024:.1f} MB")

    return target


def benchmark_model(model_path: Path = None):
    """Benchmark the exported model. 对导出的模型进行基准测试。

    Args:
        model_path: Path to the OpenVINO model; uses config default if None.
    """
    from ultralytics import YOLO

    path = model_path or MODEL.exported_path
    print(f"\n{'=' * 60}")
    print(f"  Benchmark / 基准测试: {path.name}")
    print(f"  Device / 设备: {MODEL.inference_device}")
    print(f"{'=' * 60}\n")

    model = YOLO(str(path))

    # Run benchmark / 运行基准测试
    results = model.benchmark(
        data="coco128.yaml",
        imgsz=MODEL.imgsz,
        device=MODEL.inference_device,
        verbose=True,
    )

    print(f"\nBenchmark results / 基准测试结果:")
    # Some ultralytics versions return a float (mAP) instead of a dict.
    # Handle both cases for version compatibility.
    # 某些 ultralytics 版本返回 float (mAP) 而非 dict，需兼容处理。
    if isinstance(results, dict):
        inference_time = results.get('speed/inference') or results.get('inference_time')
        map_val = results.get('metrics/mAP50-95(B)') or results.get('mAP50-95(B)')
    else:
        # Some ultralytics versions return a float directly / 某些版本直接返回 float
        inference_time = None
        map_val = results
    if inference_time is not None:
        print(f"  Inference speed / 推理速度: {inference_time:.2f} ms/im")
        fps = 1000.0 / inference_time if inference_time > 0 else 0.0
        print(f"  Frame rate / 对应帧率: {fps:.1f} FPS")
    else:
        print(f"  Inference speed / 推理速度: N/A")
        print(f"  Frame rate / 对应帧率: N/A")
    print(f"  mAP50-95: {map_val if map_val is not None else 'N/A'}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLO model export tool / YOLO 模型导出工具"
    )
    parser.add_argument("--model", type=str, default=None,
                        help="Model name (e.g. yolo26s.pt, yolo26m.pt) / 模型名称 (如 yolo26s.pt, yolo26m.pt)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for export (cpu or 0) / 导出时使用的设备 (cpu 或 0)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark after export / 导出后运行基准测试")
    args = parser.parse_args()

    model_path = export_model(args.model, args.device)

    if args.benchmark:
        benchmark_model(model_path)
