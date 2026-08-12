"""
模型导出脚本
====================
将 YOLO PyTorch 模型导出为 OpenVINO INT8 格式
首次运行需要下载模型权重并进行 INT8 校准量化

用法:
    python export_model.py                    # 使用 config.py 默认配置导出
    python export_model.py --model yolo26m    # 指定模型
    python export_model.py --device cpu       # 导出时使用 CPU (无 GPU 时)
"""

import argparse
import sys
import shutil
import logging
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config import MODEL

logger = logging.getLogger(__name__)


def export_model(model_name: str = None, export_device: str = "cpu"):
    """
    导出 YOLO 模型为 OpenVINO 格式

    Args:
        model_name: 模型文件名 (如 "yolo26s.pt"), 为 None 时使用 config 默认值
        export_device: 导出时使用的设备 ("cpu" 或 "0" 表示 GPU)
    """
    from ultralytics import YOLO

    # 确定模型名称
    name = model_name or MODEL.model_name
    model_path = MODEL.models_dir / name

    print("=" * 60)
    print(f"  模型导出: {name}")
    print(f"  目标格式: OpenVINO ({'INT8' if MODEL.int8 else 'FP16' if MODEL.half else 'FP32'})")
    print(f"  输入尺寸: {MODEL.imgsz}x{MODEL.imgsz}")
    print("=" * 60)

    # 检查模型文件是否存在, 不存在则自动下载
    if not model_path.exists():
        print(f"\n[1/3] 下载模型: {name}")
        MODEL.models_dir.mkdir(parents=True, exist_ok=True)
        # YOLO 构造时会自动下载到当前工作目录
        yolo = YOLO(name)
        # 移动到 models 目录 (下载路径相对于 CWD)
        downloaded = Path(name)
        if downloaded.exists():
            shutil.move(str(downloaded), str(model_path))
        else:
            # 某些版本可能下载到 ~/.config/ultralytics 或其他位置
            logger.warning(f"未在当前目录找到下载的模型文件 {name}, 请检查模型路径")
        print(f"      模型已保存到: {model_path}")
    else:
        print(f"\n[1/3] 模型已存在: {model_path}")
        yolo = YOLO(str(model_path))

    # 构建导出参数
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
        print(f"\n[2/3] 导出 OpenVINO INT8 (校准数据: {MODEL.calib_data})")
        print(f"      校准比例: {MODEL.calib_fraction*100:.0f}%")
        print(f"      注意: 首次 INT8 导出需要下载校准数据集, 请耐心等待...")
    elif MODEL.half:
        export_kwargs["half"] = True
        print(f"\n[2/3] 导出 OpenVINO FP16")
    else:
        print(f"\n[2/3] 导出 OpenVINO FP32")

    # 执行导出
    exported_path = yolo.export(**export_kwargs)
    exported_path = Path(exported_path)
    print(f"      导出完成: {exported_path}")

    # 重命名到目标目录
    MODEL.exported_dir.mkdir(parents=True, exist_ok=True)
    target = MODEL.exported_path

    if exported_path != target:
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(exported_path), str(target))

    print(f"\n[3/3] 模型已就绪: {target}")
    print("\n导出完成! 现在可以运行 main.py 启动推理:")
    print(f"  python main.py")

    # 打印模型大小信息
    if target.is_dir():
        total_size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
        print(f"\n  模型大小: {total_size / 1024 / 1024:.1f} MB")

    return target


def benchmark_model(model_path: Path = None):
    """
    对导出的模型进行基准测试

    Args:
        model_path: OpenVINO 模型路径, 为 None 时使用 config 默认值
    """
    from ultralytics import YOLO

    path = model_path or MODEL.exported_path
    print(f"\n{'=' * 60}")
    print(f"  基准测试: {path.name}")
    print(f"  设备: {MODEL.inference_device}")
    print(f"{'=' * 60}\n")

    model = YOLO(str(path))

    # 运行基准测试
    results = model.benchmark(
        data="coco128.yaml",
        imgsz=MODEL.imgsz,
        device=MODEL.inference_device,
        verbose=True,
    )

    print(f"\n基准测试结果:")
    # Ultralytics benchmark 返回的键名兼容多种版本
    inference_time = results.get('speed/inference') or results.get('inference_time')
    map_val = results.get('metrics/mAP50-95(B)') or results.get('mAP50-95(B)')
    if inference_time is not None:
        print(f"  推理速度: {inference_time:.2f} ms/im")
        fps = 1000.0 / inference_time if inference_time > 0 else 0.0
        print(f"  对应帧率: {fps:.1f} FPS")
    else:
        print(f"  推理速度: N/A")
        print(f"  对应帧率: N/A")
    print(f"  mAP50-95: {map_val if map_val is not None else 'N/A'}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO 模型导出工具")
    parser.add_argument("--model", type=str, default=None,
                        help="模型名称 (如 yolo26s.pt, yolo26m.pt)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="导出时使用的设备 (cpu 或 0)")
    parser.add_argument("--benchmark", action="store_true",
                        help="导出后运行基准测试")
    args = parser.parse_args()

    model_path = export_model(args.model, args.device)

    if args.benchmark:
        benchmark_model(model_path)
