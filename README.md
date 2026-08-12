# DJI 运动相机视觉识别系统

基于 **DJI Osmo Action 运动相机** + **Intel Core Ultra 7 155H** + **OpenVINO + YOLO** 的实时目标检测系统。

## 系统架构

```
DJI Action 相机 (UVC 模式)
        │ Type-C USB
        ▼
┌───────────────────────────────────────────┐
│  Intel Core Ultra 7 155H 笔记本            │
│                                           │
│  ┌──────────┐   ┌──────────┐   ┌────────┐ │
│  │ OpenCV   │──▶│ YOLO26   │──▶│ 可视化  │ │
│  │ UVC取流  │   │ OpenVINO │   │ 输出    │ │
│  │          │   │ Arc GPU  │   │        │ │
│  └──────────┘   └──────────┘   └────────┘ │
│                                           │
│  32GB DDR5 | Intel Arc GPU | NPU         │
└───────────────────────────────────────────┘
```

## 环境要求

### 硬件

| 组件 | 要求 |
|------|------|
| 计算机 | Intel Core Ultra 7 155H (或其他 Intel CPU) |
| 内存 | 16GB+ (推荐 32GB) |
| 相机 | DJI Osmo Action 3/4/5 Pro/6 |
| 连接线 | USB Type-C 数据线 (USB 3.2 Gen1 以上) |

### 软件

| 组件 | 要求 |
|------|------|
| 操作系统 | Windows 11 (推荐) 或 Windows 10 |
| Python | 3.10 / 3.11 / 3.12 |
| 显卡驱动 | Intel Arc 显卡驱动 (最新版) |
| Git | 用于克隆仓库 |

## 环境搭建步骤

### 第一步：安装 Python

1. 访问 https://www.python.org/downloads/ 下载 Python 3.11 或 3.12
2. 安装时勾选 "Add Python to PATH"
3. 验证: 打开 PowerShell 或 CMD

```bash
python --version
# 输出应为: Python 3.11.x 或 3.12.x
```

### 第二步：安装 Intel 显卡驱动

1. 访问 https://www.intel.com/content/www/us/en/download-center/home.html
2. 搜索 "Intel Arc GPU driver" 并下载安装最新版
3. 重启电脑
4. 验证: 设备管理器 → 显示适配器 → 应显示 "Intel Arc Graphics"

### 第三步：创建项目环境

```bash
# 进入项目目录
cd dji-vision-system

# 创建虚拟环境 (推荐)
python -m venv venv

# 激活虚拟环境
# Windows PowerShell:
venv\Scripts\Activate.ps1
# Windows CMD:
venv\Scripts\activate.bat

# 升级 pip
python -m pip install --upgrade pip
```

### 第四步：安装依赖

```bash
# 安装所有依赖
pip install -r requirements.txt
```

如果安装 OpenVINO 较慢, 可使用国内镜像:

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 第五步：验证安装

```bash
# 验证 OpenVINO
python -c "from openvino import Core; print('OpenVINO 版本:', Core().get_versions())"

# 验证 Ultralytics
python -c "import ultralytics; print('Ultralytics 版本:', ultralytics.__version__)"

# 验证 OpenCV
python -c "import cv2; print('OpenCV 版本:', cv2.__version__)"

# 验证可用推理设备
python -c "
from openvino import Core
core = Core()
print('可用设备:', core.available_devices)
# 应输出类似: ['CPU', 'GPU']
"
```

### 第六步：连接相机

1. 用 Type-C 数据线将 DJI Osmo Action 连接到电脑
2. 相机屏幕上选择 **"网络摄像头"** 模式
3. 电脑会识别为 UVC 摄像头设备
4. 验证: 运行 `python main.py --list-cameras`

```bash
# 应输出类似:
# 可用摄像头设备: [0]
```

## 使用方法

### 1. 导出模型 (首次运行必须)

```bash
# 使用默认配置导出 (yolo26s + INT8)
python export_model.py

# 指定模型
python export_model.py --model yolo26m.pt

# 导出后运行基准测试
python export_model.py --benchmark
```

> 首次导出 INT8 模型会自动下载 COCO128 校准数据集, 需要联网, 耗时约 5-15 分钟。

### 2. 启动实时检测

```bash
# 默认启动 (Arc GPU 推理 + 实时显示)
python main.py

# 使用 CPU 推理 (兼容性最好)
python main.py --device intel:cpu

# 调整置信度阈值
python main.py --confidence 0.7

# 保存输出视频
python main.py --save

# 无显示 (性能测试)
python main.py --no-display
```

### 3. 键盘操作

| 按键 | 功能 |
|------|------|
| `q` | 退出 |
| `s` | 截图保存到 `output/screenshots/` |
| `r` | 重置 FPS 计数器 |

## 项目结构

```
dji-vision-system/
├── config.py              # 全局配置 (摄像头/模型/显示参数)
├── export_model.py        # 模型导出脚本 (PyTorch → OpenVINO INT8)
├── camera_capture.py      # UVC 摄像头采集模块
├── detector.py            # OpenVINO YOLO 推理引擎
├── visualizer.py          # 可视化模块 (绘制框/FPS/信息面板)
├── main.py                # 主程序入口
├── requirements.txt       # Python 依赖
├── README.md              # 本文件
├── models/                # 模型文件目录 (自动创建)
│   ├── yolo26s.pt         # 原始 PyTorch 模型
│   └── exported/          # 导出后的 OpenVINO 模型
│       └── yolo26s_int8_openvino_model/
└── output/                # 输出目录 (自动创建)
    ├── screenshots/       # 截图
    └── result.mp4         # 输出视频
```

## 配置说明

编辑 `config.py` 可调整所有参数:

### 摄像头配置

```python
@dataclass
class CameraConfig:
    device_index: int = 0        # 摄像头索引
    width: int = 1920            # 分辨率宽度
    height: int = 1080           # 分辨率高度
    fps: int = 30                # 帧率
```

### 模型配置

```python
@dataclass
class ModelConfig:
    model_name: str = "yolo26s.pt"   # 模型选择
    int8: bool = True                 # INT8 量化
    inference_device: str = "intel:gpu"  # 推理设备
    conf_threshold: float = 0.5       # 置信度阈值
```

### 模型选择参考 (155H 实测数据)

| 模型 | INT8 推理时间 | 对应 FPS | 适用场景 |
|------|:-----------:|:-------:|---------|
| yolo26n | 5.86 ms | ~170 fps | 速度优先, 精度要求低 |
| yolo26s | 10.33 ms | ~97 fps | **推荐, 速度与精度平衡** |
| yolo26m | 15.99 ms | ~63 fps | 精度优先, 仍满足 30fps |
| yolo26l | 20.31 ms | ~49 fps | 高精度, 接近极限 |
| yolo26x | 35.16 ms | ~28 fps | 最高精度, 无法实时 |

## 性能优化建议

### 1. 推理优化

- **使用 INT8 量化**: 比 FP32 快 2-3 倍, 精度损失约 1-3%
- **使用 Arc GPU**: `device="intel:gpu"` 比 CPU 快 6-15 倍
- **预热推理引擎**: 首次推理较慢 (需编译内核), 之后稳定

### 2. 取流优化

- **使用高质量 USB 线**: DJI 官方自带线材或 USB 3.2 Gen1 认证线
- **设置 buffer_size=1**: 最大限度降低取流延迟
- **避免 USB 扩展坞**: 直连主板 USB 口供电更稳定

### 3. 散热管理

- 长时间运行时垫高笔记本底部, 确保通风
- 可使用散热底座辅助降温
- 如频繁降频, 可在电源管理中设置"最佳性能"模式

## 常见问题

### Q: 提示 "无法打开摄像头设备"

1. 确认相机已通过 Type-C 连接电脑
2. 确认相机屏幕上选择了 "网络摄像头" 模式
3. 确认没有其他程序占用摄像头 (如 OBS、腾讯会议等)
4. 尝试更换 USB 接口

### Q: 提示 "模型文件不存在"

需要先导出模型: `python export_model.py`

### Q: GPU 设备不可用

1. 确认安装了 Intel Arc 显卡驱动
2. 确认 OpenVINO 版本 >= 2026.1
3. 运行 `python -c "from openvino import Core; print(Core().available_devices)"`
4. 如 GPU 不在列表中, 回退使用 CPU: `python main.py --device intel:cpu`

### Q: 推理速度比基准慢

1. 确认使用了 INT8 模型 (检查 models/exported/ 目录名包含 int8)
2. 确认设备为 intel:gpu (查看终端输出)
3. 运行 `python export_model.py --benchmark` 查看基准
4. 检查是否因散热问题导致降频

### Q: UVC 模式下画面卡顿

1. 使用 USB 3.0/3.2 接口 (不要用 USB 2.0)
2. 使用高质量数据线 (DJI 自带线材最佳)
3. 降低分辨率到 720P 测试

## 技术栈版本

| 组件 | 版本 |
|------|------|
| Python | 3.11+ |
| OpenVINO | 2026.1+ |
| Ultralytics | 8.4+ |
| OpenCV | 4.9+ |
| NumPy | 1.24+ |
