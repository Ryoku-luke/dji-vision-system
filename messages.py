"""
Bilingual message definitions (zh / en).
Use t(key) to get the current language string.
"""

import config

def _lang():
    return config.LANGUAGE

_MSG = {
    # ---- system init ----
    "system_title": {"zh": "DJI 运动相机视觉识别系统", "en": "DJI Action Camera Vision System"},
    "init_model": {"zh": "加载推理模型", "en": "Loading inference model"},
    "init_camera": {"zh": "打开视频源", "en": "Opening video source"},
    "init_output": {"zh": "初始化视频输出", "en": "Initializing video output"},
    "init_done": {"zh": "系统初始化完成", "en": "System initialized"},
    "system_ready": {"zh": "系统已关闭", "en": "System shutdown"},
    "system_closing": {"zh": "正在关闭系统", "en": "Shutting down"},
    # ---- model ----
    "model_loaded": {"zh": "模型加载成功, 设备验证通过", "en": "Model loaded, device verified"},
    "model_not_found": {"zh": "模型文件不存在", "en": "Model file not found"},
    "model_warmup": {"zh": "推理引擎预热中 ({iterations} 轮)", "en": "Warming up inference engine ({iterations} rounds)"},
    "model_warmup_done": {"zh": "预热完成, 推理引擎就绪", "en": "Warmup complete, engine ready"},
    "model_not_loaded": {"zh": "模型未加载, 请先调用 load()", "en": "Model not loaded, call load() first"},
    "infer_failed": {"zh": "推理失败: {error}", "en": "Inference failed: {error}"},
    "infer_batch_failed": {"zh": "批量推理失败: {error}", "en": "Batch inference failed: {error}"},
    "device_verify_fail": {"zh": "设备 {device} 验证失败: {error}", "en": "Device {device} verification failed: {error}"},
    "device_fallback": {"zh": "回退到 {device}", "en": "Falling back to {device}"},
    "device_fallback_ok": {"zh": "已回退到 {device}", "en": "Fallback to {device} successful"},
    "device_fallback_fail": {"zh": "回退设备 {device} 验证也失败: {error}", "en": "Fallback device {device} also failed: {error}"},
    "no_fallback": {"zh": "无可用的回退设备", "en": "No fallback device available"},
    "cuda_unavailable": {"zh": "CUDA 不可用! 当前后端: {backend}", "en": "CUDA unavailable! Current backend: {backend}"},
    "torch_not_installed": {"zh": "PyTorch 未安装! CUDA/TensorRT 后端需要 PyTorch", "en": "PyTorch not installed! CUDA/TensorRT backend requires PyTorch"},
    "tensorrt_not_installed": {"zh": "TensorRT 未安装! TensorRT 后端需要 TensorRT 运行库", "en": "TensorRT not installed! TensorRT backend requires TensorRT runtime"},
    "openvino_not_installed": {"zh": "OpenVINO 未安装! OpenVINO 后端需要 openvino 包", "en": "OpenVINO not installed! OpenVINO backend requires openvino package"},
    # ---- camera ----
    "camera_opening": {"zh": "正在打开 UVC 摄像头 (设备索引: {index})", "en": "Opening UVC camera (index: {index})"},
    "camera_opened": {"zh": "摄像头已打开: {w}x{h} @ {fps}fps", "en": "Camera opened: {w}x{h} @ {fps}fps"},
    "camera_open_fail": {"zh": "无法打开摄像头设备 {index}", "en": "Cannot open camera device {index}"},
    "camera_not_open": {"zh": "摄像头未打开, 请先调用 open()", "en": "Camera not opened, call open() first"},
    "camera_closed": {"zh": "摄像头已关闭", "en": "Camera closed"},
    "camera_warmup": {"zh": "摄像头预热中 ({frames} 帧)", "en": "Camera warmup ({frames} frames)"},
    "camera_warmup_done": {"zh": "预热完成", "en": "Warmup complete"},
    "frame_read_fail": {"zh": "读取帧失败, 可能是摄像头断开连接", "en": "Frame read failed, camera may be disconnected"},
    "camera_dropped": {"zh": "摄像头连续丢帧, 可能已断开连接", "en": "Camera frame drops, may be disconnected"},
    "camera_devices": {"zh": "可用摄像头设备: {devices}", "en": "Available cameras: {devices}"},
    # ---- video file ----
    "video_not_found": {"zh": "视频文件不存在: {path}", "en": "Video file not found: {path}"},
    "video_opening": {"zh": "正在打开视频文件: {path}", "en": "Opening video file: {path}"},
    "video_open_fail": {"zh": "无法打开视频文件: {path}", "en": "Cannot open video file: {path}"},
    "video_opened": {"zh": "视频已打开: {w}x{h} @ {fps}fps, 共 {frames} 帧", "en": "Video opened: {w}x{h} @ {fps}fps, {frames} frames"},
    "video_finished": {"zh": "视频文件播放完毕", "en": "Video playback finished"},
    # ---- main loop ----
    "user_quit": {"zh": "用户退出", "en": "User quit"},
    "fps_reset": {"zh": "FPS 计数器已重置", "en": "FPS counter reset"},
    "interrupted": {"zh": "收到中断信号, 正在退出", "en": "Interrupted, exiting"},
    "frame_drop": {"zh": "丢帧 ({count}), 等待下一帧", "en": "Frame drop ({count}), waiting"},
    "stats_line": {"zh": "已处理 {count} 帧, 平均 {fps} FPS, 推理 {ms} ms/帧, 检测到 {objects} 个目标", "en": "{count} frames, avg {fps} FPS, {ms} ms/frame, {objects} objects"},
    "total_stats": {"zh": "总计: {count} 帧, {elapsed}s, 平均 {fps} FPS", "en": "Total: {count} frames, {elapsed}s, avg {fps} FPS"},
    "screenshot_saved": {"zh": "截图已保存: {path}", "en": "Screenshot saved: {path}"},
    "init_failed": {"zh": "系统初始化失败, 请检查错误信息", "en": "Initialization failed, check error messages"},
    "not_initialized": {"zh": "系统未初始化, 请先调用 initialize()", "en": "System not initialized, call initialize() first"},
    # ---- detection log ----
    "log_created": {"zh": "检测日志已创建: {path}", "en": "Detection log created: {path}"},
    "log_create_fail": {"zh": "无法创建检测日志: {error}", "en": "Cannot create detection log: {error}"},
    "log_closed": {"zh": "检测日志已关闭", "en": "Detection log closed"},
    # ---- class filter ----
    "class_id_oor": {"zh": "类别 ID {id} 超出范围 (0-{max}), 已忽略", "en": "Class ID {id} out of range (0-{max}), skipped"},
    "class_unknown": {"zh": "未知类别名称: {name}, 已忽略", "en": "Unknown class name: {name}, skipped"},
    # ---- video writer ----
    "video_out_created": {"zh": "视频输出: {path} ({w}x{h} @ {fps}fps)", "en": "Video output: {path} ({w}x{h} @ {fps}fps)"},
    "video_out_fail": {"zh": "无法创建视频文件: {path}", "en": "Cannot create video file: {path}"},
    "video_out_skip": {"zh": "视频输出初始化失败, 将忽略保存功能", "en": "Video output init failed, save disabled"},
    "log_init_skip": {"zh": "检测日志初始化失败, 将忽略日志记录", "en": "Detection log init failed, logging disabled"},
    # ---- resolution ----
    "res_mismatch": {"zh": "请求 {w}x{h}, 实际 {aw}x{ah} (部分相机不支持自定义分辨率)", "en": "Requested {w}x{h}, got {aw}x{ah} (some cameras ignore custom resolution)"},
}


def t(key: str, **kwargs) -> str:
    """Get a localized message string."""
    entry = _MSG.get(key)
    if entry is None:
        return key
    text = entry.get(_lang(), entry.get("zh", key))
    return text.format(**kwargs) if kwargs else text
