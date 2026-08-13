"""
Pre-exported model download script.
预导出模型下载脚本。

Download pre-exported OpenVINO INT8 models from a GitHub Release.
Users can skip downloading raw weights and running INT8 calibration
(time-consuming and requires network access) by fetching a ready-to-use
OpenVINO INT8 model directly.

Usage / 用法:
    python download_model.py                    # Download default model (yolo26s)
    python download_model.py --model yolo26m    # Download a specific model
    python download_model.py --list             # List available models and sizes
    python download_model.py --force            # Overwrite an existing model

Note / 注意:
    The Release may not exist yet. If download fails, follow the error
    message, or use `python export_model.py` to export the model manually.
"""

import argparse
import hashlib
import sys
import shutil
import zipfile
import urllib.request
import urllib.error
from pathlib import Path


# GitHub repo info / GitHub 仓库信息
GITHUB_OWNER = "Ryoku-luke"
GITHUB_REPO = "dji-vision-system"

# Release version (where pre-exported models are hosted) / Release 版本号
RELEASE_VERSION = "v1.0.0"

# Download URL template / 下载 URL 模板
DOWNLOAD_URL_TEMPLATE = (
    "https://github.com/{owner}/{repo}/releases/download/{version}/{asset}"
)

# Project root and exported dir / 项目根目录与导出目录
PROJECT_ROOT = Path(__file__).parent
EXPORTED_DIR = PROJECT_ROOT / "models" / "exported"

# Available models and reference info / 可用模型列表及其参考信息
# size_mb: estimated INT8 OpenVINO model size (.bin + .xml + metadata.yaml)
# inference_ms / fps: measured on Intel Core Ultra 7 155H Arc GPU (from README)
AVAILABLE_MODELS = {
    "yolo26n": {
        "description": "Nano 模型 - 速度优先, 精度要求低",
        "size_mb": 6.2,
        "inference_ms": 5.86,
        "fps": "~170",
    },
    "yolo26s": {
        "description": "Small 模型 - 推荐, 速度与精度平衡",
        "size_mb": 10.0,
        "inference_ms": 10.33,
        "fps": "~97",
    },
    "yolo26m": {
        "description": "Medium 模型 - 精度优先, 仍满足 30fps",
        "size_mb": 26.0,
        "inference_ms": 15.99,
        "fps": "~63",
    },
    "yolo26l": {
        "description": "Large 模型 - 高精度, 接近极限",
        "size_mb": 44.0,
        "inference_ms": 20.31,
        "fps": "~49",
    },
    "yolo26x": {
        "description": "Extra Large 模型 - 最高精度, 无法实时",
        "size_mb": 69.0,
        "inference_ms": 35.16,
        "fps": "~28",
    },
}

# Backend download configuration / 后端下载配置
# Defines what file to download and where to put it for each backend
BACKEND_CONFIG = {
    "openvino": {
        "asset_suffix": "_int8_openvino_model.zip",
        "target_suffix": "_int8_openvino_model",
        "needs_extract": True,
        "verify_ext": ".xml",
    },
    "cuda": {
        "asset_suffix": ".pt",
        "target_suffix": ".pt",
        "needs_extract": False,
        "verify_ext": ".pt",
    },
    # TensorRT engines are GPU-architecture-specific and cannot be distributed
    # TensorRT 引擎与 GPU 架构绑定, 无法预分发, 需用户本地导出
    "tensorrt": {
        "needs_export": True,
    },
}


class ModelDownloader:
    """Downloader for pre-exported models. 预导出模型下载器。

    Supports OpenVINO INT8 models (download .zip) and CUDA .pt models (download .pt).
    TensorRT engines are GPU-architecture-specific and must be exported locally.
    """

    def __init__(self, version: str = RELEASE_VERSION, backend: str = "openvino"):
        """Init the downloader.

        Args:
            version: Release version (e.g. "v1.0.0").
            backend: Inference backend ("openvino", "cuda", or "tensorrt").
        """
        self.version = version
        self.backend = backend
        if backend == "openvino":
            self.exported_dir = EXPORTED_DIR
        elif backend == "cuda":
            self.exported_dir = PROJECT_ROOT / "models"
        else:
            self.exported_dir = EXPORTED_DIR

    def get_model_dir(self, model_name: str) -> Path:
        """Return the target model directory/file path."""
        cfg = BACKEND_CONFIG.get(self.backend, BACKEND_CONFIG["openvino"])
        suffix = cfg.get("target_suffix", "_int8_openvino_model")
        target = self.exported_dir / f"{model_name}{suffix}"
        return target

    def get_download_url(self, model_name: str) -> str:
        """Build the Release asset download URL."""
        cfg = BACKEND_CONFIG.get(self.backend, BACKEND_CONFIG["openvino"])
        asset_name = f"{model_name}{cfg.get('asset_suffix', '_int8_openvino_model.zip')}"
        return DOWNLOAD_URL_TEMPLATE.format(
            owner=GITHUB_OWNER,
            repo=GITHUB_REPO,
            version=self.version,
            asset=asset_name,
        )

    @staticmethod
    def format_size(size_bytes: int) -> str:
        """Format a byte count into a human-readable string (e.g. "10.3 MB")."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / 1024 / 1024:.1f} MB"
        else:
            return f"{size_bytes / 1024 / 1024 / 1024:.2f} GB"

    @staticmethod
    def print_progress(downloaded: int, total: int):
        """Print a download progress bar (no tqdm dependency).

        Args:
            downloaded: bytes downloaded so far.
            total: total bytes (0 means unknown).
        """
        bar_length = 40

        if total <= 0:
            # Unknown total size: only show downloaded amount / 未知总大小时仅显示已下载量
            print(f"\r  Downloading / 下载中... {ModelDownloader.format_size(downloaded)}", end="", flush=True)
            return

        percent = min(downloaded / total, 1.0)
        filled = int(bar_length * percent)
        bar = "=" * filled + "-" * (bar_length - filled)
        percent_str = f"{percent * 100:.1f}%"
        size_str = f"{ModelDownloader.format_size(downloaded)} / {ModelDownloader.format_size(total)}"
        print(f"\r  [{bar}] {percent_str} ({size_str})", end="", flush=True)

    def _check_existing(self, model_name: str, force: bool) -> bool:
        """Check whether the target model already exists.

        Returns:
            True to continue downloading, False to abort.
        """
        model_dir = self.get_model_dir(model_name)
        if not model_dir.exists():
            return True

        if not force:
            print(f"Model already exists / 模型已存在: {model_dir}")
            print(f"To re-download, use the --force flag / 如需重新下载, 请使用 --force 参数:")
            print(f"  python download_model.py --model {model_name} --force")
            return False

        print(f"Existing model will be overwritten / 已有模型将被覆盖: {model_dir}")
        return True

    def _download_file(self, url: str, dest: Path) -> int:
        """Download a file with a progress bar.

        Returns:
            Bytes downloaded, or -1 on failure.

        Raises:
            urllib.error.HTTPError: HTTP error (e.g. 404).
            urllib.error.URLError: network error.
        """
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "dji-vision-system/1.0"},
        )

        downloaded = 0
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get("Content-Length", 0))
            chunk_size = 8192

            with open(dest, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    self.print_progress(downloaded, total_size)

            print()  # newline after progress bar / 进度条换行

            if total_size > 0 and downloaded < total_size:
                print(f"Warning: download may be incomplete ({downloaded}/{total_size} bytes)")
                print(f"警告: 下载可能不完整 ({downloaded}/{total_size} 字节)")

        return downloaded

    def _extract_model(self, zip_path: Path, model_name: str) -> Path:
        """Extract a model archive into the exported directory.

        Supports two archive layouts:
        1. With a top-level dir: yolo26s_int8_openvino_model/yolo26s.bin ...
        2. Flat files: yolo26s.bin, yolo26s.xml, metadata.yaml

        Raises:
            RuntimeError: no valid model file found after extraction.
        """
        model_dir = self.get_model_dir(model_name)
        expected_dir_name = f"{model_name}_int8_openvino_model"

        # Extract to a temp directory / 解压到临时目录
        temp_extract = self.exported_dir / f"_{model_name}_extract_tmp"
        if temp_extract.exists():
            shutil.rmtree(temp_extract)

        with zipfile.ZipFile(zip_path, "r") as zf:
            namelist = zf.namelist()
            print(f"  Archive contains {len(namelist)} files / 压缩包包含 {len(namelist)} 个文件")
            zf.extractall(temp_extract)

        # Locate the extracted model directory / 尝试定位解压后的模型目录
        # Case 1: top-level dir with the expected name
        candidate = temp_extract / expected_dir_name
        if candidate.is_dir():
            source_dir = candidate
        else:
            # Case 2: search subdirs for one containing {model_name}.xml
            source_dir = None
            for item in temp_extract.rglob("*"):
                if item.is_dir() and (item / f"{model_name}.xml").exists():
                    source_dir = item
                    break
            if source_dir is None:
                # Case 3: files are directly under the temp root
                if (temp_extract / f"{model_name}.xml").exists():
                    source_dir = temp_extract
                else:
                    shutil.rmtree(temp_extract)
                    raise RuntimeError(
                        f"Model file {model_name}.xml not found after extraction; "
                        f"please check the archive contents. "
                        f"/ 解压后未找到模型文件 {model_name}.xml, 请检查压缩包内容是否正确"
                    )

        # Move to the target location / 移动到目标位置
        if model_dir.exists():
            shutil.rmtree(model_dir)

        # If source_dir is temp_extract itself (case 3), create the target dir and move files in
        if source_dir == temp_extract:
            model_dir.mkdir(parents=True, exist_ok=True)
            for item in source_dir.iterdir():
                shutil.move(str(item), str(model_dir / item.name))
        else:
            shutil.move(str(source_dir), str(model_dir))

        # Clean up temp directory / 清理临时目录
        if temp_extract.exists():
            shutil.rmtree(temp_extract)

        return model_dir

    def _verify_model(self, model_dir: Path, model_name: str) -> bool:
        """Verify model file integrity based on backend type."""
        cfg = BACKEND_CONFIG.get(self.backend, BACKEND_CONFIG["openvino"])
        if self.backend == "cuda":
            # CUDA: verify .pt file exists
            pt_file = self.exported_dir / f"{model_name}.pt"
            if not pt_file.exists():
                print(f"Error: missing model file / 错误: 缺少模型文件: {pt_file.name}")
                return False
            return True
        else:
            # OpenVINO: verify .xml and .bin
            required_files = [f"{model_name}.xml", f"{model_name}.bin"]
            missing = [f for f in required_files if not (model_dir / f).exists()]
            if missing:
                print(f"Error: missing required model files / 错误: 缺少必要的模型文件: {', '.join(missing)}")
                return False
            return True

    @staticmethod
    def _compute_sha256(file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _verify_checksum(self, file_path: Path, model_name: str) -> bool:
        """Verify file integrity via SHA256 checksum.

        Downloads a checksums.txt file from the Release and verifies the
        downloaded file against the expected hash. If checksums.txt is not
        available, a warning is printed and verification is skipped.
        """
        checksum_url = DOWNLOAD_URL_TEMPLATE.format(
            owner=GITHUB_OWNER,
            repo=GITHUB_REPO,
            version=self.version,
            asset="checksums.txt",
        )
        try:
            req = urllib.request.Request(
                checksum_url,
                headers={"User-Agent": "dji-vision-system/1.0"},
            )
            with urllib.request.urlopen(req) as response:
                checksums_text = response.read().decode("utf-8")
        except urllib.error.HTTPError:
            print(f"  Warning: checksums.txt not found, skipping SHA256 verification")
            print(f"  警告: 未找到 checksums.txt, 跳过 SHA256 校验")
            return True

        # Parse checksums file (format: "<hash>  <filename>")
        expected_hash = None
        target_name = file_path.name
        for line in checksums_text.strip().split("\n"):
            parts = line.strip().split(None, 1)
            if len(parts) == 2 and parts[1] == target_name:
                expected_hash = parts[0]
                break

        if expected_hash is None:
            print(f"  Warning: {target_name} not found in checksums.txt, skipping verification")
            return True

        actual_hash = self._compute_sha256(file_path)
        if actual_hash != expected_hash:
            print(f"  Error: SHA256 mismatch!")
            print(f"    Expected: {expected_hash}")
            print(f"    Actual:   {actual_hash}")
            print(f"  The downloaded file may be corrupted, please retry with --force")
            return False

        print(f"  SHA256 verification passed / SHA256 校验通过")
        return True

    def download(self, model_name: str, force: bool = False) -> bool:
        """Download and extract a model.

        Args:
            model_name: model name (e.g. "yolo26s").
            force: overwrite an existing model.

        Returns:
            True on success.
        """
        # Validate model name / 验证模型名称
        if model_name not in AVAILABLE_MODELS:
            print(f"Error: unsupported model '{model_name}' / 错误: 不支持的模型 '{model_name}'")
            print(f"Available models / 可用模型: {', '.join(AVAILABLE_MODELS.keys())}")
            return False

        # TensorRT: engines are GPU-specific, cannot be pre-distributed
        if self.backend == "tensorrt":
            print("=" * 60)
            print(f"  TensorRT backend / TensorRT 后端")
            print("=" * 60)
            print(f"TensorRT engines are compiled for specific GPU architectures")
            print(f"and cannot be pre-distributed. Please export locally:")
            print(f"TensorRT 引擎与 GPU 架构绑定, 无法预分发, 请在本地导出:")
            print(f"")
            print(f"  python export_model.py --model {model_name}.pt")
            print(f"")
            print(f"Then set backend in config.py:")
            print(f"在 config.py 中设置后端:")
            print(f'  MODEL.backend = "tensorrt"')
            print(f"{'=' * 60}")
            return True

        # Check for an existing model / 检查目标路径是否已存在模型
        if not self._check_existing(model_name, force):
            return True

        # Build download URL / 构建下载 URL
        url = self.get_download_url(model_name)
        info = AVAILABLE_MODELS[model_name]
        cfg = BACKEND_CONFIG.get(self.backend, BACKEND_CONFIG["openvino"])
        needs_extract = cfg.get("needs_extract", True)

        print("=" * 60)
        print(f"  Download pre-exported model / 下载预导出模型: {model_name}")
        print(f"  Version / 版本: {self.version}")
        print(f"  Backend / 后端: {self.backend}")
        print(f"  Source / 来源: GitHub Release")
        print(f"  Description / 描述: {info['description']}")
        print(f"  Estimated size / 预估大小: {info['size_mb']:.1f} MB")
        print("=" * 60)

        # Prepare download directory / 准备下载目录
        self.exported_dir.mkdir(parents=True, exist_ok=True)

        # For zip downloads, use a temp file; for .pt, download directly
        if needs_extract:
            temp_file = self.exported_dir / f"_{model_name}_download_tmp.zip"
        else:
            temp_file = self.exported_dir / f"{model_name}{cfg.get('target_suffix', '.pt')}"

        try:
            # Step 1: download / 第 1 步: 下载
            print(f"\n[1/4] Downloading model file / 下载模型文件...")
            print(f"  URL: {url}")
            downloaded_bytes = self._download_file(url, temp_file)
            print(f"  Download complete / 下载完成: {self.format_size(downloaded_bytes)}")

            # Step 2: SHA256 verification / 第 2 步: SHA256 校验
            print(f"\n[2/4] Verifying SHA256 checksum / 校验 SHA256 哈希...")
            if not self._verify_checksum(temp_file, model_name):
                if temp_file.exists():
                    temp_file.unlink()
                return False

            # Step 3: extract or move / 第 3 步: 解压或移动
            if needs_extract:
                print(f"\n[3/4] Extracting model file / 解压模型文件...")
                model_dir = self._extract_model(temp_file, model_name)
                print(f"  Extraction complete / 解压完成: {model_dir}")

                for f in sorted(model_dir.iterdir()):
                    if f.is_file():
                        size = f.stat().st_size
                        print(f"    - {f.name} ({self.format_size(size)})")
            else:
                print(f"\n[3/4] Model file ready / 模型文件就绪: {temp_file}")
                model_dir = temp_file

            # Step 4: verify model files / 第 4 步: 验证模型文件
            print(f"\n[4/4] Verifying model files / 验证模型文件...")
            if not self._verify_model(model_dir if needs_extract else self.exported_dir, model_name):
                return False
            print(f"  Verification passed: all required files present / 验证通过: 所有必要文件均存在")

            # Compute actual size / 计算实际大小
            if needs_extract:
                total_size = sum(
                    f.stat().st_size for f in model_dir.rglob("*") if f.is_file()
                )
            else:
                total_size = temp_file.stat().st_size

            print(f"\nModel download complete! / 模型下载完成!")
            print(f"  Path / 路径: {model_dir}")
            print(f"  Size / 大小: {self.format_size(total_size)}")
            print(f"  Backend / 后端: {self.backend}")
            print(f"\nYou can now run main.py to start inference / 现在可以运行 main.py 启动推理:")
            print(f"  python main.py")

            return True

        except urllib.error.HTTPError as e:
            print(f"\nDownload failed: HTTP {e.code} {e.reason}")
            print(f"下载失败: HTTP {e.code} {e.reason}")
            if e.code == 404:
                print(f"\nModel file not found (HTTP 404). Possible reasons / 模型文件未找到 (HTTP 404)。可能的原因:")
                print(f"  1. Release {self.version} has not been created yet / Release {self.version} 尚未创建")
                print(f"  2. The pre-exported model for {model_name} ({self.backend}) has not been uploaded")
                print(f"     / {model_name} ({self.backend}) 的预导出模型尚未上传到 Release")
                print(f"  3. Misspelled model name / 模型名称拼写错误")
                print(f"\nPlease check the GitHub Release page / 请检查 GitHub Release 页面:")
                print(f"  https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases")
            elif e.code == 403:
                print(f"\nAccess denied (HTTP 403). This may be rate limiting, please retry later.")
                print(f"访问被拒绝 (HTTP 403)。可能是速率限制, 请稍后重试。")
            else:
                print(f"\nServer returned an error status code / 服务器返回错误状态码: {e.code}")

            self._print_alternative(model_name)
            return False

        except urllib.error.URLError as e:
            print(f"\nNetwork error / 网络错误: {e.reason}")
            print(f"\nPlease check your network connection and retry / 请检查网络连接后重试。")
            self._print_alternative(model_name)
            return False

        except zipfile.BadZipFile as e:
            print(f"\nCorrupt archive / 压缩包损坏: {e}")
            print(f"The downloaded file may be incomplete or corrupted, please retry / 下载的文件可能不完整或已损坏, 请重试。")
            self._print_alternative(model_name)
            return False

        except Exception as e:
            print(f"\nUnknown error during download / 下载过程中发生未知错误: {e}")
            self._print_alternative(model_name)
            return False

        finally:
            # Clean up temp download file (only for zip downloads) / 清理临时下载文件 (仅 zip 模式)
            if needs_extract:
                temp_file = self.exported_dir / f"_{model_name}_download_tmp.zip"
                if temp_file.exists():
                    temp_file.unlink()
            # Clean up any leftover temp extract dir / 清理可能残留的临时解压目录
            temp_extract = self.exported_dir / f"_{model_name}_extract_tmp"
            if temp_extract.exists():
                shutil.rmtree(temp_extract)

    @staticmethod
    def _print_alternative(model_name: str):
        """Print the manual-export fallback instructions."""
        print(f"\n{'=' * 60}")
        print(f"  Alternative: export the model manually / 替代方案: 手动导出模型")
        print(f"{'=' * 60}")
        print(f"If the pre-exported model cannot be downloaded, you can export it manually:")
        print(f"如果无法通过下载获取预导出模型, 可以手动导出:")
        print(f"")
        print(f"  python export_model.py --model {model_name}.pt")
        print(f"")
        print(f"Note: manual INT8 export requires / 注意: 手动导出 INT8 模型需要:")
        print(f"  1. Network access to download raw PyTorch weights / 联网下载原始 PyTorch 权重")
        print(f"  2. Network access to download the COCO128 calibration dataset / 联网下载 COCO128 校准数据集")
        print(f"  3. Running INT8 quantization calibration (~5-15 min) / 执行 INT8 量化校准 (耗时约 5-15 分钟)")
        print(f"{'=' * 60}")


def list_available_models():
    """List all available pre-exported models and their reference info."""
    print("=" * 75)
    print("  Available pre-exported models / 可用的预导出模型")
    print("=" * 75)
    print()
    print(f"  {'Model':<12} {'Size':<10} {'Infer(ms)':<12} {'FPS':<10} {'Description'}")
    print(f"  {'模型':<12} {'大小':<10} {'推理(ms)':<12} {'帧率':<10} {'描述'}")
    print(f"  {'-' * 12} {'-' * 10} {'-' * 12} {'-' * 10} {'-' * 28}")

    for name, info in AVAILABLE_MODELS.items():
        print(
            f"  {name:<12} "
            f"~{info['size_mb']:<8.1f} "
            f"{info['inference_ms']:<12.2f} "
            f"{info['fps']:<10} "
            f"{info['description']}"
        )

    print()
    print(f"  * Sizes are estimates; actual size depends on the download / 大小为预估值, 实际大小以下载为准")
    print(f"  * Inference speed measured on Intel Core Ultra 7 155H Arc GPU / 推理速度基于 Intel Core Ultra 7 155H Arc GPU 实测")
    print(f"  * CUDA backend downloads .pt models; OpenVINO downloads INT8 models / CUDA 后端下载 .pt 模型; OpenVINO 下载 INT8 模型")
    print(f"  * TensorRT engines must be exported locally (GPU-specific) / TensorRT 引擎需本地导出 (与 GPU 绑定)")
    print()
    print(f"Download command / 下载命令:")
    print(f"  python download_model.py --model <model_name> --backend <backend>")
    print()
    print(f"Examples / 示例:")
    print(f"  python download_model.py                                # OpenVINO yolo26s (default)")
    print(f"  python download_model.py --model yolo26m                # OpenVINO yolo26m")
    print(f"  python download_model.py --backend cuda                 # CUDA .pt model (yolo26s)")
    print(f"  python download_model.py --model yolo26n --backend cuda # CUDA yolo26n .pt")
    print(f"  python download_model.py --backend tensorrt             # TensorRT export guidance")
    print(f"  python download_model.py --list                         # List available models")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Pre-exported model downloader / 预导出模型下载工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples / 示例:
  python download_model.py                                # Download OpenVINO yolo26s (default)
  python download_model.py --model yolo26m                # Download a specific OpenVINO model
  python download_model.py --backend cuda                 # Download CUDA .pt model (yolo26s)
  python download_model.py --backend cuda --model yolo26n # Download CUDA yolo26n .pt
  python download_model.py --backend tensorrt             # Show TensorRT export guidance
  python download_model.py --list                         # List available models and sizes
  python download_model.py --force                        # Overwrite an existing model

Backends / 后端:
  openvino  Download INT8 OpenVINO model (.zip) - default
  cuda      Download PyTorch .pt model
  tensorrt  Show local export guidance (engines are GPU-specific)

Alternative / 替代方案:
  If the download fails (e.g. the Release does not exist yet), export the model manually:
  如果下载失败 (如 Release 尚未创建), 可以手动导出模型:
  python export_model.py --model yolo26s.pt
""",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26s",
        choices=list(AVAILABLE_MODELS.keys()),
        help="Model name (default: yolo26s) / 模型名称 (默认: yolo26s)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="openvino",
        choices=list(BACKEND_CONFIG.keys()),
        help="Inference backend: openvino (default), cuda, tensorrt / 推理后端",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and sizes / 列出可用模型及其大小",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing model / 覆盖已存在的模型",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=RELEASE_VERSION,
        help=f"Release version (default: {RELEASE_VERSION}) / Release 版本号 (默认: {RELEASE_VERSION})",
    )

    args = parser.parse_args()

    # List available models / 列出可用模型
    if args.list:
        list_available_models()
        return

    # Download the model / 下载模型
    downloader = ModelDownloader(version=args.version, backend=args.backend)
    success = downloader.download(args.model, force=args.force)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
