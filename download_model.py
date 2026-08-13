"""
预导出模型下载脚本
====================
从 GitHub Release 下载预导出的 OpenVINO INT8 模型

解决 H-05: 模型分发问题
用户无需自行下载原始权重和执行 INT8 校准（耗时长且需联网），
可以直接下载预导出的 OpenVINO INT8 模型, 快速开始使用。

用法:
    python download_model.py                    # 下载默认模型 (yolo26s)
    python download_model.py --model yolo26m    # 下载指定模型
    python download_model.py --list             # 列出可用模型及其大小
    python download_model.py --force            # 覆盖已存在的模型

注意:
    Release 可能尚未创建, 下载失败时请参考错误提示,
    或使用 python export_model.py 手动导出模型。
"""

import argparse
import sys
import shutil
import zipfile
import urllib.request
import urllib.error
from pathlib import Path


# ============================================================
# 配置常量
# ============================================================

# GitHub 仓库信息
GITHUB_OWNER = "Ryoku-luke"
GITHUB_REPO = "dji-vision-system"

# Release 版本号 (预导出模型所在的 Release)
RELEASE_VERSION = "v1.0.0"

# 下载 URL 模板
# 格式: https://github.com/{owner}/{repo}/releases/download/{version}/{asset_name}
DOWNLOAD_URL_TEMPLATE = (
    "https://github.com/{owner}/{repo}/releases/download/{version}/{asset}"
)

# 项目根目录与导出目录
PROJECT_ROOT = Path(__file__).parent
EXPORTED_DIR = PROJECT_ROOT / "models" / "exported"

# 可用模型列表及其参考信息
# size_mb: INT8 量化后 OpenVINO 模型的预估大小 (含 .bin + .xml + metadata.yaml)
# inference_ms / fps: 基于 Intel Core Ultra 7 155H Arc GPU 的实测数据 (来自 README)
AVAILABLE_MODELS = {
    "yolo26n": {
        "description": "Nano 模型 - 速度优先, 精度要求低",
        "size_mb": 3.2,
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


# ============================================================
# ModelDownloader 类: 封装下载逻辑
# ============================================================

class ModelDownloader:
    """预导出 OpenVINO INT8 模型下载器"""

    def __init__(self, version: str = RELEASE_VERSION):
        """
        初始化下载器

        Args:
            version: Release 版本号 (如 "v1.0.0")
        """
        self.version = version
        self.exported_dir = EXPORTED_DIR

    def get_model_dir(self, model_name: str) -> Path:
        """
        获取模型的目标目录路径

        Args:
            model_name: 模型名称 (如 "yolo26s")

        Returns:
            模型目录路径 (如 models/exported/yolo26s_int8_openvino_model)
        """
        return self.exported_dir / f"{model_name}_int8_openvino_model"

    def get_download_url(self, model_name: str) -> str:
        """
        构建 Release 资产下载 URL

        Args:
            model_name: 模型名称 (如 "yolo26s")

        Returns:
            完整的下载 URL
        """
        asset_name = f"{model_name}_int8_openvino_model.zip"
        return DOWNLOAD_URL_TEMPLATE.format(
            owner=GITHUB_OWNER,
            repo=GITHUB_REPO,
            version=self.version,
            asset=asset_name,
        )

    @staticmethod
    def format_size(size_bytes: int) -> str:
        """
        将字节数格式化为人类可读的字符串

        Args:
            size_bytes: 文件大小 (字节)

        Returns:
            格式化后的大小字符串 (如 "10.3 MB")
        """
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
        """
        打印下载进度条 (不依赖 tqdm, 使用简单的 print 实现)

        Args:
            downloaded: 已下载字节数
            total: 总字节数 (为 0 表示无法获取总大小)
        """
        bar_length = 40

        if total <= 0:
            # 无法获取总大小时, 只显示已下载量
            print(f"\r  下载中... {ModelDownloader.format_size(downloaded)}", end="", flush=True)
            return

        percent = min(downloaded / total, 1.0)
        filled = int(bar_length * percent)
        bar = "=" * filled + "-" * (bar_length - filled)
        percent_str = f"{percent * 100:.1f}%"
        size_str = f"{ModelDownloader.format_size(downloaded)} / {ModelDownloader.format_size(total)}"
        print(f"\r  [{bar}] {percent_str} ({size_str})", end="", flush=True)

    def _check_existing(self, model_name: str, force: bool) -> bool:
        """
        检查目标路径是否已存在模型

        Args:
            model_name: 模型名称
            force: 是否强制覆盖

        Returns:
            True 表示可以继续下载, False 表示应中止
        """
        model_dir = self.get_model_dir(model_name)
        if not model_dir.exists():
            return True

        if not force:
            print(f"模型已存在: {model_dir}")
            print(f"如需重新下载, 请使用 --force 参数:")
            print(f"  python download_model.py --model {model_name} --force")
            return False

        print(f"已有模型将被覆盖: {model_dir}")
        return True

    def _download_file(self, url: str, dest: Path) -> int:
        """
        执行文件下载, 显示进度条

        Args:
            url: 下载 URL
            dest: 本地保存路径

        Returns:
            已下载的字节数, 失败时返回 -1

        Raises:
            urllib.error.HTTPError: HTTP 错误 (如 404)
            urllib.error.URLError: 网络错误
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

            print()  # 进度条换行

            if total_size > 0 and downloaded < total_size:
                print(f"警告: 下载可能不完整 ({downloaded}/{total_size} 字节)")

        return downloaded

    def _extract_model(self, zip_path: Path, model_name: str) -> Path:
        """
        解压模型压缩包到 exported 目录

        支持两种压缩包结构:
        1. 包含顶层目录: yolo26s_int8_openvino_model/yolo26s.bin ...
        2. 直接包含文件: yolo26s.bin, yolo26s.xml, metadata.yaml

        Args:
            zip_path: 压缩包路径
            model_name: 模型名称

        Returns:
            解压后的模型目录路径

        Raises:
            RuntimeError: 解压后未找到有效的模型文件
        """
        model_dir = self.get_model_dir(model_name)
        expected_dir_name = f"{model_name}_int8_openvino_model"

        # 解压到临时目录
        temp_extract = self.exported_dir / f"_{model_name}_extract_tmp"
        if temp_extract.exists():
            shutil.rmtree(temp_extract)

        with zipfile.ZipFile(zip_path, "r") as zf:
            namelist = zf.namelist()
            print(f"  压缩包包含 {len(namelist)} 个文件")
            zf.extractall(temp_extract)

        # 尝试定位解压后的模型目录
        # 情况1: 顶层有同名的模型目录
        candidate = temp_extract / expected_dir_name
        if candidate.is_dir():
            source_dir = candidate
        else:
            # 情况2: 遍历子目录, 找到包含 {model_name}.xml 的目录
            source_dir = None
            for item in temp_extract.rglob("*"):
                if item.is_dir() and (item / f"{model_name}.xml").exists():
                    source_dir = item
                    break
            if source_dir is None:
                # 情况3: 文件直接在临时目录根下
                if (temp_extract / f"{model_name}.xml").exists():
                    source_dir = temp_extract
                else:
                    shutil.rmtree(temp_extract)
                    raise RuntimeError(
                        f"解压后未找到模型文件 {model_name}.xml, "
                        f"请检查压缩包内容是否正确"
                    )

        # 移动到目标位置
        if model_dir.exists():
            shutil.rmtree(model_dir)

        # 如果 source_dir 是 temp_extract 本身 (情况3), 需要创建目标目录再移动文件
        if source_dir == temp_extract:
            model_dir.mkdir(parents=True, exist_ok=True)
            for item in source_dir.iterdir():
                shutil.move(str(item), str(model_dir / item.name))
        else:
            shutil.move(str(source_dir), str(model_dir))

        # 清理临时目录
        if temp_extract.exists():
            shutil.rmtree(temp_extract)

        return model_dir

    def _verify_model(self, model_dir: Path, model_name: str) -> bool:
        """
        验证模型文件完整性

        检查必要的 .xml 和 .bin 文件是否存在

        Args:
            model_dir: 模型目录
            model_name: 模型名称

        Returns:
            验证是否通过
        """
        required_files = [f"{model_name}.xml", f"{model_name}.bin"]
        missing = [f for f in required_files if not (model_dir / f).exists()]
        if missing:
            print(f"错误: 缺少必要的模型文件: {', '.join(missing)}")
            return False
        return True

    def download(self, model_name: str, force: bool = False) -> bool:
        """
        下载并解压模型

        Args:
            model_name: 模型名称 (如 "yolo26s")
            force: 是否覆盖已存在的模型

        Returns:
            下载是否成功
        """
        # 验证模型名称
        if model_name not in AVAILABLE_MODELS:
            print(f"错误: 不支持的模型 '{model_name}'")
            print(f"可用模型: {', '.join(AVAILABLE_MODELS.keys())}")
            return False

        # 检查目标路径是否已存在模型
        if not self._check_existing(model_name, force):
            # 模型已存在且未指定 --force, 视为成功 (无需重复下载)
            return True

        # 构建下载 URL
        url = self.get_download_url(model_name)
        info = AVAILABLE_MODELS[model_name]

        print("=" * 60)
        print(f"  下载预导出模型: {model_name}")
        print(f"  版本: {self.version}")
        print(f"  来源: GitHub Release")
        print(f"  描述: {info['description']}")
        print(f"  预估大小: {info['size_mb']:.1f} MB")
        print("=" * 60)

        # 准备下载目录
        self.exported_dir.mkdir(parents=True, exist_ok=True)
        temp_zip = self.exported_dir / f"_{model_name}_download_tmp.zip"

        try:
            # ---- 第 1 步: 下载 ----
            print(f"\n[1/3] 下载模型文件...")
            print(f"  URL: {url}")
            downloaded_bytes = self._download_file(url, temp_zip)
            print(f"  下载完成: {self.format_size(downloaded_bytes)}")

            # ---- 第 2 步: 解压 ----
            print(f"\n[2/3] 解压模型文件...")
            model_dir = self._extract_model(temp_zip, model_name)
            print(f"  解压完成: {model_dir}")

            # 列出解压后的文件
            for f in sorted(model_dir.iterdir()):
                if f.is_file():
                    size = f.stat().st_size
                    print(f"    - {f.name} ({self.format_size(size)})")

            # ---- 第 3 步: 验证 ----
            print(f"\n[3/3] 验证模型文件...")
            if not self._verify_model(model_dir, model_name):
                return False
            print(f"  验证通过: 所有必要文件均存在")

            # 计算实际大小
            total_size = sum(
                f.stat().st_size for f in model_dir.rglob("*") if f.is_file()
            )

            print(f"\n模型下载完成!")
            print(f"  路径: {model_dir}")
            print(f"  大小: {self.format_size(total_size)}")
            print(f"\n现在可以运行 main.py 启动推理:")
            print(f"  python main.py")

            return True

        except urllib.error.HTTPError as e:
            print(f"\n下载失败: HTTP {e.code} {e.reason}")
            if e.code == 404:
                print(f"\n模型文件未找到 (HTTP 404)。可能的原因:")
                print(f"  1. Release {self.version} 尚未创建")
                print(f"  2. {model_name} 的预导出模型尚未上传到 Release")
                print(f"  3. 模型名称拼写错误")
                print(f"\n请检查 GitHub Release 页面:")
                print(f"  https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases")
            elif e.code == 403:
                print(f"\n访问被拒绝 (HTTP 403)。可能是速率限制, 请稍后重试。")
            else:
                print(f"\n服务器返回错误状态码: {e.code}")

            self._print_alternative(model_name)
            return False

        except urllib.error.URLError as e:
            print(f"\n网络错误: {e.reason}")
            print(f"\n请检查网络连接后重试。")
            self._print_alternative(model_name)
            return False

        except zipfile.BadZipFile as e:
            print(f"\n压缩包损坏: {e}")
            print(f"下载的文件可能不完整或已损坏, 请重试。")
            self._print_alternative(model_name)
            return False

        except Exception as e:
            print(f"\n下载过程中发生未知错误: {e}")
            self._print_alternative(model_name)
            return False

        finally:
            # 清理临时下载文件
            if temp_zip.exists():
                temp_zip.unlink()
            # 清理可能残留的临时解压目录
            temp_extract = self.exported_dir / f"_{model_name}_extract_tmp"
            if temp_extract.exists():
                shutil.rmtree(temp_extract)

    @staticmethod
    def _print_alternative(model_name: str):
        """
        打印手动导出模型的替代方案提示

        Args:
            model_name: 模型名称
        """
        print(f"\n{'=' * 60}")
        print(f"  替代方案: 手动导出模型")
        print(f"{'=' * 60}")
        print(f"如果无法通过下载获取预导出模型, 可以手动导出:")
        print(f"")
        print(f"  python export_model.py --model {model_name}.pt")
        print(f"")
        print(f"注意: 手动导出 INT8 模型需要:")
        print(f"  1. 联网下载原始 PyTorch 权重")
        print(f"  2. 联网下载 COCO128 校准数据集")
        print(f"  3. 执行 INT8 量化校准 (耗时约 5-15 分钟)")
        print(f"{'=' * 60}")


# ============================================================
# list_available_models() 函数: 列出可用模型
# ============================================================

def list_available_models():
    """列出所有可用的预导出模型及其参考信息"""
    print("=" * 75)
    print("  可用的预导出 OpenVINO INT8 模型")
    print("=" * 75)
    print()
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
    print(f"  * 大小为预估值, 实际大小以下载为准")
    print(f"  * 推理速度基于 Intel Core Ultra 7 155H Arc GPU 实测")
    print()
    print(f"下载命令:")
    print(f"  python download_model.py --model <模型名称>")
    print()
    print(f"示例:")
    print(f"  python download_model.py                    # 下载默认模型 (yolo26s)")
    print(f"  python download_model.py --model yolo26m    # 下载 yolo26m")
    print(f"  python download_model.py --list             # 列出可用模型")


# ============================================================
# main() 函数: CLI 入口
# ============================================================

def main():
    """CLI 入口函数"""
    parser = argparse.ArgumentParser(
        description="预导出模型下载工具 - 从 GitHub Release 下载 OpenVINO INT8 模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python download_model.py                    # 下载默认模型 (yolo26s)
  python download_model.py --model yolo26m    # 下载指定模型
  python download_model.py --list             # 列出可用模型及其大小
  python download_model.py --force            # 覆盖已存在的模型

替代方案:
  如果下载失败 (如 Release 尚未创建), 可以手动导出模型:
  python export_model.py --model yolo26s.pt
""",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26s",
        choices=list(AVAILABLE_MODELS.keys()),
        help="模型名称 (默认: yolo26s)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出可用模型及其大小",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="覆盖已存在的模型",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=RELEASE_VERSION,
        help=f"Release 版本号 (默认: {RELEASE_VERSION})",
    )

    args = parser.parse_args()

    # 列出可用模型
    if args.list:
        list_available_models()
        return

    # 下载模型
    downloader = ModelDownloader(version=args.version)
    success = downloader.download(args.model, force=args.force)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
