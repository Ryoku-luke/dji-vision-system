"""Tests for export_model.py, focusing on BUG-03 regression.
export_model.py 测试，聚焦 BUG-03 回归。

BUG-03 根本修复: 无论调用方的工作目录 (CWD) 在哪，YOLO 权重都必须下载到
target_dir 而非 CWD。这些测试验证修复行为并防止回归。
"""
# pylint: disable=import-outside-toplevel,redefined-outer-name,too-few-public-methods,unused-argument

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture
def isolated_models_dirs(tmp_path, monkeypatch):
    """Provide isolated models_dir / exported_dir under tmp_path.
    在 tmp_path 下提供隔离的 models_dir 和 exported_dir，避免污染真实项目目录。
    """
    from config import MODEL

    models_dir = tmp_path / "models"
    exported_dir = tmp_path / "exported"
    models_dir.mkdir()
    exported_dir.mkdir()

    monkeypatch.setattr(MODEL, "models_dir", models_dir)
    monkeypatch.setattr(MODEL, "exported_dir", exported_dir)
    return models_dir, exported_dir


def _make_fake_yolo_factory(captured: dict, model_name: str):
    """Build a fake YOLO constructor that simulates ultralytics download-to-CWD.

    构造一个伪造的 YOLO 构造函数，模拟 ultralytics "下载到 CWD/name" 的行为。
    """

    def _fake_yolo(name_or_path, *args, **kwargs):
        # ultralytics downloads weights to CWD / name when given a bare name.
        # 当传入纯文件名时，ultralytics 把权重下载到 CWD / name。
        captured["cwd_at_init"] = os.getcwd()
        Path(name_or_path).touch()
        mock = MagicMock()
        mock.export.return_value = str(Path("dummy_exported"))
        return mock

    return _fake_yolo


class TestDownloadYoloWeightsBug03:
    """BUG-03 回归: 权重下载必须对 CWD 鲁棒。"""

    def test_weights_land_in_target_dir_from_unrelated_cwd(self, isolated_models_dirs, monkeypatch):
        """TC-03b: 从无关目录运行时，权重必须落在 target_dir 而非 CWD。

        复现 BUG-03 场景: 用户从 /home/user 运行
        ``python /path/to/export_model.py``，修复前权重会丢失在 /home/user。
        """
        models_dir, _ = isolated_models_dirs
        # 模拟从无关目录运行 (如 /home/user)
        other_cwd = models_dir.parent / "other_cwd"
        other_cwd.mkdir()
        monkeypatch.chdir(other_cwd)

        name = "yolo26s.pt"
        captured: dict = {}

        with patch("ultralytics.YOLO", side_effect=_make_fake_yolo_factory(captured, name)):
            from export_model import _download_yolo_weights
            _yolo, model_path = _download_yolo_weights(name, models_dir)

        # 关键断言: 权重在 target_dir 而非 other_cwd
        assert model_path == models_dir / name
        assert model_path.exists()
        assert not (other_cwd / name).exists(), "权重不应遗留在调用方 CWD"
        # 调用 YOLO 时 CWD 已切换到 target_dir
        assert captured["cwd_at_init"] == str(models_dir)

    def test_cwd_restored_after_download(self, isolated_models_dirs, monkeypatch):
        """TC-03c: 下载完成后 CWD 必须恢复到原值 (try/finally 正确)。"""
        models_dir, _ = isolated_models_dirs
        original_cwd = models_dir.parent / "original_cwd"
        original_cwd.mkdir()
        monkeypatch.chdir(original_cwd)

        name = "yolo26s.pt"
        captured: dict = {}

        with patch("ultralytics.YOLO", side_effect=_make_fake_yolo_factory(captured, name)):
            from export_model import _download_yolo_weights
            _download_yolo_weights(name, models_dir)

        # CWD 已恢复，不会被切换污染
        assert os.getcwd() == str(original_cwd)

    def test_cwd_restored_even_on_exception(self, isolated_models_dirs, monkeypatch):
        """TC-03c-ext: YOLO 构造抛异常时 CWD 仍必须恢复。"""
        models_dir, _ = isolated_models_dirs
        original_cwd = models_dir.parent / "original_cwd"
        original_cwd.mkdir()
        monkeypatch.chdir(original_cwd)

        def _raising_yolo(name_or_path, *args, **kwargs):
            raise RuntimeError("simulated download failure")

        with patch("ultralytics.YOLO", side_effect=_raising_yolo):
            from export_model import _download_yolo_weights
            with pytest.raises(RuntimeError):
                _download_yolo_weights("yolo26s.pt", models_dir)

        assert os.getcwd() == str(original_cwd)

    def test_fallback_to_settings_dir(self, isolated_models_dirs, monkeypatch, tmp_path):
        """TC-03d: 兜底路径 - 权重在 ultralytics settings_dir 而非 CWD 时也能找到。

        某些 ultralytics 版本会下载到 SETTINGS['weights_dir'] 而非 CWD。
        """
        models_dir, _ = isolated_models_dirs
        other_cwd = tmp_path / "other"
        other_cwd.mkdir()
        monkeypatch.chdir(other_cwd)

        name = "yolo26s.pt"
        # 模拟 settings_dir 存放权重
        settings_dir = tmp_path / "ultralytics_weights"
        settings_dir.mkdir()
        (settings_dir / name).write_bytes(b"fake weights")

        def _fake_yolo(name_or_path, *args, **kwargs):
            # 权重未下载到 CWD，而是落在 settings_dir
            return MagicMock()

        fake_settings = {"weights_dir": str(settings_dir)}

        with patch("ultralytics.YOLO", side_effect=_fake_yolo), \
             patch("ultralytics.utils.SETTINGS", fake_settings):
            from export_model import _download_yolo_weights
            _yolo, model_path = _download_yolo_weights(name, models_dir)

        assert model_path.exists()
        assert model_path == models_dir / name
        # 原始 settings_dir 的文件已被移走
        assert not (settings_dir / name).exists()

    def test_raises_when_weights_not_found_anywhere(self, isolated_models_dirs, monkeypatch, tmp_path):
        """TC-03e: 权重既不在 CWD 也不在 settings_dir 时抛 FileNotFoundError。"""
        models_dir, _ = isolated_models_dirs
        other_cwd = tmp_path / "other"
        other_cwd.mkdir()
        monkeypatch.chdir(other_cwd)

        def _fake_yolo(name_or_path, *args, **kwargs):
            # 模拟下载未在 CWD 留下文件
            return MagicMock()

        fake_settings = {"weights_dir": str(tmp_path / "nonexistent")}

        with patch("ultralytics.YOLO", side_effect=_fake_yolo), \
             patch("ultralytics.utils.SETTINGS", fake_settings):
            from export_model import _download_yolo_weights
            with pytest.raises(FileNotFoundError, match="yolo26s.pt"):
                _download_yolo_weights("yolo26s.pt", models_dir)


class TestExportModelBug03Integration:
    """BUG-03 集成层: export_model() 调用 _download_yolo_weights 的衔接。"""

    def test_export_model_uses_target_dir_not_cwd(self, isolated_models_dirs, monkeypatch, tmp_path):
        """export_model() 在模型缺失时通过 _download_yolo_weights 下载到 models_dir。"""
        from config import MODEL
        models_dir, _exported_dir = isolated_models_dirs

        # 模拟从无关 CWD 运行
        other_cwd = tmp_path / "caller_cwd"
        other_cwd.mkdir()
        monkeypatch.chdir(other_cwd)

        name = MODEL.model_name  # 默认 yolo26s.pt

        # 让 export() 返回 exported_path，并提前创建该目录，跳过 move 逻辑
        target = MODEL.exported_path
        target.mkdir(parents=True, exist_ok=True)
        fake_yolo = MagicMock()
        fake_yolo.export.return_value = str(target)

        with patch("export_model._download_yolo_weights",
                   return_value=(fake_yolo, models_dir / name)) as mock_dl:
            from export_model import export_model
            export_model()

        # _download_yolo_weights 被调用，且 target_dir 是 models_dir
        mock_dl.assert_called_once()
        call_args = mock_dl.call_args
        assert call_args.args[0] == name
        assert call_args.args[1] == models_dir
