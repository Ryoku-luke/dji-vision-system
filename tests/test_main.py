"""Unit tests for main.py — parse_classes() argument parsing."""

import pytest

from main import parse_classes


class TestParseClasses:
    """Tests for parse_classes() / parse_classes() 函数测试"""

    @pytest.mark.parametrize(
        "class_str, expected",
        [
            # By name / 按名称解析
            ("person,car", [0, 2]),
            # By ID / 按 ID 解析
            ("0,2", [0, 2]),
            # Mixed: name + ID + name / 混合解析: 名称 + ID + 名称
            ("person,2,truck", [0, 2, 7]),
        ],
    )
    def test_valid_inputs(self, class_str, expected):
        """Test valid input parsing / 测试合法输入的解析结果"""
        assert parse_classes(class_str) == expected

    @pytest.mark.parametrize(
        "class_str, expected",
        [
            ("", None),       # empty string / 空字符串
            (None, None),     # None
        ],
    )
    def test_empty_or_none(self, class_str, expected):
        """Empty string / None input returns None / 空字符串 / None 返回 None"""
        assert parse_classes(class_str) is None

    def test_unknown_class_ignored(self):
        """Unknown class names are ignored, valid results still returned."""
        result = parse_classes("person,foobar,car")
        assert result == [0, 2]

    def test_out_of_range_id_ignored(self):
        """Out-of-range IDs are ignored (COCO is 0-79)."""
        result = parse_classes("person,999,car")
        assert result == [0, 2]

    def test_all_invalid_returns_none(self):
        """All invalid items return None / 全部为无效项时返回 None"""
        assert parse_classes("foobar,999,xyz") is None

    def test_whitespace_stripped(self):
        """Leading/trailing whitespace is stripped / 输入项前后空白被去除"""
        assert parse_classes(" person , car ") == [0, 2]

    def test_empty_items_skipped(self):
        """Empty items from consecutive commas are skipped / 连续逗号产生的空项被跳过"""
        assert parse_classes("person,,car,") == [0, 2]

    def test_duplicates_deduplicated_without_order_change(self):
        """Duplicate IDs/names are deduplicated, first occurrence order is preserved (BUG-7).
        重复 ID/名称被去重，保留首次出现顺序 (BUG-7)。"""
        # "car,person,car" => deduplicated to [2, 0] (order of first occurrence)
        # "person,person,0,2" => person and 0 both resolve to ID 0, so [0, 2]
        assert parse_classes("car,person,car") == [2, 0]
        assert parse_classes("person,person,0,2") == [0, 2]

    def test_boundary_ids(self):
        """Boundary IDs (0 and 79) are valid / 边界 ID (0 与 79) 均有效"""
        # COCO_CLASSES has 80 classes, indices 0-79
        assert parse_classes("0,79") == [0, 79]

    def test_mixed_valid_invalid(self):
        """Mixed valid/invalid/out-of-range items keep only valid ones."""
        result = parse_classes("person,999,2,unknown,7")
        assert result == [0, 2, 7]
