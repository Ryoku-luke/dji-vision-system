"""
main.py 工具函数单元测试
=========================
覆盖:
  - parse_classes() 类别过滤参数解析

  - 按名称解析: "person,car" -> [0, 2]
  - 按 ID 解析:   "0,2"       -> [0, 2]
  - 混合解析:     "person,2,truck" -> [0, 2, 7]
  - 空字符串 -> None
  - 未知类别 -> 忽略并返回有效结果
  - 超范围 ID -> 忽略

注意: 仅测试 parse_classes() 纯逻辑函数, 不启动视觉系统, 不依赖硬件。
导入 main 模块需要 cv2 / numpy (环境已具备, 或由 conftest 桩模块提供)。
"""

import pytest

from main import parse_classes


class TestParseClasses:
    """parse_classes() 函数测试"""

    @pytest.mark.parametrize(
        "class_str, expected",
        [
            # 按名称解析
            ("person,car", [0, 2]),
            # 按 ID 解析
            ("0,2", [0, 2]),
            # 混合解析: 名称 + ID + 名称
            ("person,2,truck", [0, 2, 7]),
        ],
    )
    def test_valid_inputs(self, class_str, expected):
        """测试合法输入的解析结果"""
        assert parse_classes(class_str) == expected

    @pytest.mark.parametrize(
        "class_str, expected",
        [
            ("", None),       # 空字符串
            (None, None),     # None
        ],
    )
    def test_empty_or_none(self, class_str, expected):
        """空字符串 / None 输入返回 None"""
        assert parse_classes(class_str) is None

    def test_unknown_class_ignored(self):
        """未知类别名称被忽略, 仍返回有效结果"""
        result = parse_classes("person,foobar,car")
        assert result == [0, 2]

    def test_out_of_range_id_ignored(self):
        """超范围 ID 被忽略 (COCO 仅 0-79)"""
        result = parse_classes("person,999,car")
        assert result == [0, 2]

    def test_all_invalid_returns_none(self):
        """全部为无效项时返回 None"""
        assert parse_classes("foobar,999,xyz") is None

    def test_whitespace_stripped(self):
        """输入项前后的空白被去除"""
        assert parse_classes(" person , car ") == [0, 2]

    def test_empty_items_skipped(self):
        """连续逗号产生的空项被跳过"""
        assert parse_classes("person,,car,") == [0, 2]

    def test_preserves_order_and_duplicates(self):
        """保留输入顺序与重复项"""
        assert parse_classes("car,person,car") == [2, 0, 2]

    def test_boundary_ids(self):
        """边界 ID (0 与 79) 均有效"""
        # COCO_CLASSES 共 80 类, 索引范围 0-79
        assert parse_classes("0,79") == [0, 79]

    def test_mixed_valid_invalid(self):
        """混合有效 / 无效 / 超范围项, 仅保留有效项"""
        result = parse_classes("person,999,2,unknown,7")
        assert result == [0, 2, 7]
