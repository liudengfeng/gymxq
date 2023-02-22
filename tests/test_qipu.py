import pytest

from gymxq.constants import *
from gymxq.game import Game
from gymxq.utils import front_and_back_tag, make_last_move_qipu


@pytest.mark.parametrize(
    "color_id,ys,y,expected",
    [
        (RED_PLAYER, [3, 6], 6, "前"),
        (RED_PLAYER, [3, 6], 3, "后"),
        (BLACK_PLAYER, [4, 8], 4, "前"),
        (BLACK_PLAYER, [4, 8], 8, "后"),
    ],
)
def test_move_direction_tag_2(color_id, ys, y, expected):
    # 二项重叠
    actual = front_and_back_tag(color_id, ys, y)
    assert expected == actual


@pytest.mark.parametrize(
    "color_id,ys,y,expected",
    [
        (RED_PLAYER, [5, 6, 7, 8], 8, "前１"),
        (RED_PLAYER, [5, 6, 7, 8], 7, "前２"),
        (RED_PLAYER, [5, 6, 7, 8], 6, "前３"),
        (RED_PLAYER, [5, 6, 7, 8], 5, "前４"),
        (RED_PLAYER, [8, 6, 5, 7], 5, "前４"),
        (BLACK_PLAYER, [6, 2, 4, 8], 8, "前四"),
        (BLACK_PLAYER, [6, 2, 4, 8], 2, "前一"),
    ],
)
def test_move_direction_tag_4(color_id, ys, y, expected):
    # 3+
    actual = front_and_back_tag(color_id, ys, y)
    assert expected == actual


@pytest.mark.parametrize(
    "color_id,ys,y,expected",
    [
        (RED_PLAYER, [5, 7, 9], 9, "前"),
        (RED_PLAYER, [5, 7, 9], 7, "中"),
        (RED_PLAYER, [5, 7, 9], 5, "后"),
        (BLACK_PLAYER, [2, 4, 8], 2, "前"),
        (BLACK_PLAYER, [2, 4, 8], 4, "中"),
        (BLACK_PLAYER, [2, 4, 8], 8, "后"),
    ],
)
def test_move_direction_tag_3(color_id, ys, y, expected):
    # 三项重叠
    actual = front_and_back_tag(color_id, ys, y)
    assert expected == actual


@pytest.mark.parametrize(
    "init_fen,moves_list,expected_list",
    [
        (
            "1Pb2aN1C/5k3/1P3a3/7P1/1Pb2c1P1/1rp2c3/1rp3p2/2p6/6p2/3K5 b - 110 0 1",
            [
                ["5848", "3040"],
                ["2947", "3031"],
                ["2947", "8979"],
                ["5565", "7666"],
                ["5464", "7585"],
                ["5450", "7677"],
                ["2333", "1929"],
            ],
            [
                ["将６平５", "帅六平五"],
                ["象３进５", "帅六进一"],
                ["象３进５", "炮一平二"],
                ["后砲平７", "前二平三"],
                ["前砲平７", "后二平一"],
                ["前砲进４", "前二进一"],
                ["中３平４", "前八平七"],
            ],
        ),
    ],
)
def test_cn_qipu(init_fen, moves_list, expected_list):
    use_rule = False
    for i, moves in enumerate(moves_list):
        g = Game(init_fen, use_rule)
        # g.board.show_board()
        expected = expected_list[i]
        actual = []
        for move in moves:
            g.board.do_move_str(move)
            # g.board.show_board(True, move)
            actual.append(make_last_move_qipu(g.board, move))
        assert actual == expected


@pytest.mark.parametrize(
    "init_fen,moves_list,expected_list",
    [
        (
            "3ak1NrC/4a1P2/4b1P2/2p3PP1/6P2/2p6/2p6/2p6/2p1r4/3K5 r - - 0 1",
            [
                ["6757", "2434"],
                ["7677", "2120"],
                ["6575", "2625"],
            ],
            [
                ["前２平四", "前四平４"],
                ["兵二进一", "前一进１"],
                ["前４平二", "前五进１"],
            ],
        ),
    ],
)
def test_cn_qipu_2(init_fen, moves_list, expected_list):
    use_rule = False
    for i, moves in enumerate(moves_list):
        g = Game(init_fen, use_rule)
        # g.board.show_board()
        expected = expected_list[i]
        actual = []
        for move in moves:
            g.board.do_move_str(move)
            # g.board.show_board(True, move)
            actual.append(make_last_move_qipu(g.board, move))
        assert actual == expected


@pytest.mark.parametrize(
    "init_fen,moved_list,not_move,expected",
    [
        ("3akar2/9/9/9/9/9/2C6/4C4/4R4/4K4 r - 0 0 1", [], "2343", "炮七平五"),
        ("3akar2/9/9/9/9/9/2C6/4C4/4R4/4K4 r - 0 0 1", ["2329"], "4948", "将５进１"),
    ],
)
def test_gen_qp(init_fen, moved_list, not_move, expected):
    # 测试尚未执行的移动中文记谱
    use_rule = False
    g = Game(init_fen, use_rule)
    for move in moved_list:
        g.board.do_move_str(move)
    actual = g.gen_qp(not_move)
    assert actual == expected
