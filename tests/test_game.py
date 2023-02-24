import numpy as np
import pytest
import xqcpp

from gymxq.game import Game, encoded_action
from gymxq.utils import move_to_coordinate
from gymxq.constants import *


def _check_action_feature(f, move):
    assert np.count_nonzero(f) == 2
    x0, y0, x1, y1 = move_to_coordinate(move, True)
    assert f[0, y0, x0] == 1
    assert f[1, y1, x1] == 1


def test_encoded_action():
    move = "2324"
    lr_move = xqcpp.move2lr(move)
    action = xqcpp.movestr2action(move)
    f1 = encoded_action(action)
    f2 = encoded_action(action, True)
    _check_action_feature(f1, move)
    _check_action_feature(f2, lr_move)


@pytest.mark.parametrize(
    "init_fen,expected",
    [
        (
            "3ak1NrC/9/4b4/9/2n1c1P2/9/3R5/9/2p6/2BK1A3 r - - 0 1",
            [
                [0, 0, 3, 7, 0, 2, 0, 0, 0],
                [0, 0, -1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 6, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, -5, 0, -4, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, -3, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, -2, -7, 0, 5, -6, 4],
            ],
        ),
    ],
)
def test_feature_pieces(init_fen, expected):
    g = Game(init_fen, False)
    # g.board.show_board()
    actual = g.feature_pieces()
    np.testing.assert_array_equal(np.array(expected).flatten(), actual)


@pytest.mark.parametrize(
    "init_fen,expected_to_play_id,expected_no_eat_num,expected_action_num",
    [
        ("3akar2/9/9/9/9/9/2C6/4C4/4R4/4K4 r - 0 0 1", 1, 0, 41),
        ("3akar2/9/9/9/9/9/2C6/4C4/4R4/4K4 b - 0 0 1", 2, 0, 12),
        ("3ak1NrC/4a4/4b4/9/9/9/9/9/2p1r4/3K5 b - 110 0 1", 2, 110, 23),
    ],
)
def test_init_game(
    init_fen,
    expected_to_play_id,
    expected_no_eat_num,
    expected_action_num,
):
    g = Game(init_fen, False)
    to_check_fields = [
        "to_play_id_history",
        "continuous_uneaten_history",
        "legal_actions_history",
    ]
    for f in to_check_fields:
        # 添加初始值
        assert len(getattr(g, f)) == 1
    assert g.to_play_id_history[-1] == expected_to_play_id
    assert g.continuous_uneaten_history[-1] == expected_no_eat_num
    assert len(g.legal_actions_history[-1]) == expected_action_num


@pytest.mark.parametrize(
    "use_rule,expected_len",
    [(False, 1), (True, 1)],
)
def test_stacked_feature(use_rule, expected_len):
    g = Game("", use_rule)
    # 定义对象本身为空
    assert len(g) == 0

    assert len(g.action_history) == expected_len
    # 填充
    assert all([a is NUM_ACTIONS for a in g.action_history])

    assert len(g.pieces_history) == expected_len

    s0 = g.feature_pieces()
    np.testing.assert_array_equal(g.pieces_history[-1], s0)

    # 测试 index = -1
    s = g.pieces_history[len(g.reward_history)]
    a = g.action_history[-1]
    assert s.dtype == np.int8 and s.shape == (expected_len * NUM_ROW * NUM_COL,)
    # pre action
    assert a is NUM_ACTIONS

    # g.board.show_board()

    l = expected_len
    # 确保没有吃子
    moves = [
        "1242",
        "1927",
        "1022",
        "2625",
        "7276",
        "2715",
        "7062",
        "6947",
        "2324",
        "4645",
        "4344",
        "6665",
    ]
    for i, move in enumerate(moves, 1):
        ma = g.move_string_to_action(move)
        obs, reward, done = g.step(ma)
        assert not done
        assert reward == 0
        s = obs["s"]
        a = obs["a"]
        uneaten = obs["continuous_uneaten"]
        to_play = obs["to_play"]
        assert to_play == RED_PLAYER if i % 2 == 0 else BLACK_PLAYER
        assert uneaten == i
        # g.board.show_board()

        assert s.dtype == np.int8 and s.shape == (l * NUM_ROW * NUM_COL,)
        np.testing.assert_array_equal(g.pieces_history[-1], g.feature_pieces())

        # assert a.dtype == np.int16 and a.shape == (l,)
        # assert a[-1].item() == g.move_string_to_action(moves[i - 1])
        assert a == g.move_string_to_action(moves[i - 1])

        if use_rule:
            if i < l:
                assert all(x == -1 for x in a[:-i])
                # s0 ... st
                filled = s[: -(i + 1) * NUM_ROW * NUM_COL].reshape(
                    (-1, NUM_ROW, NUM_COL)
                )
                assert filled.shape[0] == l - i - 1
                assert np.count_nonzero(filled) == 0
            else:
                assert np.count_nonzero(s) == 2 * 16 * l


def test_step1():
    # 红方非法走子判负
    init_fen = "3ak1NrC/4a4/4b4/9/9/9/9/9/2p1r4/3K5 r - 118 0 297"
    g = Game(init_fen, False)
    obs = g.reset()
    action = Game.move_string_to_action("0001")
    s, r, t = g.step(action)
    for k in obs.keys():
        np.testing.assert_array_equal(obs[k], s[k])
    assert r == -1
    assert t


def test_step2():
    # 游戏已经结束时，继续执行移动不会影响最终结果
    init_fen = "3ak1NrC/4a4/4b4/9/9/9/9/9/2p1r4/3K5 r - 118 0 297"
    g = Game(init_fen, False)
    obs = g.reset()
    action = Game.move_string_to_action("8988")
    s1, r1, t1 = g.step(action)
    assert r1 == 0
    assert not t1
    action = Game.move_string_to_action("7978")
    s2, r2, t2 = g.step(action)
    assert (s1["s"] != s2["s"]).any()
    assert r2 == 0
    # 连续未吃子判和
    assert t2
    assert g._reward == 0
    # 非法走子不成立，因为游戏已经结束
    action = Game.move_string_to_action("7978")
    s3, r3, t3 = g.step(action)
    assert (s2["s"] == s3["s"]).all()
    assert r3 == 0
    assert t3
