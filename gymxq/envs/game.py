import os
import sys
from typing import List, Optional

import numpy as np
import xqcpp
from gymxq.constants import (
    NUM_ROW,
    NUM_COL,
    NUM_PIECE,
    NUM_HISTORY,
    MAX_NUM_NO_EAT,
    NUM_PLAYER,
    RED_PLAYER,
    BLACK_PLAYER,
)
from gymxq.utils import move_to_coordinate, make_last_move_qipu


def get_init_board(init_fen: Optional[str], use_rule: bool):
    board = xqcpp.XqBoard()
    board.reset()
    # 设置移动类型判断规则
    board.set_use_rule_flag(use_rule)
    include_no_eat_ = False
    if init_fen:
        fs = init_fen.split(" ")
        assert len(fs) <= 6, "空格分隔列表数量必须小于6"
        if len(fs) == 6 and fs[3] != "-":
            include_no_eat_ = True
        board.init_set(init_fen, True, include_no_eat_)
    return board


def game_feature_shape(use_rule: bool):
    """游戏特征shape

    Args:
        use_rule (bool): 是否使用规则

    Returns:
        tuple: (int,int,int)
    """
    if use_rule:
        # S_1, A_1,S_2, A_2 ... S_18 [缺A_18]
        return (
            (NUM_HISTORY - 1) * (NUM_PIECE * NUM_PLAYER + NUM_PLAYER)
            + NUM_PIECE * NUM_PLAYER
            + 3,
            NUM_ROW,
            NUM_COL,
        )
    else:
        # S_1
        return (NUM_PIECE * NUM_PLAYER + 3, NUM_ROW, NUM_COL)


def encoded_action(action: int, lr: bool = False):
    """编码移动序号

    Args:
        action (int): 移动序号

    Returns:
        ndarray: np.ndarray(2,10,9)
    """
    res = np.zeros((2, NUM_ROW, NUM_COL), dtype=np.uint8)
    if action == -1:
        return res
    else:
        move = xqcpp.action2movestr(action)
        if lr:
            move = xqcpp.move2lr(move)
        x0, y0, x1, y1 = move_to_coordinate(move, True)
        res[0][y0][x0] = 1
        res[1][y1][x1] = 1
        return res


class Game:
    """中国象棋游戏"""

    def __init__(self, init_fen: Optional[str], use_rule: bool):
        self.init_fen = init_fen
        self.use_rule = use_rule
        # 使用GameHistory表达堆积
        # self.k = NUM_HISTORY if self.use_rule else 1
        self.k = 1

        self._reset()

    def _reset(self):
        # 初始化列表
        self._reward = 2
        self._illegal_move = False  # 用于指示非法走子
        self.to_play_id_history = []
        self.continuous_uneaten_history = []
        # self.to_eat_history = []
        self.legal_actions_history = []

        self.pieces_history = []
        self.action_history = []
        self.reward_history = []

        # TODO:暂时保留
        self.child_visits = []
        self.root_values = []

        self.board = get_init_board(self.init_fen, self.use_rule)
        # next_player 为整数 1 代表红方 2 代表黑方
        self.first_player = (
            RED_PLAYER if self.board.next_player() == RED_PLAYER else BLACK_PLAYER
        )
        self.player_id_ = self.board.next_player()

        self._append_for_next_batch()

        # 中国象棋需要观察历史18着
        # 设置不需要检查重复时，只使用当前状态，不堆积历史
        piece_filled = np.zeros(NUM_ROW * NUM_COL, dtype=np.uint8)
        # 历史 S A
        # T - 1
        for _ in range(self.k - 1):
            self.pieces_history.append(piece_filled)
            # 注意填充`-1`
            self.action_history.append(-1)

        self.action_history.append(-1)
        # 初始状态
        s0 = self.feature_pieces()
        self.pieces_history.append(s0)

    def __len__(self):
        return len(self.reward_history)

    @staticmethod
    def action_to_move_string(action: int) -> str:
        """移动编码转换为4位整数代表的移动字符串

        Args:
            action (int): 移动编码 [0,2085]

        Returns:
            str: 4位整数代表的移动字符串
        """
        return xqcpp.action2movestr(action)

    @staticmethod
    def move_string_to_action(move: str) -> int:
        """移动字符串转换为移动编码

        Args:
            move (str): 4位整数代表的移动字符串

        Returns:
            int: 移动编码
        """
        return xqcpp.movestr2action(move)

    def get_fen(self):
        """棋盘状态fen字符串

        Returns:
            str: fen字符串
        """
        return self.board.get_fen()

    def make_last_record(self):
        """中文记谱

        Returns:
            str: 移动中文记谱
        """
        if self._illegal_move:
            return "非法走子"
        if len(self.action_history) >= 1 and self.action_history[-1] != -1:
            qp = make_last_move_qipu(
                self.board, self.action_to_move_string(self.action_history[-1])
            )
            return qp

    def gen_qp(self, not_move: str):
        """生成尚未执行的移动棋谱

        Args:
            not_move (str): 四位整数代表的移动字符串

        Returns:
            str: 中文棋谱
        """
        b0 = self.board.clone()
        # b0.show_board()
        if len(self) >= 1:
            b0.back_one_step()
        # 简化计算量
        b0.set_use_rule_flag(False)
        b0.do_move_str(not_move)
        return make_last_move_qipu(b0, not_move)

    def _append_for_next_batch(self):
        # next batch
        self.continuous_uneaten_history.append(self.board.no_eat())
        self.to_play_id_history.append(self.player_id_)
        self.legal_actions_history.append(self.board.legal_actions())

    def step(self, action):
        # 吸收状态
        if self._reward != 2:
            s, a = self.get_stacked_feature(len(self.reward_history))
            return (
                {
                    "s": s,
                    "a": a,
                    "continuous_uneaten": self.continuous_uneaten_history[-1],
                    "to_play": self.to_play_id_history[-1],
                },
                self._reward,
                True,
            )
        # 非法走子
        if self._illegal_move or (action not in self.legal_actions_history[-1]):
            self._illegal_move = True
            termination = True
            reward = -1 if self.player_id_ == RED_PLAYER else 1
            self._reward = reward
            s, a = self.get_stacked_feature(len(self.reward_history))
            # self.board.show_board(True, "非法走子{}".format(self.action_to_move_string(action)))
            return (
                {
                    "s": s,
                    "a": a,
                    "continuous_uneaten": self.continuous_uneaten_history[-1],
                    "to_play": self.to_play_id_history[-1],
                },
                reward,
                termination,
            )

        self.board.move(action)
        # 走子后更新done状态
        termination = self.board.is_finished()
        # 棋盘假设红方先行，其结果以红方角度定义 [1：红胜, -1：红负, 0：平局]
        reward = self.board.reward() if termination else 0
        if termination:
            self._reward = reward
        # reward始终以当前走子方角度来修正，即当前走子方胜得分1，负得分-1，否则为0
        # Final outcomes {lose, draw, win} in board games are treated as reward_history ut ∈ {−1, 0, +1}
        # reward *= 1 if self.player_id_ == RED_PLAYER else -1
        # assert reward != -1, "不可能存在自杀移动"

        # 更新走子方，务必在符号修正之后
        self.player_id_ = self.board.next_player()

        self.action_history.append(action)
        self.reward_history.append(reward)
        self.pieces_history.append(self.feature_pieces())

        # next batch
        self._append_for_next_batch()
        s, a = self.get_stacked_feature(len(self.reward_history))
        return (
            {
                "s": s,
                "a": a,
                "continuous_uneaten": self.continuous_uneaten_history[-1],
                "to_play": self.to_play_id_history[-1],
            },
            reward,
            termination,
        )

    def result(self):
        tip = ""
        reason = ""
        reward = 0
        # 红方角度定义 [1：红胜, -1：红负, 0：平局]
        if self._illegal_move:
            reward = -1 if self.player_id_ == RED_PLAYER else 1
            tip = "红负" if self.player_id_ == RED_PLAYER else "红胜"
            reason = "红方非法走子" if self.player_id_ == RED_PLAYER else "黑方非法走子"
        else:
            termination = self.board.is_finished()
            if termination:
                output = self.board.game_result_string().split("（")
                # 去除前后符号
                tip = output[1][:2]
                reason = output[1].split("[")[1][:-1]
        return (reward, tip, reason)

    def reset(self):
        self._reset()
        s, a = self.get_stacked_feature(0)
        return {
            "s": s,
            "a": a,
            "continuous_uneaten": self.continuous_uneaten_history[-1],
            "to_play": self.to_play_id_history[-1],
        }

    def get_stacked_feature(self, idx: int = 0):
        """获取指定序号的编码特征

        Args:
            idx (int, optional): 序号. Defaults to 0.

        Returns:
            ndarray: 特征编码数组
        """
        max_idx = len(self.reward_history)
        if idx == -1:
            return self.get_stacked_feature(max_idx)
        assert idx >= 0 and idx <= max_idx, "idx有效范围{}~{},无效输入{}".format(
            0, max_idx, idx
        )
        s = self.pieces_history[idx : idx + self.k]
        a = self.action_history[idx : idx + self.k]
        return np.concatenate(s, dtype=np.int8), np.array(a, dtype=np.int16)

    def to_play(self) -> int:
        return self.player_id_

    def feature_pieces(self):
        """棋子特征编码【负数代表黑方棋子】

        Returns:
            ndarray: (10, 9)数组
        """
        return np.array(self.board.get2d(), dtype=np.int8).flatten()
