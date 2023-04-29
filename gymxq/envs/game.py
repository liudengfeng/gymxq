import os
import sys
from typing import List, Optional

import numpy as np
import cppxq
from ..constants import (
    NUM_ROW,
    NUM_COL,
    NUM_PIECE,
    NUM_HISTORY,
    MAX_NUM_NO_EAT,
    NUM_ACTIONS,
    NUM_PLAYER,
    RED_PLAYER,
    BLACK_PLAYER,
)
from ..utils import move_to_coordinate, make_last_move_qipu


def get_init_board(init_fen: Optional[str], use_rule: bool):
    board = cppxq.XqBoard()
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


class Game:
    """中国象棋游戏"""

    def __init__(self, init_fen: Optional[str], use_rule: bool):
        self.init_fen = init_fen
        self.use_rule = use_rule
        self._reset()

    def _reset(self):
        # 初始化列表
        self.to_play_id_history = []
        self.continuous_uneaten_history = []
        self.legal_actions_history = []

        self.pieces_history = []
        self.action_history = []
        self.reward_history = []

        self.board = get_init_board(self.init_fen, self.use_rule)
        # next_player 为整数 1 代表红方 2 代表黑方
        self.first_player = (
            RED_PLAYER if self.board.next_player() == RED_PLAYER else BLACK_PLAYER
        )
        self.player_id_ = self.board.next_player()

        self._append_for_next_batch()

        # 初始状态
        s0 = self.feature_pieces()
        self.pieces_history.append(s0)

        self.steps = self.board.steps() - 1

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
        return cppxq.a2m(action)

    @staticmethod
    def move_string_to_action(move: str) -> int:
        """移动字符串转换为移动编码

        Args:
            move (str): 4位整数代表的移动字符串

        Returns:
            int: 移动编码
        """
        return cppxq.m2a(move)

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
        if len(self.action_history) >= 1:
            qp = make_last_move_qipu(
                self.board, self.action_to_move_string(self.action_history[-1])
            )
            return qp
        return None

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
        if action not in self.legal_actions_history[-1]:
            legal_moves = [cppxq.a2m(a) for a in self.legal_actions_history[-1]]
            to_move = cppxq.a2m(action)
            raise RuntimeError("非法走子。合法移动={}，选中={}".format(legal_moves, to_move))

        self.board.move(action)
        self.steps += 1
        # 走子后更新done状态
        termination = self.board.is_finished()

        # 游戏结果以红方角度定义 [1：红胜, -1：红负, 0：平局]
        reward = self.board.reward() if termination else 0
        # 单代理始终以本方角度定义reward
        sign = 1 if self.player_id_ == RED_PLAYER else -1
        reward *= sign

        # 更新走子方
        self.player_id_ = self.board.next_player()

        self.action_history.append(action)
        self.reward_history.append(reward)
        self.pieces_history.append(self.feature_pieces())

        # next batch
        self._append_for_next_batch()

        s = self.pieces_history[-1]
        return (
            {
                "s": s,
                "steps": self.steps,
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
        # if self._illegal_move:
        #     reward = -1 if self.player_id_ == RED_PLAYER else 1
        #     tip = "红负" if self.player_id_ == RED_PLAYER else "红胜"
        #     reason = "红方非法走子" if self.player_id_ == RED_PLAYER else "黑方非法走子"
        # else:
        #     termination = self.board.is_finished()
        #     if termination:
        #         output = self.board.game_result_string().split("（")
        #         # 去除前后符号
        #         tip = output[1][:2]
        #         reason = output[1].split("[")[1][:-1]
        # 游戏并不限制步数，在环境中处理超限
        termination = self.board.is_finished()
        if termination:
            output = self.board.game_result_string().split("（")
            # 去除前后符号
            tip = output[1][:2]
            reason = output[1].split("[")[1][:-1]
        return (reward, tip, reason)

    def reset(self):
        self._reset()
        s = self.pieces_history[-1]
        return {
            "s": s,
            "steps": self.steps,
            "continuous_uneaten": self.continuous_uneaten_history[-1],
            "to_play": self.to_play_id_history[-1],
        }

    def to_play(self) -> int:
        return self.player_id_

    def feature_pieces(self):
        """棋子特征编码【负数代表黑方棋子】

        Returns:
            ndarray: shape = (10*9,) 数组
        """
        return np.array(self.board.get2d(), dtype=np.int8).ravel()
