import math
import os
import time
from typing import Optional

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from gymxq.constants import *
from gymxq.envs.game import Game
from gymxq.utils import (
    get_center,
    get_piece_png_file,
    load_image,
    move_to_coordinate,
    render_board_to_text,
)


class XQEnvBase(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "max_episode_steps": MAX_EPISODE_STEPS,
        "render_fps": FPS,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        init_fen: Optional[str] = "",
        use_rule: bool = False,
        gen_qp: bool = False,
    ):
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.reward_range = (-1.0, 1.0)

        self.init_fen = init_fen
        self.use_rule = use_rule

        self._make_observation_space()
        # 是否生成棋谱
        self.gen_qp = gen_qp
        self.game = Game(self.init_fen, self.use_rule)
        self.window_surface = None
        """
        If human-rendering is used, `self.window_surface ` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.clock = None

        # 对局信息
        self._init_satistics_info()
        self.over_max_episode_steps = False

        self._init_cn_qipu()

        # 政策提示信息 [(k,v),(k,v),...]
        self.ai_pi_tip = []

    def _make_observation_space(self):
        raise NotImplementedError()

    def _init_satistics_info(self):
        # 对局信息
        self.satistics_info = {
            "total": 0,
            "win": 0,
            "draw": 0,
            "loss": 0,
            # 红方胜率
            "win_rate": 0.0,
            "draw": 0,
            # 黑方胜率
            "loss_rate": 0.0,
            "l": 0,
        }

    def _init_cn_qipu(self):
        self.cn_qipu = []
        if self.game.first_player == BLACK_PLAYER:
            self.cn_qipu.append(None)

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # episode steps
        self.satistics_info["l"] = 0
        self.over_max_episode_steps = False

        self._init_cn_qipu()

        self.ai_pi_tip = []

        # 重置统计信息
        if options and options["reset_satistics"]:
            self._init_satistics_info()

        self.game.reset()
        if self.render_mode != "ansi":
            self._render_gui(self.render_mode)
        observation = self._get_obs()
        return observation, self.satistics_info

    def _get_obs(self):
        raise NotImplementedError()

    def sample_action(self):
        """输出随机取样有效的action

        Returns:
            int: action
        """
        action = self.action_space.sample()
        while action not in self.game.legal_actions_history[-1]:
            action = self.action_space.sample()
        return action

    def set_ai_top(self, pi: dict):
        """设置AI提示信息

        Args:
            pi (dict): 政策字典
        """
        assert pi, "输入政策不得为空"
        tip = {}
        keys = list(pi.keys())
        # 转换为 move str
        if isinstance(keys[0], int):
            tip = {self.game.action_to_move_string(k): v for k, v in pi.items()}
        else:
            assert isinstance(keys[0], str), "输入政策键要么为整数、要么为代表移动的四位数字符串"
            tip = pi
        # 按值降序排列 -> [(k,v),(k,v)]
        # top 10
        self.ai_pi_tip = sorted(tip.items(), key=lambda x: x[1], reverse=True)[:10]

    def _update_info(self):
        self.satistics_info["total"] += 1
        reward, tip, reason = self.game.result()
        if reward == 1:
            self.satistics_info["win"] += 1
        elif reward == -1:
            self.satistics_info["loss"] += 1
        else:
            self.satistics_info["draw"] += 1
        self.satistics_info["win_rate"] = round(
            self.satistics_info["win"] / self.satistics_info["total"], 2
        )
        self.satistics_info["loss_rate"] = round(
            self.satistics_info["loss"] / self.satistics_info["total"], 2
        )

    def step(self, action):
        raise NotImplementedError()

    def render(self):
        if self.render_mode == "ansi":
            return render_board_to_text(self.game.board, self.last_move(), None)
        else:
            self._render_gui(self.render_mode)
        if self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )

    def last_move(self):
        actions = self.game.action_history
        if len(actions) >= 1:
            action = actions[-1]
            if action != -1:
                return self.game.action_to_move_string(action)
        return None

    def _init_load_images(self):
        # 加载图片
        self.icon_image = load_image("icon.ico")
        self.board_image = load_image("board540x600.png")
        self.cross_image = load_image("cross.png")
        self.frame_image = load_image("frame.png")
        self.piece_image_maps = {}
        for color_id in [1, 2]:
            self.piece_image_maps[color_id] = {}
            for piece_id in range(1, 8):
                self.piece_image_maps[color_id][piece_id] = load_image(
                    get_piece_png_file(color_id, piece_id)
                )

    def _draw_cell(self, image, x, y):
        # 绘制
        rect = image.get_rect()
        rect.center = get_center(x, y)
        self.window_surface.blit(image, rect)

    def _setting_text_render(self):
        """打印文本基础设置【字体、坐标】"""
        # self.head_font = pygame.font.SysFont("microsoftjhengheiui", 20)
        self.head_font = pygame.font.SysFont("Alibaba PuHuiTi 2.0", 18)
        # self.detail_font = pygame.font.SysFont("microsoftjhengheiui", 14)
        self.detail_font = pygame.font.SysFont("Alibaba PuHuiTi 2.0", 14)
        self.head_color = (21, 78, 210)
        self.detail_color = (75, 81, 93)
        self.w_s = FEATURE_WIDTH
        self.info_w = SCREEN_WIDTH - FEATURE_WIDTH
        self.wc1 = self.w_s + int(self.info_w / 6) * 1
        self.wc2 = self.w_s + int(self.info_w / 6) * 3
        self.wc3 = self.w_s + int(self.info_w / 6) * 5
        rows = [5, 12, 12]
        # 行高度
        self.info_h = int(SCREEN_HEIGHT / sum(rows))
        # 每个区域行开始坐标
        self.area_starts = [
            int(self.info_h / 2) + sum(rows[:i]) * self.info_h for i in range(len(rows))
        ]

    def _draw_text(self, info, center, is_head):
        color = self.head_color if is_head else self.detail_color
        font = self.head_font if is_head else self.detail_font
        text = font.render(info, True, color)
        rect = text.get_rect()
        rect.center = center
        self.window_surface.blit(text, rect)

    def _draw_info(self):
        # 比赛信息
        h_s = self.area_starts[0]
        h = self.info_h
        h_offset = 10
        # 统计信息
        self._draw_text(
            "红方{:>3d}%".format(int(self.satistics_info["win_rate"] * 100)),
            (self.wc1, h_s),
            True,
        )
        self._draw_text(
            "黑方{:>3d}%".format(int(self.satistics_info["loss_rate"] * 100)),
            (self.wc3, h_s),
            True,
        )
        w_info = "{:^5d}".format(self.satistics_info["win"])
        d_info = "{:^5d}".format(self.satistics_info["draw"])
        l_info = "{:^5d}".format(self.satistics_info["loss"])
        self._draw_text(w_info, (self.wc1, h_s + h * 1 + h_offset), False)
        self._draw_text(d_info, (self.wc2, h_s + h * 1 + h_offset), False)
        self._draw_text(l_info, (self.wc3, h_s + h * 1 + h_offset), False)
        # 胜负信息
        if not self.over_max_episode_steps:
            _, tip, reason = self.game.result()
        else:
            tip, reason = "平局", "步数超限({})判和".format(self.metadata["max_episode_steps"])
        tip = "{}({:3d}步)".format(tip, self.satistics_info["l"])
        self._draw_text(tip, (self.wc2, h_s + h * 2 + h_offset), True)
        self._draw_text(reason, (self.wc2, h_s + h * 3 + h_offset), False)

    def _draw_qipu(self):
        # 对局棋谱
        area_idx = 1
        h_s = self.area_starts[area_idx]
        h = self.info_h
        w_offset = 20
        h_offset = 10
        self._draw_text("棋谱", (self.w_s + w_offset, h_s), True)
        # 如果黑方先行，首先在棋谱中添加 None
        steps = math.ceil(len(self.cn_qipu) / 2)
        # 最多显示近20条
        to_select = 19 if len(self.cn_qipu) % 2 == 1 else 20
        to_view = self.cn_qipu[-to_select:]
        n = math.ceil(len(to_view) / 2)
        for i, qp in enumerate(to_view, 1):
            row = (i + 1) // 2
            hi = h_s + h * row + h_offset
            if i % 2 == 1:
                self._draw_text(
                    "{:>3d}:".format(steps - n + row),
                    (self.wc1, hi),
                    False,
                )
                red_qp = "" if qp is None else qp
                self._draw_text(red_qp, (self.wc2, hi), False)
            else:
                self._draw_text(qp, (self.wc3, hi), False)

    def _draw_ai_tip(self):
        if self.ai_pi_tip:
            # ai提示
            area_idx = 2
            h_s = self.area_starts[area_idx]
            h = self.info_h
            w_offset = 30
            h_offset = 10
            self._draw_text("AI提示", (self.w_s + w_offset, h_s), True)
            hi = h_s + h * 1 + h_offset
            self._draw_text("移动", (self.wc1, hi), False)
            self._draw_text("记谱", (self.wc2, hi), False)
            self._draw_text("分值", (self.wc3, hi), False)
            for i, (k, v) in enumerate(self.ai_pi_tip, 1):
                hi = h_s + h * (i + 1) + h_offset
                self._draw_text(k, (self.wc1, hi), False)
                cn_move = self.game.gen_qp(k)
                self._draw_text(cn_move, (self.wc2, hi), False)
                self._draw_text("{:.2f}".format(v), (self.wc3, hi), False)

    def _draw_pieces(self):
        # 绘制棋子
        pieces = self.game.board.get_pieces()
        for p in pieces:
            image = self.piece_image_maps[p.color_id][p.piece_id]
            self._draw_cell(image, p.x, p.y)

    def _mark_move(self):
        # 标记移动
        last_move = self.last_move()
        if last_move:
            x0, y0, x1, y1 = move_to_coordinate(last_move)
            # 标记移动结束坐标
            self._draw_cell(self.frame_image, x1, y1)
            # 标记移动开始坐标
            self._draw_cell(self.cross_image, x0, y0)

    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError as e:
            raise NotImplementedError(
                "pygame is not installed, run `pip install pygame`"
            )
        if self.window_surface is None:
            pygame.init()
            self._init_load_images()
            self._setting_text_render()
            if mode == "human":
                os.environ["SDL_VIDEO_CENTERED"] = "1"
                pygame.display.init()
                pygame.display.set_caption("中国象棋")
                self.window_surface = pygame.display.set_mode(
                    (SCREEN_WIDTH, SCREEN_HEIGHT)
                )
            elif mode == "rgb_array":
                self.window_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        if mode == "human":
            pygame.display.set_icon(self.icon_image)

        self.window_surface.fill(BACKGROUND)
        # 必须重新绘制棋盘
        self.window_surface.blit(self.board_image, (0, 0))
        # 绘制棋子
        self._draw_pieces()
        # 绘制移动标记
        self._mark_move()

        # 走子方
        player = "红方" if self.game.to_play() == 1 else "黑方"
        self._draw_text(player, (270, 300), True)

        # 对局信息
        self._draw_info()

        # 中文记谱
        if self.gen_qp:
            self._draw_qipu()

        # 绘制AI提示
        self._draw_ai_tip()

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window_surface is not None:
            pygame.display.quit()
            pygame.quit()
            self.window_surface = None


class XiangQiV0(XQEnvBase):
    """v0版本以单个RGB image作为观察对象"""

    def __init__(
        self,
        render_mode: Optional[str] = "rgb_array",
        init_fen: Optional[str] = "",
        gen_qp: bool = False,
    ):
        assert render_mode in [
            "human",
            "rgb_array",
        ], "render mode must in human,rgb_array"
        super().__init__(render_mode, init_fen, False, gen_qp)

    def _make_observation_space(self):
        self.observation_space = spaces.Box(
            0,
            255,
            (SCREEN_HEIGHT, SCREEN_WIDTH, 3),
            dtype=np.uint8,
        )

    def _get_obs(self):
        # self._render_gui(self.render_mode)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
        )

    def step(self, action):
        truncated = False
        _, reward, terminated = self.game.step(action)
        # 走后清空
        self.ai_pi_tip = []
        self.satistics_info["l"] += 1
        if (
            not terminated
            and self.satistics_info["l"] == self.metadata["max_episode_steps"]
        ):
            truncated = True
            reward = 0
            terminated = True
            self.over_max_episode_steps = True

        if self.gen_qp:
            qp = self.game.make_last_record()
            self.cn_qipu.append(qp)

        if terminated:
            self._update_info()

        self._render_gui(self.render_mode)

        observation = self._get_obs()

        return observation, reward, terminated, truncated, self.satistics_info


class XiangQiV1(XQEnvBase):
    """观察为编码对象"""

    def _make_observation_space(self):
        k = NUM_HISTORY if self.use_rule else 1
        self.observation_space = spaces.Dict(
            {
                # [a,b]
                "s": spaces.Box(
                    -NUM_PIECE,
                    NUM_PIECE,
                    (k * NUM_ROW * NUM_COL,),
                    dtype=np.int8,
                ),
                "a": spaces.Box(-1, NUM_ACTIONS - 1, (k,), dtype=np.int16),
                "continuous_uneaten": spaces.Discrete(MAX_NUM_NO_EAT, start=-1),
                "to_play": spaces.Discrete(NUM_PLAYER),
            }
        )

    def _get_obs(self):
        return self.game.reset()

    def step(self, action):
        truncated = False
        observation, reward, terminated = self.game.step(action)
        # 走后清空
        self.ai_pi_tip = []
        self.satistics_info["l"] += 1
        if (
            not terminated
            and self.satistics_info["l"] == self.metadata["max_episode_steps"]
        ):
            truncated = True
            reward = 0
            # terminated = True
            self.over_max_episode_steps = True

        if self.gen_qp:
            qp = self.game.make_last_record()
            self.cn_qipu.append(qp)

        if terminated:
            self._update_info()

        # if self.render_mode == "ansi":
        #     render_board_to_text(self.game.board, self.last_move(), None)
        if self.render_mode == "human":
            self._render_gui(self.render_mode)

        return observation, reward, terminated, truncated, self.satistics_info
