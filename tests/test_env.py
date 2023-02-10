import pytest
import gymxq
from gymxq import Game
import time
import gymnasium
from gymxq.constants import *
from gymnasium.wrappers.resize_observation import ResizeObservation
from PIL import Image


def test_basic():
    env = gymnasium.make("gymxq/xqv0")
    obs, _ = env.reset()
    assert obs.shape == (SCREEN_HEIGHT, SCREEN_WIDTH, 3)


# def test_view_qipu():
#     env = gymnasium.make("gymxq/xqv0", gen_qp=True)
#     obs, _ = env.reset()
#     while True:
#         action = env.sample_action()
#         observation, reward, terminated, truncated, info = env.step(action)
#         env.render()
#         # time.sleep(1)
#         if terminated or truncated:
#             # time.sleep(3)
#             break


def test_resize():
    env = gymnasium.make("gymxq/xqv0", gen_qp=False)
    H, W = 300, 270
    env = ResizeObservation(env, (H, W))
    obs, _ = env.reset()
    assert obs.shape == (H, W, 3)
    action = env.sample_action()
    observation, reward, terminated, truncated, info = env.step(action)
    assert observation.shape == (H, W, 3)
    # image = Image.fromarray(observation)
    # image.show()
    # time.sleep(3)


def test_vector():
    # 测试矢量环境
    n = 3
    envs = gymnasium.vector.make("gymxq/xqv0", gen_qp=False, num_envs=n)
    obs, _ = envs.reset()
    assert obs.shape == (n, SCREEN_HEIGHT, SCREEN_WIDTH, 3)
    actions = [Game.move_string_to_action(move) for move in ["4041", "1219", "1242"]]
    observations, rewards, termination, truncation, infos = envs.step(actions)
    assert observations.shape == (n, SCREEN_HEIGHT, SCREEN_WIDTH, 3)
