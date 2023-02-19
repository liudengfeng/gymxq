import time

import gymnasium
import numpy as np
import pytest
from gymnasium.wrappers.resize_observation import ResizeObservation
from PIL import Image

import gymxq
from gymxq import Game
from gymxq.constants import *


def test_basic_v0():
    env = gymnasium.make("gymxq/xqv0")
    assert env.metadata["max_episode_steps"] == 300
    obs, _ = env.reset()
    assert obs.shape == (SCREEN_HEIGHT, SCREEN_WIDTH, 3)


def test_basic_v1():
    env = gymnasium.make("gymxq/xqv1")
    assert env.metadata["max_episode_steps"] == 300
    obs, _ = env.reset()
    k = 1
    assert obs["s"].shape == (k * NUM_ROW * NUM_COL,)


# def test_view_qipu_v0():
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


def test_resize_v0():
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


def test_truncated():
    # 设置连续未吃子
    init_fen = "3ak1NrC/4a4/4b4/9/9/9/9/9/2p1r4/3K5 r - 0 - 298"
    actions = [Game.move_string_to_action(move) for move in ["8988", "7978"]]
    truncation = False
    env0 = gymnasium.make("gymxq/xqv0", init_fen=init_fen)
    env0.reset()
    for action in actions:
        _, _, _, truncation, _ = env0.step(action)
    assert truncation

    truncation = False
    env1 = gymnasium.make("gymxq/xqv1", init_fen=init_fen)
    env1.reset()
    for action in actions:
        _, _, _, truncation, _ = env1.step(action)
    assert truncation


def test_vector_v0():
    # 测试矢量环境
    n = 3
    envs = gymnasium.vector.make("gymxq/xqv0", gen_qp=False, num_envs=n)
    obs, _ = envs.reset()
    assert obs.shape == (n, SCREEN_HEIGHT, SCREEN_WIDTH, 3)
    actions = [Game.move_string_to_action(move) for move in ["4041", "1219", "1242"]]
    observations, rewards, termination, truncation, infos = envs.step(actions)
    assert observations.shape == (n, SCREEN_HEIGHT, SCREEN_WIDTH, 3)


def test_vector_v1():
    # 测试矢量环境
    n = 4
    k = 1
    init_fen = "3ak1NrC/4a4/4b4/9/9/9/9/9/2p1r4/3K5 r - 118 0 297"
    envs = gymnasium.vector.make(
        "gymxq/xqv1", init_fen=init_fen, gen_qp=False, num_envs=n
    )
    obs, _ = envs.reset()
    assert obs["s"].shape == (n, k * NUM_ROW * NUM_COL)

    # step 1
    actions = [
        Game.move_string_to_action(move) for move in ["6957", "8988", "8988", "6948"]
    ]
    observations, rewards, termination, truncation, infos = envs.step(actions)
    final_observation = infos["final_observation"]
    obs = final_observation[0]["s"]
    assert np.any(observations["s"][0] != obs)
    np.testing.assert_array_equal(rewards, np.array([1, 0, 0, 0]))
    np.testing.assert_array_equal(termination, np.array([True, False, False, False]))
    np.testing.assert_array_equal(truncation, np.array([False, False, False, False]))

    # step 2 连续未吃子
    actions = [
        Game.move_string_to_action(move) for move in ["6957", "2131", "7978", "7989"]
    ]
    observations, rewards, termination, truncation, infos = envs.step(actions)
    final_observation = infos["final_observation"]
    obs = final_observation[1]["s"]
    assert np.any(observations["s"][1] != obs)
    np.testing.assert_array_equal(rewards, np.array([1, -1, 0, 0]))
    np.testing.assert_array_equal(termination, np.array([True, True, True, False]))
    np.testing.assert_array_equal(truncation, np.array([False, False, False, False]))

    # step 3 最大步数
    actions = [
        Game.move_string_to_action(move) for move in ["6957", "2131", "7978", "4867"]
    ]
    observations, rewards, termination, truncation, infos = envs.step(actions)
    final_observation = infos["final_observation"]
    # obs = final_observation[3]["s"]
    # assert obs is None
    # 矢量化环境失败
    # np.testing.assert_array_equal(rewards, np.array([1, -1, 0, 0]))
    np.testing.assert_array_equal(rewards, np.array([1, -1, -1, 0]))
    np.testing.assert_array_equal(termination, np.array([True, True, True, False]))
    np.testing.assert_array_equal(truncation, np.array([False, False, False, True]))
