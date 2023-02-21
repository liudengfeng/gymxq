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
    obs, info = env.reset()
    assert obs.shape == (SCREEN_HEIGHT, SCREEN_WIDTH, 3)
    assert info["to_play"] == 1
    assert len(info["legal_actions"]) == 44


def test_basic_v1():
    env = gymnasium.make("gymxq/xqv1")
    assert env.metadata["max_episode_steps"] == 300
    obs, info = env.reset()
    k = 1
    assert obs["s"].shape == (k * NUM_ROW * NUM_COL,)
    assert info["to_play"] == 1
    assert len(info["legal_actions"]) == 44


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
    # 注意 步数从1开始，实际步数为298
    init_fen = "3ak1NrC/4a4/4b4/9/9/9/9/9/2p1r4/3K5 r - 0 - 299"
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
    assert len(infos["to_play"]) == n
    assert len(infos["legal_actions"]) == n


def test_vector_v1_1():
    # 测试矢量环境
    n = 1
    k = 1
    init_fen = "3ak1NrC/4a4/4b4/9/9/9/9/9/2p1r4/3K5 r - 118 0 297"
    envs = gymnasium.vector.make(
        "gymxq/xqv1", init_fen=init_fen, gen_qp=False, num_envs=n
    )
    obs0, _ = envs.reset()
    assert obs0["s"].shape == (n, k * NUM_ROW * NUM_COL)

    key = "final_observation"
    # step 1
    actions = [Game.move_string_to_action(move) for move in ["8988"]]
    obs1, r1, t1, tr1, i1 = envs.step(actions)
    assert key not in i1.keys()
    np.testing.assert_array_equal(r1, np.array([0]))
    np.testing.assert_array_equal(t1, np.array([False]))
    np.testing.assert_array_equal(tr1, np.array([False]))

    # step 2 连续未吃子
    actions = [Game.move_string_to_action(move) for move in ["7978"]]
    obs2, r2, t2, tr2, i2 = envs.step(actions)
    assert key in i2.keys()
    o2 = i2[key][0]["s"]
    # print("\n", np.flipud(o2.reshape(10, 9)))
    # print("\n", np.flipud(obs2["s"][0].reshape(10, 9)))
    # print("\n", np.flipud(obs0["s"][0].reshape(10, 9)))
    # print(i2["legal_actions"])
    # 输出的状态为初始状态
    for key in obs2.keys():
        np.testing.assert_array_equal(obs2[key], obs0[key])
    # 最终状态与输出状态不同
    assert np.any(obs2["s"][0] != o2)

    np.testing.assert_array_equal(r2, np.array([0]))
    # 连续120着未吃子判和，游戏结束
    np.testing.assert_array_equal(t2, np.array([True]))
    np.testing.assert_array_equal(tr2, np.array([False]))

    # step 3 最大步数
    actions = [Game.move_string_to_action(move) for move in ["7978"]]
    # 自动reset恢复至初始状态
    _, r3, t3, tr3, i3 = envs.step(actions)
    f3 = i3["final_observation"][0]
    for k in f3.keys():
        np.testing.assert_array_equal(f3[k], obs0[k].ravel())

    # 矢量化环境重置后视同重新开始新的棋局，非法移动
    np.testing.assert_array_equal(r3, np.array([-1]))
    np.testing.assert_array_equal(t3, np.array([True]))
    # 重置后不会触及最大步数
    np.testing.assert_array_equal(tr3, np.array([False]))


def test_vector_v1_2():
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
    _, rewards, termination, truncation, infos = envs.step(actions)
    np.testing.assert_array_equal(rewards, np.array([1, -1, -1, 0]))
    np.testing.assert_array_equal(termination, np.array([True, True, True, False]))
    np.testing.assert_array_equal(truncation, np.array([False, False, False, False]))
