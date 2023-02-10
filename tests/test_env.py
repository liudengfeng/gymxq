import pytest
import gymxq
import time
import gymnasium
from gymxq.constants import *


def test_basic():
    env = gymnasium.make("gymxq/xqv0")
    obs, _ = env.reset()
    assert obs.shape == (SCREEN_HEIGHT, SCREEN_WIDTH, 3)


def test_view_qipu():
    env = gymnasium.make("gymxq/xqv0", gen_qp=True)
    obs, _ = env.reset()
    while True:
        action = env.sample_action()
        observation, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(1)
        if terminated or truncated:
            break
