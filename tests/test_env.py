import pytest
import gymxq

import gymnasium
from gymxq.constants import *


def test_basic():
    env = gymnasium.make("gymxq/xqv0")
    obs, _ = env.reset()
    assert obs.shape == (SCREEN_HEIGHT, SCREEN_WIDTH, 3)
    # env.render()
    # action = env.game.move_string_to_action("4041")
    # env.step(action)
    # env.render()
