from gymnasium.envs.registration import register
from .game import Game, encoded_action
from .utils import move_to_coordinate

register(
    id="xqv0",
    entry_point="gymxq.xiangqi:XiangQiV0",
    max_episode_steps=300,
    # order_enforce=False,
    # disable_env_checker=True,
)

register(
    id="xqv1",
    entry_point="gymxq.xiangqi:XiangQiV1",
    max_episode_steps=300,
    # order_enforce=False,
    # disable_env_checker=True,
)

__all__ = ["Game", "encoded_action", "move_to_coordinate"]
