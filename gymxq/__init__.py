from gymnasium.envs.registration import register
from .envs.game import Game, game_feature_shape, encoded_action
from .utils import move_to_coordinate

register(
    id="gymxq/xqv0",
    entry_point="gymxq.envs:XiangQiV0",
    max_episode_steps=300,
)

register(
    id="gymxq/xqv1",
    entry_point="gymxq.envs:XiangQiV1",
    max_episode_steps=300,
)

__all__ = ["Game", "game_feature_shape", "encoded_action", "move_to_coordinate"]
