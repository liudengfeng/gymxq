from gymnasium.envs.registration import register

register(
    id="xqv0",
    entry_point="gymxq.envs:XiangQiV0",
    max_episode_steps=200,
    # order_enforce=False,
    # disable_env_checker=True,
)

register(
    id="xqv1",
    entry_point="gymxq.envs:XiangQiV1",
    max_episode_steps=200,
    # order_enforce=False,
    # disable_env_checker=True,
)
