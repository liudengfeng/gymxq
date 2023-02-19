import gymxq
from gymxq import Game
import time
import gymnasium
from gymxq.constants import *
import numpy as np

n = 4
k = 1
init_fen = "3ak1NrC/4a4/4b4/9/9/9/9/9/2p1r4/3K5 r - 118 0 297"


envs = gymnasium.vector.make(
    "gymxq/xqv1", render_mode="ansi", init_fen=init_fen, gen_qp=False, num_envs=n
)
# envs.render()
obs, _ = envs.reset()
assert obs["s"].shape == (n, k * NUM_ROW * NUM_COL)
# ss = obs["s"]
# for s in ss:
#     print(np.flipud(s.reshape((10, 9))))
actions = [
    Game.move_string_to_action(move) for move in ["6957", "8988", "8988", "6948"]
]
observations, rewards, termination, truncation, infos = envs.step(actions)
ss = observations["s"]
for s in ss:
    print(np.flipud(s.reshape((10, 9))))
print("=" * 60)
print("step 1")
print(rewards, termination, truncation)
final_observation = infos["final_observation"]
for i in range(n):
    s = final_observation[i]
    if s is not None:
        print(np.flipud(s["s"].reshape((10, 9))))
    else:
        print(s)

# final_info = infos["final_info"]

actions = [
    Game.move_string_to_action(move) for move in ["6957", "2131", "7978", "7989"]
]
observations, rewards, termination, truncation, infos = envs.step(actions)
print("step 2")
print(rewards, termination, truncation)
final_observation = infos["final_observation"]
for i in range(n):
    s = final_observation[i]
    if s is not None:
        print(np.flipud(s["s"].reshape((10, 9))))
    else:
        print(s)

actions = [
    Game.move_string_to_action(move) for move in ["6957", "2131", "7978", "4867"]
]
observations, rewards, termination, truncation, infos = envs.step(actions)
print("step 3")
print(rewards, termination, truncation)
final_observation = infos["final_observation"]
for i in range(n):
    s = final_observation[i]
    if s is not None:
        print(np.flipud(s["s"].reshape((10, 9))))
    else:
        print(s)
