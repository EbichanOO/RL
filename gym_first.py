import sys
import gym
from gym.envs.registration import register
try:
    register(
        id='FrozenLakeNotSlippery-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name' : '4x4', 'is_slippery': False}
    )
except gym.error.Error:
    pass

import matplotlib.pyplot as plt
env = gym.make('FrozenLakeNotSlippery-v0') # 用意されている環境を指定して読み込む
env.reset() # 状態を初期化
from gym_ai_first import sarsa
Q, history, episode_history = sarsa(env, 50)

env.render()#状態の表示

maxQ_by_step = []
for e in range(len(history)):
    for i in range(len(history[e])):
        maxQ_by_step.append(max(history[e][i][2]))
plt.plot(maxQ_by_step)
plt.xlabel('step')
plt.ylabel('max_Q')
plt.title('max_Q by step')
#plt.legend()
plt.show()