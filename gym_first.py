import sys
from gym.envs.registration import register
try:
    register(
        id='FrozenLakeNotSlippery-v0',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name' : '4x4', 'is_slippery': False}
    )
except gym.error.Error:
    pass

import gym
env = gym.make('FrozenLakeNotSlippery-v0') # 用意されている環境を指定して読み込む
env.reset() # 状態を初期化
#env.render() #状態を表示

actions = [2, 3, 2, 1, 0, 3] # 左=0、下=1、右=2、上=3
for action in actions:
    step=env.step(action)
    print(step)
    env.render()