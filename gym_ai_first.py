import numpy as np
import itertools
import sys
from collections import defaultdict
sys.path.append('C:/Users/b1017089/Project/RL/')
#%matplotlib inline
import matplotlib
from lib import plotting
matplotlib.style.use('ggplot')

def sarsa(env, num_episodes, discount_factor=0.9, alpha=0.5):
    #num_episodes = 繰り返しエピソード数
    # テーブル作成
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    V = np.zeros(env.observation_space.n)

    policy = make_greedy_policy(Q, env.action_space.n)
    history = []
    episode_history = []
    
    for i_episode in range(num_episodes):
        if (i_episode+1)%10==0:
            print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
            sys.stdout.flush()
        state = env.reset()
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
        if i_episode%1==0:
            history.append([[state, np.nan, list(Q[state])]])
            episode_history.append([0.0, 0.0, [], 0.0])
            V[state] = 0.0 + max(Q[state])
        for t in itertools.count():
            next_state, reward, done, _ = env.step(action)
            if done and reward == 0:
                reward = -10
            elif done and reward == 1:
                reward = 10
            else:
                reward = -1
            
            if i_episode % 1 == 0:
                history[int(i_episode/1)][-1][1] = action
                history[int(i_episode/1)].append([next_state, np.nan, list(Q[next_state])])
                V[next_state] = reward + max(Q[next_state])
            
            #次の行動を記録決定する
            next_action_probs = policy(next_state)
            next_action = np.random.choice(np.arange(len(next_action_probs)), p=next_action_probs)
            
            #価値関数の更新
            td_target = reward + discount_factor*Q[next_state][next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            #log
            episode_history[i_episode][0] += reward
            episode_history[i_episode][1] += td_delta**2

            if done:
                episode_history[i_episode][2] = list(V)
                episode_history[i_episode][3] = t
                break
            action = next_action
            state = next_state
    return Q, history, episode_history

def make_greedy_policy(Q, nA):
    def policy_fn(observation):
        A = np.zeros(nA)
        best_action = np.random.choice(np.flatnonzero(Q[observation]==Q[observation].max()))
        A[best_action] = 1.0
        return A
    return policy_fn