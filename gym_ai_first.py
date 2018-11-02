import numpy as np
import itertools
import sys
from collections import defaultdict
sys.path.append('C:/Users/b1017089/Project/RL/')
%matplotlib inline
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
            history.append([[state, np.nun, list(Q[state])]])
            episode_history.append([0.0, 0.0, [], 0.0])
            V[state] = 0.0 + max(Q[state])

def make_greedy_policy(Q, nA):
    def policy_fn(observation):
        A = np.zeros(nA)
        best_action = np.random.choice(np.flatnonzero(Q[observation]==Q[observation].max()))
        A[best_action] = 1.0
        return A
    return policy_fn