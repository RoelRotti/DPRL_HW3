#hyper parameters
size = 10
gamma = 0.9
alpha = 0.2
eps = 0.9
a,b = 0,1
import numpy as np
import random

"""
de meest linker state is 0 en de meest rechter state is size -1
de actie a is 0 en de actie b is 1
Q tabel heeft de vorm Q[state][action]
"""


def next_state(state, action):
    #returns next state and reward
    if state == size-1 and action == a:
        return state, 1
    if state == 0 and action == b:
        return state, 0.2
    if action == a:
        return state + 1 ,0
    if action == b:
        return state - 1 ,0
    print("next state error, fix inmediately")


class Tab_Q_Agent:

    def __init__(self):
        self.current_state = random.randint(0,size-1)
        self.Q = [[0.0] * 2 for _ in range(size)]
        self.total_reward = 0
        self.moves_made = 0

    def do_action(self):
        #random with p = eps
        if random.uniform(0,1) < eps:
            action = random.choice([a,b])
        else:
        # pick highest Q
            max_q = max(self.Q[self.current_state])
            action = self.Q[self.current_state].index(max_q)

        new_state, rew = next_state(self.current_state, action)
        self.update_q(self.current_state, action, rew, new_state)
        #update state and statistics
        self.current_state = new_state
        self.total_reward += rew
        self.moves_made += 1

    def update_q(self, state, action, rew, new_state):
        self.Q[state][action] = (1-alpha) * self.Q[state][action] + \
                                alpha*(rew + gamma*max(self.Q[new_state]))

    def output(self):
        #function to print
        print(self.moves_made , " moves has been made, gaining a reward of ", self.total_reward)
        print("average reward :", self.total_reward / self.moves_made)
        print(self.Q)
        print("________________________________")


test_agent = Tab_Q_Agent()
for i in range(10000):
    test_agent.do_action()
    if i%100 == 0:
        test_agent.output()



