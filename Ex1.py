import numpy as np
import random

action_space = [1, 2, 3, 4]  # 1 for action up, 2 for down, 3 for left and 4 for right.
rewards = np.array([[0, 0, 0, 1], [0, 0, 0, -1], [0, 0, 0, 0]])  # reward matrix
rows = rewards.shape[0]
cols = rewards.shape[1]
gamma = 0.9
alpha = 0.3


class QLearning:
    def __init__(self, episodes):
        self.determine = False
        self.episodes = episodes
        # initial Q values
        self.q_values = {}
        for x in range(0, rows):
            for y in range(0, cols):
                self.q_values[(x, y)] = {}
                for a in action_space:
                    self.q_values[(x, y)][a] = 0  # Q value is a dict of dict

    def run_q(self):
        epsilon = 0.3
        for episode in range(self.episodes):
            self.state = (2, 0)  # starting position down left
            done = False
            while done == False:
                exp_rate = random.uniform(0, 1)
                # Exploration-Exploitation-TradeOff
                if exp_rate > epsilon:  # take max q-value
                    action = self.get_q_max(self.q_values[self.state])[0]
                else:  # take random action
                    action = np.random.choice(action_space)
                new_state, reward = self.simulator(self.state, action)

                # Update Q-Table for Q(s,a)
                self.q_values[self.state][action] = (1-alpha)*self.q_values[self.state][action] + \
                                                    alpha*(reward + gamma*self.get_q_max(self.q_values[new_state])[1])
                self.state = new_state
                if self.is_done(self.state):
                    done = True

        return self.q_values

    def run_sarsa(self):
        epsilon = 0.3
        for episode in range(self.episodes):
            self.state = (2, 0)  # starting position down left
            done = False
            while done == False:
                action = self.choose_action(self.state, epsilon)
                new_state, reward = self.simulator(self.state, action)
                action2 = self.choose_action(new_state, epsilon)
                # Update Q-Table for Q(s,a)
                self.q_values[self.state][action] = (1 - alpha) * self.q_values[self.state][action] + \
                                                    alpha * (reward + gamma * self.q_values[new_state][action2])
                self.state = new_state
                if self.is_done(self.state):
                    done = True

        return self.q_values

    def choose_action(self, state, epsilon):
        exp_rate = random.uniform(0, 1)
        # Exploration-Exploitation-TradeOff
        if exp_rate > epsilon:  # take max q-value
            return self.get_q_max(self.q_values[state])[0]
        else:  # take random action
            return np.random.choice(action_space)

    def simulator(self, s, a):
        s_next = self.get_next_states(s, a)
        r = rewards[s_next[0], s_next[1]]  # reward in the new position after executing action a
        return s_next, r

    #  return the next 3 states ( up, left, right) regarding the actual state s and the action taken a
    def get_next_states(self, state, action):
        if self.determine:
            if action == 1:
                new_state = (state[0] - 1, state[1])
            elif action == 2:
                new_state = (state[0] + 1, state[1])
            elif action == 3:
                new_state = (state[0], state[1] - 1)
            else:
                new_state = (state[0], state[1] + 1)
            self.determine = False
        else:
            # non-deterministic
            action = self.prob_action(action)
            self.determine = True
            new_state = self.get_next_states(state, action)

        if 0 <= new_state[0] <= 2 and 0 <= new_state[1] <= 3 and new_state != (1, 1):  # next state stays in the grid
            return new_state
        else:  # next state is the same state
            return state

    @staticmethod
    def prob_action(action):
        if action == 1:
            return np.random.choice([1, 3, 4], p=[0.8, 0.1, 0.1])  # "up", "left", "right"
        if action == 2:
            return np.random.choice([2, 3, 4], p=[0.8, 0.1, 0.1])  # "down", "left", "right"
        if action == 3:
            return np.random.choice([3, 1, 2], p=[0.8, 0.1, 0.1])  # "left", "up", "down"
        if action == 4:
            return np.random.choice([4, 1, 2], p=[0.8, 0.1, 0.1])  # "right", "up", "down"

    @staticmethod
    def get_q_max(action_dict):
        max_q = -1000
        max_a = 0
        for a in action_dict:
            if action_dict[a] > max_q:
                max_a = a
                max_q = action_dict[a]
        if max_q == 0:
            max_a = np.random.choice(action_space)
        return max_a, max_q

    @staticmethod
    def is_done(state):
        if state == (0, 3) or state == (1, 3):
            return True
        return False


def main():
    print('\n_____________________Q-LEARNING_____________________\n')
    q_table = QLearning(500).run_q()
    print_table(q_table)
    print('\n_____________________SARSA_____________________\n')
    q_table = QLearning(500).run_sarsa()
    print_table(q_table)


def print_table(q_table):
    print(q_table)
    grid = np.zeros((3, 4), dtype=int)
    for state in q_table:
        if state in [(0, 3), (1, 3), (1, 1)]:
            continue
        action_dict = q_table[state]
        q_list = []
        for a in action_dict:
            q_list.append(action_dict[a])
        idx = q_list.index(max(q_list)) + 1
        grid[state[0], state[1]] = idx
    print(grid)


if __name__ == "__main__":
    main()

