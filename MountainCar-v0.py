import gym
import os
import random
import numpy as np


class MountainCar:
    def __init__(self, epsilon, alpha, gamma):
        self.system = "OS X"
        self.env = gym.make('MountainCar-v0')
        self.action_space = self.env.action_space.n  # 0 is push left, 1 is  no push and 2 is push right
        self.observation_space = self.env.observation_space  # 0 is position [-1.2 - 0.6], 1 is velocity [-0.07 - 0.07]

        self.epsilon = epsilon  # probability of choosing a random action
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount rate

        self.velocity_obs = [i / 100 for i in range(-7, 8, 1)]
        self.pos_obs = [i / 10 for i in range(-12, 7, 1)]
        obs_size = len(self.velocity_obs) * len(self.pos_obs)
        self.path = r"{}".format(os.path.expanduser("~/Desktop/Q"))  # Desktop file path

        try:
            self.Q = np.load(self.path + ".npy")
        except FileNotFoundError:
            self.Q = np.zeros([obs_size, self.action_space])  # Initialize Q values

    def saveQ(self, path):  # save Q values
        np.save(path + ".npy", self.Q)

    def get_Q_index(self, state):
        for i in range(len(self.pos_obs)):
            # Position
            if self.pos_obs[i] <= state[0] < self.pos_obs[i + 1]:
                # Velocity
                for j in range(len(self.velocity_obs)):
                    if self.velocity_obs[j] <= state[1] < self.velocity_obs[j + 1]:
                        return len(self.velocity_obs) * i + j  # row we need in Q

    def learn(self, episodes, until_solved, rendering):

        print("Learning MountainCar-v0 model with {} episodes ".format(episodes))
        print("Learning  model until solved status: {}\n".format(until_solved))

        global_max_score = -1e10
        global_max_height = -1e10
        episodes_to_solve = 0
        self.env.seed(0)

        for i in range(1, episodes):
            obs = self.env.reset()
            state = self.get_Q_index(obs)
            done = False
            total_score = 0
            max_height = -1e10
            step = 0

            while not done:
                step += 1
                self.env.render() if rendering else 0  # picture
                if random.uniform(0, 1) < self.epsilon:  # e-greedy policy
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.Q[state])

                next_obs, reward, done, info = self.env.step(action)
                modified_reward = reward + self.gamma * abs(next_obs[1]) - abs(obs[1])  # reward based on potentials
                next_state = self.get_Q_index(next_obs)

                # update Q
                self.Q[state, action] = (1 - self.alpha) * self.Q[state, action] + self.alpha * (
                        modified_reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state, action])
                state = next_state
                self.epsilon -= self.epsilon / episodes  # epsilon reduction
                total_score += reward
                max_height = max(max_height, next_obs[0])

                # end if solved
                if done and step < 200:
                    if until_solved:
                        print("Solved in {} episodes".format(i))
                        self.env.close()
                        raise SystemExit()
                    if not episodes_to_solve:
                        episodes_to_solve = i

            global_max_score = max(global_max_score, total_score)
            global_max_height = max(global_max_height, max_height)
            if i % 5 == 0:
                print("Episode: {}".format(i))
                print(" Total score for episode {} : {}, Max height : {}".format(i, total_score, max_height))
                print(" GLOBAL MAXIMUMS: Max score : {}, Max height  : {}".format(global_max_score, global_max_height))
                print('-' * 150)
                self.saveQ(self.path)

        print("Training finished\n")
        solve_status = "Solved in {} episodes".format(episodes_to_solve) if global_max_height >= 0.5 else "Not Solved"
        print("Max score: {} ({}), Max height: {}, Solve status : {}".format(global_max_score, 200 + global_max_score,
                                                                             global_max_height,
                                                                             solve_status))
        self.env.close()


if __name__ == '__main__':
    episode_number = 100  # number of episodes
    learn_until_solved = False  # stop if solved
    rendering = False  # picture
    epsilon = 0.5  # probability of choosing a random action
    alpha = 0.5  # learning rate
    gamma = 0.9  # discount rate
    MountainCar(epsilon, alpha, gamma).learn(episode_number, learn_until_solved, rendering)

'''
results for  episode_number = 100:
Max score: -111.0 (89.0), Max height: 0.5436075341139893, Solve status : Solved in 30 episodes

results for  episode_number = 1000:
Max score: -88.0 (112.0), Max height: 0.5393644329783768, Solve status : Solved in 29 episodes
'''
