import gym
import os
import random
import numpy as np
from statistics import mean


class MountainCarContinuous:
    def __init__(self, epsilon, alpha, gamma):
        self.system = "OS X"
        self.env = gym.make('MountainCarContinuous-v0')

        self.epsilon = epsilon  # probability of choosing a random action
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount rate

        self.velocity_obs = [i / 100 for i in range(-7, 8, 1)]
        self.pos_obs = [i / 10 for i in range(-12, 7, 1)]
        self.acts = [i / 10 for i in range(-10, 11, 1)]  # sampling of the action space
        obs_size = len(self.velocity_obs) * len(self.pos_obs)
        act_size = len(self.acts)
        self.path = r"{}".format(os.path.expanduser("~/Desktop/Q"))  # Desktop file path

        try:
            self.Q = np.load(self.path + ".npy")
        except FileNotFoundError:
            self.Q = np.zeros([obs_size, act_size])  # Initialize arbitrary values

    def saveQ(self, path):
        np.save(path + ".npy", self.Q)

    def get_Q_index(self, state):
        for i in range(len(self.pos_obs)):
            # Position
            if self.pos_obs[i] <= state[0] < self.pos_obs[i + 1]:
                # Velocity
                for j in range(len(self.velocity_obs)):
                    if self.velocity_obs[j] <= state[1] < self.velocity_obs[j + 1]:
                        return len(self.velocity_obs) * i + j  # row we need in Q

    def get_Q_action(self, action):
        for i in range(len(self.acts)):
            # Position
            if self.acts[i] <= action < self.acts[i + 1]:
                return i

    def learn(self, episodes, rendering):
        print("Learning MountainCar-v0 model with {} episodes ".format(episodes))
        global_max_score = -1e10
        global_max_height = -1e10
        self.env.seed(0)
        scores = []
        for i in range(1, episodes):
            obs = self.env.reset()
            state = self.get_Q_index(obs)
            done = False
            total_score = 0
            max_height = -1e10

            self.epsilon -= self.epsilon / episodes
            while not done:
                self.env.render() if rendering else 0  # picture
                if random.uniform(0, 1) < self.epsilon:  # e-greedy policy
                    action = self.get_Q_action(self.env.action_space.sample())
                else:
                    action = np.argmax(self.Q[state])

                next_obs, reward, done, info = self.env.step([self.acts[action]])
                modified_reward = reward + 100 * self.gamma * (
                            abs(next_obs[1]) - abs(obs[1]))  # 10 * self.gamma * (abs(next_obs[1]) - abs(obs[1]))
                next_state = self.get_Q_index(next_obs)
                self.Q[state, action] = (1 - self.alpha) * self.Q[state, action] + self.alpha * (
                        modified_reward + self.gamma * np.max(self.Q[next_state]) - self.Q[state, action])
                state = next_state
                total_score += reward
                max_height = max(max_height, next_obs[0])

            scores.append(total_score)
            global_max_score = max(global_max_score, total_score)
            global_max_height = max(global_max_height, max_height)
            if i % 5 == 0:
                print("Episode: {}".format(i))
                print(" Total score for episode {} : {}, Max height : {}, Current epsilon {}".format(i, total_score,
                                                                                                     max_height,
                                                                                                     self.epsilon))
                print(" GLOBAL MAXIMUMS: Max score : {}, Max height  : {}".format(global_max_score, global_max_height))
                print('-' * 150)
                self.saveQ(self.path)

        print("Training finished\n")
        print("Max score: {} ({} : mean), Max height: {}".format(int(global_max_score), int(mean(scores)),
                                                                 global_max_height))
        self.env.close()


if __name__ == '__main__':
    episode_number = 100
    epsilon = 0.1  # probability of choosing a random action
    alpha = 0.5  # learning rate
    gamma = 0.7  # discount rate
    rendering = False  # picture
    MountainCarContinuous(epsilon, alpha, gamma).learn(episode_number, rendering)

'''
results for  episode_number = 100:
Max score: 95 (81 : mean), Max height: 0.4796387289786885

results for  episode_number = 1000:
Max score: 95 (91 : mean), Max height: 0.4858011275603664

results for  episode_number = 10000:
Max score: 96 (93 : mean), Max height: 0.49215787672222855
'''
