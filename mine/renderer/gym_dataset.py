
import gym

from torch.utils.data.dataset import Dataset

class GymDataset(Dataset):

    def __init__(self, env):
        self.n_episodes = 100
        
        self.env = env

    def __getitem__(self, index):
        action = self.env.action_space.sample()
        observation, reward, done, info = self.env.step(action)
        if done:
            self.env.reset()

        return action, observation


    def __len__(self):
        return self.n_episodes


