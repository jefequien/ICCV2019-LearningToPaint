import gym
from gym import spaces

import numpy as np
from strokes import draw_oval

class CanvasEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(CanvasEnv, self).__init__()
        self.shape = (128, 128, 1)
        self.max_step_count = 200
        self.stroke_type = "oval"

        self.canvas = np.zeros(self.shape, dtype=np.float32)
        self.target = np.zeros(self.shape, dtype=np.float32)
        self.step_count = 0
        self.last_dist = 0
        self.dist = 0

        act_shape = (10,)
        obs_shape = (self.shape[0], self.shape[1], self.shape[2] * 2)
        self.action_space = spaces.Box(low=0, high=1, shape=act_shape, dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=obs_shape, dtype=np.float32)

        if self.stroke_type == "oval":
            self.draw_stroke = draw_oval
        else:
            raise Exception("Stroke type not defined: {}".format(self.stroke_type))

    def step(self, action):
        stroke = self.draw_stroke(action)
        stroke = stroke[:, :, np.newaxis]
        self.canvas = self.canvas * (1 - stroke) + stroke

        obs = np.concatenate([self.canvas, self.target], axis=2)

        self.last_dist = self.dist
        self.dist = self.cal_dist()
        reward = self.dist - self.last_dist
        
        self.step_count += 1
        done = (self.step_count >= self.max_step_count)

        info = {}
        return obs, reward, done, info

    def reset(self):
        self.canvas.fill(0)
        self.target.fill(0)
        self.step_count = 0
        self.dist = 0
        self.last_dist = 0

    def set_target(self, img):
        self.target = img.astype(np.float32) / 255
        self.dist = self.cal_dist()

    def cal_dist(self):
        return 1 - np.mean(np.square(self.canvas - self.target))
