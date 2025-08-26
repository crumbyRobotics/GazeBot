import numpy as np
import torch
import torchvision.transforms.functional as VF
import gym
from gym import spaces


class TongSystemObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, state_mean, state_std, device):
        if not isinstance(env.observation_space, spaces.Dict):
            raise ValueError("env.observation_space should be space.Dict.")
        super(TongSystemObsWrapper, self).__init__(env)

        self.device = torch.device(device)

        self.state_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,))
        self.observation_space = spaces.Dict({"state": self.state_space, "image": self.image_space, "depth": self.depth_space})

        assert len(state_mean) == len(state_std) == self.state_space.shape[0]
        self.state_mean = np.array(state_mean)
        self.state_std = np.array(state_std)

    def observation(self, observation):
        state = observation["state"][:14]
        state = (state - self.state_mean) / self.state_std

        obs_batch = {}
        obs_batch["state"] = torch.as_tensor(np.array([state]), dtype=torch.float, device=self.device)
        obs_batch["image"] = torch.as_tensor(np.array([observation["image"]]), dtype=torch.float, device=self.device)
        obs_batch["depth"] = torch.as_tensor(np.array([observation["depth"]]).astype(np.float32), dtype=torch.float, device=self.device)

        obs_batch["image"][:, :3] = VF.normalize(obs_batch["image"][:, :3], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        obs_batch["image"][:, 3:] = VF.normalize(obs_batch["image"][:, 3:], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return obs_batch

    def step(self, action, *args, **kwargs):
        observation, reward, done, info = self.env.step(action, *args, **kwargs)
        return self.observation(observation), reward, done, info

    def freeze_step(self, *args, **kwargs):
        observation, reward, done, info = self.env.freeze_step(*args, **kwargs)
        return self.observation(observation), reward, done, info


class SingleArmActionWrapper(gym.ActionWrapper):
    def __init__(self, env, control_arm=None):
        super().__init__(env)
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,))
        if control_arm is not None:
            assert control_arm in ("left", "right")
            self.control_arm = control_arm

    def action(self, action):
        assert action.shape == self.action_space.shape

        if self.env.action_space.shape[0] == 7:
            if self.control_arm == "left":
                return action[:7]
            else:
                return action[7:14]
        else:
            return action

    def step(self, action, *args, **kwargs):
        return self.env.step(self.action(action), *args, **kwargs)
