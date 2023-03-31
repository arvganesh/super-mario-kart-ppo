# Code adapted from https://github.com/thu-ml/tianshou/blob/master/examples/atari/atari_wrapper.py

import retro
import numpy as np
import gymnasium as gym
import warnings
import os

from tianshou.env import SubprocVectorEnv, ShmemVectorEnv

class MaxAndSkipEnv(gym.Wrapper):
    """Return only every `skip`-th frame (frameskipping) using most recent raw
    observations (for max pooling across time steps)
    :param gym.Env env: the environment to wrap.
    :param int skip: number of `skip`-th frame.
    """

    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Step the environment with the given action. Repeat action, sum
        reward, and max over last observations.
        """
        obs_list, total_reward = [], 0.
        new_step_api = False
        for _ in range(self._skip):
            step_result = self.env.step(action)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
            else:
                obs, reward, term, trunc, info = step_result
                done = term or trunc
                new_step_api = True
            obs_list.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(obs_list[-2:], axis=0)
        if new_step_api:
            return max_frame, total_reward, term, trunc, info

        return max_frame, total_reward, done, info

# Given an environment ID, create a wrapped environment.
def create_wrapped_env(env_id, grayscale=True, resize_frame=True, normalize=True, frame_skip = 4, frame_stack=4, max_episode_steps=1000, custom_integration_path=None):
    # Attempt to integrate new enviornment if specified
    if custom_integration_path is not None:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Directory of the script
        NEW_INTEGRATIONS_DIR = os.path.join(SCRIPT_DIR, custom_integration_path)
        retro.data.Integrations.add_custom_path(NEW_INTEGRATIONS_DIR) # Add folder containing new integrations to path.
    
    env = retro.make(env_id, inttype=retro.data.Integrations.ALL, use_restricted_actions=retro.Actions.DISCRETE, render_mode="rgb_array")

    if grayscale:
        env = gym.wrappers.GrayScaleObservation(env)
    if resize_frame:
        env = gym.wrappers.ResizeObservation(env, (84, 84))
    # if normalize:
    #     env = gym.wrappers.NormalizeObservation(env)
    if frame_skip:
        env = MaxAndSkipEnv(env, frame_skip)
    if frame_stack:
        env = gym.wrappers.FrameStack(env, frame_stack, False)
    if max_episode_steps:
        env = gym.wrappers.TimeLimit(env, max_episode_steps)

    return env

# Make environments for training and testing
def make_retro_env(task, seed, training_num, test_num, **kwargs):
    # TODO: envpool integration
    training_envs = ShmemVectorEnv([lambda: create_wrapped_env(task, **kwargs) for _ in range(training_num)])
    test_envs = ShmemVectorEnv([lambda: create_wrapped_env(task, **kwargs) for _ in range(test_num)])

    # training_envs = DummyVectorEnv([lambda: create_wrapped_env(task, **kwargs) for _ in range(training_num)])
    # test_envs = DummyVectorEnv([lambda: create_wrapped_env(task, **kwargs) for _ in range(training_num)])

    return training_envs, training_envs