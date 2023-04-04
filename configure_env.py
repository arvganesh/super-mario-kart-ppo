# Code adapted from https://github.com/thu-ml/tianshou/blob/master/examples/atari/atari_wrapper.py

import retro
import numpy as np
import gymnasium as gym
import warnings
import os

from tianshou.env import SubprocVectorEnv, ShmemVectorEnv

class FrameSkipping(gym.Wrapper):
    """Return only every `skip`-th frame (frameskipping) using most recent raw
    observations
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
        if new_step_api:
            return obs_list[-1], total_reward, term, trunc, info
        return obs_list[-1], total_reward, done, info

class Discretizer(gym.ActionWrapper):
    """
    Wrap a gym environment and make it use discrete actions.
    Args:
        combos: ordered list of lists of valid button combinations
    """

    def __init__(self, env, combos):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.MultiBinary)
        buttons = env.unwrapped.buttons
        self._decode_discrete_action = []
        for combo in combos:
            arr = np.array([False] * env.action_space.n)
            for button in combo:
                arr[buttons.index(button)] = True
            self._decode_discrete_action.append(arr)

        self.action_space = gym.spaces.Discrete(len(self._decode_discrete_action))

    def action(self, act):
        return self._decode_discrete_action[act].copy()

class SimpleDrivingDiscrtizer(Discretizer):
    """
    Simplify the action space for driving in Mario Kart. Used as a baseline.
    From 468 SNES Operations -> 6 actions.
    """
    def __init__(self, env):
        super().__init__(env=env, combos=[
            [], # Do nothing
            ['LEFT'], # Turn left
            ['RIGHT'], # Turn right
            ['B'], # Accelerate
            ['LEFT', 'B'], # Accelerate and turn left
            ['RIGHT', 'B'], # Accelerate and turn right
        ])

class RoadFocusWrapper(gym.ObservationWrapper):
    """
    Remove mini-map from observations.
    """
    def __init__(self, env):
        super().__init__(env=env)
        assert(isinstance(self.observation_space, gym.spaces.Box))
        assert(len(self.observation_space.shape) == 2)
        self.start_row = 25 # Top 25 rows of pixels include timer.
        self.end_row = self.observation_space.shape[0] // 2 - 10 # - 10 to remove black bars from the bottom rows.
        num_rows = self.end_row - self.start_row
        self.observation_space = gym.spaces.Box(low=0, high=225, shape=(num_rows, self.observation_space.shape[1]), dtype=np.uint8)

    def observation(self, obs):
        return obs[self.start_row : self.end_row, :]

# Given an environment ID, create a wrapped environment.
def create_wrapped_env(env_id, simple_action_space=True, simple_obs_space=True, grayscale=True, resize_frame=True, normalize=True, frame_skip = 4, frame_stack=4, max_episode_steps=1000, custom_integration_path=None):
    # Attempt to integrate new enviornment if specified
    if custom_integration_path is not None:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Directory of the script
        NEW_INTEGRATIONS_DIR = os.path.join(SCRIPT_DIR, custom_integration_path)
        retro.data.Integrations.add_custom_path(NEW_INTEGRATIONS_DIR) # Add folder containing new integrations to path.
    
    env = retro.make(env_id, inttype=retro.data.Integrations.ALL, render_mode="rgb_array")

    if grayscale:
        env = gym.wrappers.GrayScaleObservation(env)
    if simple_obs_space:
        env = RoadFocusWrapper(env)
    if resize_frame:
        env = gym.wrappers.ResizeObservation(env, (84, 84))
    # if normalize: currently directly normalization when forwarding in model
    #     env = gym.wrappers.NormalizeObservation(env)
    if simple_action_space:
        env = SimpleDrivingDiscrtizer(env)
    if frame_skip:
        env = FrameSkipping(env, frame_skip)
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

    return training_envs, test_envs