# Code adapted from https://github.com/thu-ml/tianshou/blob/master/examples/atari/atari_wrapper.py

import retro
import numpy as np
import gymnasium as gym
import warnings
import os
import datetime

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
                
        final_obs = np.max(obs_list[-2:], axis=0)

        if new_step_api:
            return final_obs, total_reward, term, trunc, info
        return final_obs, total_reward, done, info

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
    From 468 SNES Operations -> 4 actions.
    """
    def __init__(self, env):
        super().__init__(env=env, combos=[
            ['B'], # Accelerate
            ['LEFT', 'B'], # Accelerate and turn left
            ['RIGHT', 'B'], # Accelerate and turn right
            ['L', 'B'], # Accelerate and Hop
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
    

class RandomResetWrapper(gym.Wrapper):
    """
    On reset, take a random number of random actions to add variation to initial states.
    """
    def __init__(self, env, max_random_ops=30) -> None:
        super().__init__(env)
        self.max_random_ops = max_random_ops
    
    def step(self, action):
        return self.env.step(action)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.max_random_ops == 0:
            return obs, info

        num_ops = np.random.randint(0, self.max_random_ops)

        for _ in range(num_ops):
            obs, reward, term, trunc, info = self.env.step(np.random.randint(0, 4))
            if term or trunc:
                return self.env.reset(**kwargs)
            
        obs, reward, term, trunc, info = self.env.step(self.env.action_space.sample())
        return obs, info


# Given an environment ID, create a wrapped environment.
def create_wrapped_env(env_id, 
                       random_initial_state=True, 
                       simple_action_space=True, 
                       simple_obs_space=True, 
                       grayscale=True, 
                       resize_frame=True, 
                       normalize=True, 
                       frame_skip = 4, 
                       frame_stack=4, 
                       max_episode_steps=1000, 
                       custom_integration_path=None,
                       record_agent=False,
                       video_path=None,
                       max_random_ops=20,
                       scenario='scenario',
                       info=None):
    
    # Attempt to integrate new enviornment if specified
    if custom_integration_path is not None:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Directory of the script
        NEW_INTEGRATIONS_DIR = os.path.join(SCRIPT_DIR, custom_integration_path)
        retro.data.Integrations.add_custom_path(NEW_INTEGRATIONS_DIR) # Add folder containing new integrations to path.

    env = retro.make(env_id, inttype=retro.data.Integrations.ALL, render_mode="rgb_array", scenario=scenario, info=info)

    # Clip Rewards
    env = gym.wrappers.TransformReward(env, lambda x: np.sign(x))

    if grayscale:
        env = gym.wrappers.GrayScaleObservation(env)
    if simple_obs_space:
        env = RoadFocusWrapper(env)
    if resize_frame:
        env = gym.wrappers.ResizeObservation(env, (84, 84))    
    if simple_action_space:
        env = SimpleDrivingDiscrtizer(env)
    if frame_stack:
        env = gym.wrappers.FrameStack(env, frame_stack, False)
    if frame_skip:
        env = FrameSkipping(env, frame_skip)
    if random_initial_state:
        env = RandomResetWrapper(env, max_random_ops=max_random_ops)
    if max_episode_steps:
        env = gym.wrappers.TimeLimit(env, max_episode_steps)
    
    # TODO: Get this setup
    # if record_agent:
    #     episode_trigger = lambda num_eps: True
    #     step_trigger = None
    #     video_path = os.path.join(video_path, env_id, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    #     env = gym.wrappers.RecordVideo(env, video_path, episode_trigger, step_trigger, max_episode_steps, env_id)

    return env

# Make environments for training and testing
def make_retro_env(task, training_num, test_num, **kwargs):
    training_envs = ShmemVectorEnv([lambda: create_wrapped_env(task, **kwargs) for _ in range(training_num)])
    test_envs = ShmemVectorEnv([lambda: create_wrapped_env(task, **kwargs) for _ in range(test_num)])

    return training_envs, test_envs