# Training a PPO agent to play Mario Kart, from start to finish.

![eval_deterministic](https://user-images.githubusercontent.com/21336191/231527291-84b5e5d3-379a-4ee4-a7e2-8b9369d400fc.gif)

*actual agent footage, very sped up*

This repo describes how to integrate new ROMs into [stable-retro](https://github.com/MatPoliquin/stable-retro) + other setup guidelines. It also provides a simple training script for trianing agents.

In this example, we integrate Super Mario Kart (SNES) and use PPO to train an agent to complete Mario Circuit on Time Trials mode.

Most information in this guide summarizes [this](https://retro.readthedocs.io/en/latest/integration.html#) + adds some extra details about the game environment.

## Table of Contents

* [Setup and Overview](#setup-and-overview)
  * [Setting up stable-retro with Docker (for Mac Users)](#setting-up-stable-retro-with-docker-for-mac-users)
  * [Files in this repo](#files-in-this-repo)
  * [Integrating new games](#integrating-new-games)
* [Environment Design](#environment-design)
  * [Start State](#start-state)
  * [Action Space](#action-space)
  * [Observation Space](#observation-space)
  * [Reward Function, Done Condition, Time Penalty](#reward-function-done-condition-time-penalty)
* [Training Script Setup](#training-script-setup)
  * [Example Usage](#example-usage)

## Setup and Overview

### Setting up stable-retro with Docker (for Mac Users)

I'm running OS X 12.5 on an M1 Macbook Pro, so [I can't install stable-retro natively](https://github.com/MatPoliquin/stable-retro/issues/11) as of 3/16/2023. To get around this, I'm using a Docker container running Ubuntu 22.04 for now.

Create a docker container following the instructions in [this repo](https://github.com/arvganesh/stable-retro-docker).

Skip this step if you are running Linux – you can probably just run everything natively.

### Files in this repo
A high-level explanation of all the files in this repo:
- `Integrations/` – Each folder within this directory corresponds to a single new ROM integration.
- `Integrations/MarioKart-Snes/` – Contains integration data for Super Mario Kart on the SNES.
  - `Level1.state` – Save state from ROM, the point in the game where training episodes begin.
  - `data.json` – Locations of important variables in emulator RAM, used for reward function and done condition.
  - `scenario.json` – Specifies reward function and 'done' condition.
  - `script.lua` – Definition of reward function and 'done' condition, used in `scenario.json`.
  - `metadata.json` – Specifies potential starting states via `.state` files.
  - `rom.sfc` – ROM file for Mario Kart SNES (must be named `rom`)
  - `rom.sha` – SHA1 checksum for ROM file. Used to identify the ROM. (must be named `rom`).
  
- `example_integration.py`: Shows how to integrate new roms into stable-retro without modifying the stable-retro pacakge.
- `PPO_agent.py`: Script for training PPO agents on games in stable-retro. Uses Tianshou.
- `configure_env.py`: Create wrapped environments for training and testing agents.
- `network.py`: Defines the policy network used for PPO.

More detailed explanations of the files inside `Integrations/MarioKart-Snes` can be found [here](https://retro.readthedocs.io/en/latest/integration.html#game-integration).

### Integrating new games

*IMPORTANT: [Must read if you want to integrate new ROMs](https://retro.readthedocs.io/en/latest/integration.html#game-integration).*

*IMPORTANT: stable-retro already [contains many integrations](https://github.com/arvganesh/stable-retro/tree/master/retro/data/stable/) for games. To use them, you just need to [import the ROMs](https://retro.readthedocs.io/en/latest/getting_started.html?highlight=retro.import#importing-roms).*

Integration in this context means finding RAM locations of important variables using them to define the reward function and 'done' condition as given by the environment. This is done in `data.json`, `scenario.json`, and `metadata.json`.

To integrate new ROMs, you can use the [Gym Integration UI](https://github.com/openai/retro/releases/tag/f347d7e). However, the application is fairly buggy on my Mac, so I had to do most things it's supposed to do by hand. The one exception to this was creating the file `Level1.state`.

Note: To create `rom.sha`, I had to upload `rom.sfc` to [this](https://emn178.github.io/online-tools/sha1_checksum.html) and paste the result.

From my experience, the most time-consuming part of integration is finding RAM locations of important variables. However, many popular games already have their RAMs "mapped" out. [Here](https://datacrystal.romhacking.net/wiki/Super_Mario_Kart:RAM_map) is an example mapping for Super Mario Kart which saved me a lot of time. Pay close attention to signed vs. unsigned, the width of the variable in bytes, and endianness. For the SNES, [16-bit and 24-bit values are stored in little-endian](https://ersanio.gitbook.io/assembly-for-the-snes/the-fundamentals/endian), but this probably varies depending on the game console. This [blog post](https://www.videogames.ai/2019/01/29/Setup-OpenAI-baselines-retro.html) also has some helpful information and tutorials on integration.

## Environment Design
### Start State
`Level1.state` stores the default state in which episodes begin. The name of this file can be pretty much anything as long as it's specified in `metadata.json`.

I used the [Gym Integration UI](https://github.com/openai/retro/releases/tag/f347d7e) to create the start state file. It may also be possible to use other emulators to do this, but I haven't tried.

I've specified it to be the start of a race on Time Trials mode playing as Mario on Mario Circuit, but this is totally customizable.

<img width="256" alt="image" src="https://user-images.githubusercontent.com/21336191/225757384-95724610-adbc-461e-9cbb-19b99ba7e7a4.png">

### Action Space
There are 12 buttons on the SNES controller.

```json
"buttons": ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]
```

There are [4 primary ways](https://retro.readthedocs.io/en/latest/python.html#retro.Actions) that these actions can be represented as action spaces.

*Note: Be sure to specify the type of action space you want when calling `retro.make()`.*

However, using a wrapper defined in `configure_env.py`, I will simplify the action space into one that is useful for driving:

```
B (accelerate)
B + LEFT (accelerate + turn left)
B + RIGHT (accelerate + turn right)
B + L (accelerate + hop)
```

It's also possible to use a pre-discretized action space, defined by stable-retro. There are $2^{12}$ possible buttom combinations, but many combinations are invalid. For example, it doesn't make sense to press `UP` and `DOWN` at the same time. Additionaly, `SELECT` and `START` are removed as those two buttons are used to navigate game menus. To remove these invalid combinations, the buttons on the controller were broken down into 4 groups, with possible button combinations defined for each group. The total number of combinations is $3 * 3 * 13 * 4 = 468$.

```python
>>> print(env.action_space)
Discrete(468)
```

More on [Discrete](https://www.gymlibrary.dev/api/spaces/#discrete).

Here are the button groups defined by `stable-retro`. They are located [here](https://github.com/MatPoliquin/stable-retro/blob/90fa6f8d46416a0151d90e26296468d2f627740e/cores/snes.json#L9).

```json
"actions": [
    [[], ["UP"], ["DOWN"]],
    [[], ["LEFT"], ["RIGHT"]],
    [[], ["A"], ["B"], ["X"], ["Y"], ["A", "B"], ["B", "Y"], ["Y", "X"], ["X", "A"], ["A", "B", "Y"], ["B", "Y", "X"], ["Y", "X", "A"], ["A", "B", "Y", "X"]],
    [[], ["L"], ["R"], ["L", "R"]]
]
```
### Observation Space
Here is the observation space for SNES games. In words, a `244x256x3` tensor with values in the range `[0, 255]` of type `uint8`. Basically, just the raw pixels. This is defined by `stable-retro`.
```python
>>> print(env.observation_space)
Box(0, 255, (224, 256, 3), uint8)
```

However, using [wrappers](https://gymnasium.farama.org/api/wrappers/observation_wrappers/), these observations can be normalized, grayscaled, and more.

`configure_env.py` shows the various wrappers used to transformer the original observation space into the one used for training. In particular:

- Grayscale
- Resized to 84x84
- Normalized
- Frame Skipping + Max Pooling across the final two frames.
- Frame Stacking

The default observation wrappers are very similar to ones used for training agents to play Atari games.

### Reward Function, Done Condition, Time Penalty
`data.json` contains locations of important variables. [See this](https://retro.readthedocs.io/en/latest/integration.html#variable-locations-data-json).
```json
{
  "info": {
    "laps": {
      "address": 8261825,
      "type": "|u1"
    },
    "speed": {
      "address": 8261866,
      "type": "<s2"
    },
    "surface": {
      "address": 8261806,
      "type": "|u1"
    },
    "collision": {
      "address": 8261714,
      "type": "|u1"
    },
    "backward": {
      "address": 267,
      "type": "|u1"
    },
    "checkpoint": {
      "address": 8261824,
      "type": "|u1"
    },
    "num_checkpoints": {
      "address": 8257864,
      "type": "|u1"
    },
    "current_frame": {
      "address": 8257592,
      "type": "<u2"
    }
  }
}
```
- `laps` is at address `8261825` and is an unsigned single byte quantity. Endianness does not need to be specified for 1-byte quantities, hence `|`.

- `speed` is at address `8261866` and is a signed 2-byte integer in little-endian (denoted by `<`).

The same applies to the rest of the variables.

`scenario.json` uses these variables to define a reward function and done condition. [See this](https://retro.readthedocs.io/en/latest/integration.html#scenario-scenario-json).

```json
{
    "done": {
      "script": "lua:isDoneTrain"
    },
    "reward": {
      "script": "lua:overall_reward"
    },
    "scripts": [
      "script.lua"
    ]
}
```

The done condition is a function of variables in `data.json`. See `Integrations/MarioKart-Snes/script.lua` for more details.

Using `data.VARIABLE_NAME` allows you to access the value of `VARIABLE_NAME` as defined by `data.json`.

```lua
-- Done Conditions --
function overall_reward()
    if getLap() >= 5 or earlyStop then
        return true
    end
    return false
end
```

The reward is a function of variables in `data.json`. Here, we define an example reward by the lua function `overall_reward`:

```lua
function overall_reward()
    local wall_cf = -10.0
    local checkpoint_cf = 300.0
    local terrain_cf = -15.0
    local time_cf = -1.0
    local backwards_cf = -2.0
    local reward = 0
    if not isDoneTrain() then
        reward = wall_cf * wall_reward() + checkpoint_cf * checkpoint_reward() + terrain_cf * terrain_reward() + time_cf * time_reward() + backwards_cf * backwards_reward()
        return reward
    end
    return reward
end
```

NOTE: The other functions defined here are explained further in `Integrations/MarioKart-Snes/script.lua`.

## Training Script Setup

The training script in `PPO_agent.py` has many default arguments set. Ensure that the argument `--custom-integration-path` points to the folder containing custom integrations. Additionally, to change the name of the game being played, use the `--task` argument. By default, this argument is set to `MarioKart-Snes`. Additionally, hyperparameters can be directly configured from the command line. You can also specify which 'scenario' and 'data' file you want the environment to run from using the `--scenario` and `--info` flags respectively.

### Example Usage

```python
python PPO_agent.py --custom-integration-path Integrations --task MarioKart-Snes
```

