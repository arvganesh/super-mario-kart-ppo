# kart-agents

This repo describes how integrate new ROMs into [stable-retro](https://github.com/MatPoliquin/stable-retro) + other setup guidelines.

An example integration of Super Mario Kart on the SNES.

This guide is essentially a distilled version of [this](https://retro.readthedocs.io/en/latest/integration.html#).

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
  - `scenario.json` – Describes reward function and 'done' condition.
  - `metadata.json` – Specifies which `.state` file corresponds to the start state.
  - `rom.sfc` – ROM file for Mario Kart SNES (must be named `rom`)
  - `rom.sha` – SHA1 checksum for ROM file. Used to identify the ROM. (must be named `rom`). Used [this](https://emn178.github.io/online-tools/sha1_checksum.html) to create it.
- `sample_integration.py`: Shows how to integrate new roms into stable-retro without modifying the stable-retro pacakge.

More detailed explanations of the files inside `Integrations/MarioKart-Snes` can be found [here](https://retro.readthedocs.io/en/latest/integration.html#game-integration).

### Integrating new games

To integrate new ROMs, you can use the [Gym Integration UI](https://github.com/openai/retro/releases/tag/f347d7e). However, the application is fairly buggy on my Mac, so I had to do most things it's supposed to do by hand. The one exception to this was creating the file `Level1.state`.

Integration in this context means finding RAM locations of important variables using them to define the reward function and 'done' condition as given by the environment.

Note: stable-retro already [contains many integrations](https://github.com/arvganesh/stable-retro/tree/master/retro/data/stable/) for games, you will just need to [import the ROMs](https://retro.readthedocs.io/en/latest/getting_started.html?highlight=retro.import#importing-roms).

*IMPORTANT: [Must read if you want to integrate new ROMs.](https://retro.readthedocs.io/en/latest/integration.html#game-integration)*

From my experience, the most time-consuming part of integration is finding RAM locations of important variables. However, many popular games already have their RAMs "mapped" out. [Here](https://datacrystal.romhacking.net/wiki/Super_Mario_Kart:RAM_map) is an example mapping for Super Mario Kart which saved me a lot of time. Pay close attention to signed vs. unsigned, the width of the variable in bytes, and endianness. For the SNES, [16-bit and 24-bit values are stored in little-endian](https://ersanio.gitbook.io/assembly-for-the-snes/the-fundamentals/endian), but this probably varies depending on the game console. This [blog post](https://www.videogames.ai/2019/01/29/Setup-OpenAI-baselines-retro.html) also has some helpful information and tutorials on integration.

## Environment Design
### Start State
`Level1.state` stores the default state in which episodes begin. The name of this file can be pretty much anything as long as it's specified in `metadata.json`.

I used the [Gym Integration UI](https://github.com/openai/retro/releases/tag/f347d7e) to create the start state file. It may also be possible to use other emulators to do this, but I haven't tried.

I've specified it to be the start of a race on Time Trials mode playing as Mario on Mario Circuit, but this is totally customizable.

<img width="256" alt="image" src="https://user-images.githubusercontent.com/21336191/225757384-95724610-adbc-461e-9cbb-19b99ba7e7a4.png">

### Action Space
Here is the action space for SNES games, corresponding to the 12 buttons on the controller. More on [MultiBinary](https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.MultiBinary).
```python
>>> print(env.action_space)
MultiBinary(12)
```
[Emulator Definition](https://github.com/MatPoliquin/stable-retro/blob/master/cores/snes.json):
```json
"buttons": ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]
```
### Observation Space
Here is the observation space for SNES games. In words, a `244x256x3` tensor with values in the range `[0, 255]` of type `uint8`. Basically, just the raw pixels.
```python
>>> print(env.observation_space)
Box(0, 255, (224, 256, 3), uint8)
```

### Reward Function, Done Condition, Time Penalty
`data.json` contains locations of important variables. [See this](https://retro.readthedocs.io/en/latest/integration.html#variable-locations-data-json).
```json
{
    "info": {
      "gameover": {
        "address": 8261825,
        "type": "|u1"
      },
      "speed": {
        "address": 8261866,
        "type": "<s2"
      }
    }
}
```
- `gameover` is at address `8261825` and is an unsigned single byte quantity. Endianness does not need to be specified for 1-byte quantities, hence `|`.

- `speed` is at address `8261866` and is a signed 2-byte integer in little-endian (denoted by `<`).

`scenario.json` uses these variables to define a reward function and done condition. [See this](https://retro.readthedocs.io/en/latest/integration.html#scenario-scenario-json).
```json
{
    "done": {
      "condition": "all",
      "variables": {
        "gameover": {
          "op": "equal",
          "reference": 133
        }
      }
    },
    "reward": {
      "variables": {
        "speed": {
          "reward": 1.0
        }
      },
      "time": {
        "penalty": 1.0
      }
    }
}
```

`done`: the game ends when `gameover == 133`, indicating that we have completed the last lap. This is based on how the Super Mario Kart developers defined it.

`reward`: I've specified that the reward should come from the `speed` variable. Finally, at time step, we should have a time penalty of `1.0`.

The definition of the reward / time penalty / done condition is much more customizable than I've shown here. More information on the specifics of this definition can be found [here](https://retro.readthedocs.io/en/latest/integration.html#scenario-scenario-json).

This is just one example of a simple definition for the purposes of demonstration.

