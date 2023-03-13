import retro

# Random Agent
def main():
    env = retro.make(game='Airstriker-Genesis')
    obs = env.reset()
    while True:
        ob, rew, done, trunc, info  = env.step(env.action_space.sample())
        env.render()
        if done:
            obs = env.reset()
    env.close()

if __name__ == "__main__":
    main()