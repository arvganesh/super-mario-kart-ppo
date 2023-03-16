import retro
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Directory of the script

# Integrates a new ROM into stable-retro without explicitly adding it to the package.
def main():
    # Integration Code
    NEW_INTEGRATIONS_DIR = os.path.join(SCRIPT_DIR, "Integrations")
    INTEGRATION_NAME = "MarioKart-Snes"

    retro.data.Integrations.add_custom_path(NEW_INTEGRATIONS_DIR) # Add folder containing new integrations to path.
    print(INTEGRATION_NAME in retro.data.list_games(inttype=retro.data.Integrations.ALL)) # Should print 'True' if integration suceeded.

    # Create environment
    env = retro.make(INTEGRATION_NAME, inttype=retro.data.Integrations.ALL) # INTEGRATION_NAME should match a folder name within NEW_INTEGRATIONS_DIR
    observation, info = env.reset()

    for _ in range(1000):
        action = env.action_space.sample() # Random Agent
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()

if __name__ == "__main__":
    main()