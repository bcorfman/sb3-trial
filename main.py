import os

import comet_ml
import gymnasium as gym
from comet_ml.integration.gymnasium import CometLogger
from stable_baselines3 import A2C, DQN

from core.game import NPuzzleEnv


# Train using StableBaselines3 and Comet for experiment tracking and logging.
def train_sb3():
    # Where to store trained model file(s)
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    # Make the NPuzzle env. Monitor the experiment with Comet.
    env = gym.make("NPuzzle-v1", render_mode="human")
    experiment = comet_ml.Experiment(project_name="npuzzle", workspace="bcorfman")
    env = CometLogger(env, experiment)

    # Use Advantage Actor Critic (A2C) algorithm.
    # Use MlpPolicy for observation space 1D vector.
    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10000)  # , progress_bar=True)
    filename = os.path.join(model_dir, "a2c_1000")
    model.save(filename)


# Test using StableBaselines3.
def test_sb3(render=True):
    env = gym.make("NPuzzle-v1")

    # Load model
    model_dir = "models"
    filename = os.path.join(model_dir, "a2c_1000")
    model = A2C.load(filename, env=env)

    # Run a test
    obs = env.reset()[0]
    terminated = False
    while True:
        action, _ = model.predict(
            observation=obs, deterministic=True
        )  # Turn on deterministic, so predict always returns the same behavior
        obs, _, terminated, _, _ = env.step(int(action))

        if terminated:
            break


if __name__ == "__main__":
    # Train/test using StableBaseline3
    train_sb3()
    # test_sb3()
