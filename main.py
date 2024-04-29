import os
import pickle
import random

import comet_ml
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from comet_ml.integration.gymnasium import CometLogger
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3 import A2C

# Even though we don't use this class here, we should include it here so that it registers the NPuzzle environment.
from core.env import (
    NPuzzleEnv,
)

# For Tic-Tac-Toe, the state space would be (row, col, char[X or O], and number_of_moves_taken)
# For N-Puzzle, the state space would be a stack of board_matrices (2D_square), and the dimension of the stack is the number of valid N-puzzle configurations.


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
    model.learn(total_timesteps=10000, progress_bar=True)
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
    # run_q(1000, is_training=True, render=False)
    # run_q(1, is_training=False, render=True)
    # Train/test using StableBaseline3
    train_sb3()
    test_sb3()

    def mask_fn(env: gym.Env) -> np.ndarray:
        # Uncomment to make masking a no-op
        # return np.ones_like(env.action_mask)
        return env.valid_action_mask()

    env = gym.make("TicTacToe-v1")
    env = ActionMasker(env, mask_fn)
    model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1)
    model.learn(10000, progress_bar=True)

    evaluate_policy(model, env, n_eval_episodes=100, warn=False)

    model.save("ppo_mask")
    del model  # remove to demonstrate saving and loading

    # model = MaskablePPO.load("ppo_mask")
    # obs, _ = env.reset()
    # env.unwrapped.game.player = 1
    # while True:
    #    # Retrieve current action mask
    #    action_masks = get_action_masks(env)
    #    action, _states = model.predict(obs, action_masks=action_masks)
    #    obs, reward, terminated, truncated, info = env.step(action)
    #    print(env.unwrapped.game)
    #    if terminated:
    #        break
