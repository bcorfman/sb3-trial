import os
import pickle
import random

import comet_ml
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from comet_ml.integration.gymnasium import CometLogger
from stable_baselines3 import A2C

# Even though we don't use this class here, we should include it here so that it registers the NPuzzle environment.
from core.game import (
    NPuzzleEnv,
)


# Train or test using Q-Learning
def run_q(episodes, is_training=True, render=False):
    env = gym.make("NPuzzle-v1", render_mode="human" if render else None)

    if is_training:
        # If training, initialize the Q Table, a 5D vector: [robot_row_pos, robot_row_col, target_row_pos, target_col_pos, actions]
        q = np.zeros(
            (
                env.unwrapped.grid_rows,
                env.unwrapped.grid_cols,
                env.unwrapped.grid_rows,
                env.unwrapped.grid_cols,
                env.unwrapped.puzzle.n,
            )
        )
    else:
        # If testing, load Q Table from file.
        f = open("v1_8puzzle_solution.pkl", "rb")
        q = pickle.load(f)
        f.close()

    # Hyperparameters
    learning_rate_a = 0.9  # alpha or learning rate
    discount_factor_g = 0.9  # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
    epsilon = 1  # 1 = 100% random actions

    # Array to keep track of the number of steps per episode for the robot to find the target.
    # We know that the robot will inevitably find the target, so the reward is always obtained,
    # so we want to know if the robot is reaching the target efficiently.
    steps_per_episode = np.zeros(episodes)

    step_count = 0
    for i in range(episodes):
        if render:
            print(f"Episode {i}")

        # Reset environment at teh beginning of episode
        state = env.reset()[0]
        terminated = False

        # Robot keeps going until it finds the target
        while not terminated:
            # Select action based on epsilon-greedy
            if is_training and random.random() < epsilon:
                # select random action
                action = env.action_space.sample(env.unwrapped.puzzle.mask)
            else:
                # Convert state of [1,2,3,4] to (1,2,3,4), use this to index into the 4th dimension of the 5D array.
                q_state_idx = tuple(state)

                # select best action
                action = np.argmax(q[q_state_idx])

            # Perform action
            new_state, reward, terminated, _, _ = env.step(action)

            # Convert state of [1,2,3,4] and action of [1] into (1,2,3,4,1), use this to index into the 5th dimension of the 5D array.
            q_state_action_idx = tuple(state) + (action,)

            # Convert new_state of [1,2,3,4] into (1,2,3,4), use this to index into the 4th dimension of the 5D array.
            q_new_state_idx = tuple(new_state)

            if is_training:
                # Update Q-Table
                q[q_state_action_idx] = q[q_state_action_idx] + learning_rate_a * (
                    reward
                    + discount_factor_g * np.max(q[q_new_state_idx])
                    - q[q_state_action_idx]
                )

            # Update current state
            state = new_state

            # Record steps
            step_count += 1
            if terminated:
                steps_per_episode[i] = step_count
                step_count = 0

        # Decrease epsilon
        epsilon = max(epsilon - 1 / episodes, 0)

    env.close()

    # Graph steps
    sum_steps = np.zeros(episodes)
    for t in range(episodes):
        sum_steps[t] = np.mean(
            steps_per_episode[max(0, t - 100) : (t + 1)]
        )  # Average steps per 100 episodes
    plt.plot(sum_steps)
    plt.savefig("v1_8puzzle_solution.png")

    if is_training:
        # Save Q Table
        f = open("v1_8puzzle_solution.pkl", "wb")
        pickle.dump(q, f)
        f.close()


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
    run_q(1000, is_training=True, render=False)
    run_q(1, is_training=False, render=True)
    # Train/test using StableBaseline3
    # train_sb3()
    # test_sb3()
