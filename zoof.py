import random

import gymnasium as gym
import numpy as np
import pygame
from gymnasium.spaces import Box, Discrete
from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks


class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.player = 1

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.player = random.choice([-1, 1])  # Randomly choose the starting player

    @property
    def valid_moves(self):
        moves = []
        for row in range(3):
            for col in range(3):
                if self.board[row, col] == 0:
                    moves.append(row * 3 + col)
        return moves

    def make_move(self, row, col):
        if self.board[row, col] == 0:
            self.board[row, col] = self.player
            self.player = -self.player
            return True
        return False

    def check_winner(self):
        for i in range(3):
            if np.sum(self.board[i, :]) == 3 or np.sum(self.board[:, i]) == 3:
                print("WIN")
                return 1
            if np.sum(self.board[i, :]) == -3 or np.sum(self.board[:, i]) == -3:
                print("LOSS")
                return -1
        if (
            np.sum(np.diag(self.board)) == 3
            or np.sum(np.diag(np.fliplr(self.board))) == 3
        ):
            print("WIN")
            return 1
        if (
            np.sum(np.diag(self.board)) == -3
            or np.sum(np.diag(np.fliplr(self.board))) == -3
        ):
            print("LOSS")
            return -1
        if np.count_nonzero(self.board) == 9:
            print("TIE")
            return 0
        return None

    def render(self, mode=None):
        if mode == "ansi":
            print(str(self))

    def __str__(self):
        output = "-------------\n"
        for i in range(3):
            output += "| "
            for j in range(3):
                if self.board[i, j] == 1:
                    output += "X | "
                elif self.board[i, j] == -1:
                    output += "O | "
                else:
                    output += "  | "
            output += "\n-------------\n"
        return output


class TicTacToeEnv(gym.Env):
    def __init__(self, frames_per_second=60):
        self.game = TicTacToe()
        self.action_space = Discrete(9)
        self.observation_space = Box(low=-1, high=1, shape=(3, 3, 9), dtype=int)
        self.screen = None
        self.frames_per_second = frames_per_second
        self.clock = pygame.time.Clock()

    def reset(self):
        self.game.reset()
        return self.game.board

    def step(self, action):
        row, col = action // 3, action % 3
        valid_move = self.game.make_move(row, col)
        winner = self.game.check_winner()
        reward = 0
        done = False
        if winner is not None:
            if winner == self.game.player:
                reward = 1
            elif winner == -self.game.player:
                reward = -1
            done = True
        elif not valid_move:
            reward = -1
            done = True
        return self.game.board, reward, done, {}

    def action_masks(self):
        return [i for i, valid in enumerate(self.game.valid_moves) if valid]

    def render(self, mode="human"):
        if mode == "human":
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((300, 300))
                pygame.display.set_caption("Tic-Tac-Toe")
            self.screen.fill((255, 255, 255))
            for i in range(1, 3):
                pygame.draw.line(
                    self.screen, (0, 0, 0), (0, i * 100), (300, i * 100), 2
                )
                pygame.draw.line(
                    self.screen, (0, 0, 0), (i * 100, 0), (i * 100, 300), 2
                )
            for i in range(3):
                for j in range(3):
                    if self.game.board[i, j] == 1:
                        pygame.draw.line(
                            self.screen,
                            (255, 0, 0),
                            (j * 100 + 20, i * 100 + 20),
                            (j * 100 + 80, i * 100 + 80),
                            2,
                        )
                        pygame.draw.line(
                            self.screen,
                            (255, 0, 0),
                            (j * 100 + 80, i * 100 + 20),
                            (j * 100 + 20, i * 100 + 80),
                            2,
                        )
                    elif self.game.board[i, j] == -1:
                        pygame.draw.circle(
                            self.screen,
                            (0, 0, 255),
                            (j * 100 + 50, i * 100 + 50),
                            30,
                            2,
                        )
            pygame.display.flip()
            self.clock.tick(self.frames_per_second)
        else:
            super().render(mode=mode)

    def close(self):
        if self.screen is not None:
            pygame.quit()


if __name__ == "__main__":
    env = InvalidActionEnvDiscrete(dim=4, n_invalid_actions=2)
    model = MaskablePPO("MlpPolicy", env, gamma=0.4, seed=32, verbose=1)
    model.learn(5000, progress_bar=True)

    evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=90, warn=False)

    model.save("ppo_mask")
    del model  # remove to demonstrate saving and loading

    model = MaskablePPO.load("ppo_mask")
    obs, _ = env.reset()
    while True:
        # Retrieve current action mask
        action_masks = get_action_masks(env)
        action, _states = model.predict(obs, action_masks=action_masks)
        obs, reward, terminated, truncated, info = env.step(action)
