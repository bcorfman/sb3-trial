import gymnasium as gym
import numpy as np
import pygame
from gymnasium.spaces import Box, Discrete

from core.games import Moves, NPuzzle, TicTacToe
from core.util import _manhattan_distance


class NPuzzleEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, n=8, render_mode=None):
        self.n = n
        self.render_mode = render_mode
        self.puzzle = NPuzzle(self.n)
        self.action_space = Discrete(len(Moves))
        self.observation_space = Box(
            low=0,
            high=n,
            shape=(self.puzzle.side, self.puzzle.side),
            dtype=np.int8,
        )
        self.last_reward = 0

    def reset(self, seed=None, options=None):
        """Gymnasium-required function (and parameters) to reset the environment."""
        super().reset(
            seed=seed
        )  # gym requires this call to control randomness and reproduce scenarios.

        # Reset the N-Puzzle itself.
        self.puzzle.reset()

        # Construct the observation state.
        obs = self._get_obs()

        # Additional info to return, for debugging or other purposes.
        info = {}

        # Render environment
        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action):
        """Gymnasium-required function (and parameters) to perform an action."""
        self.puzzle.move(action)  # take a single action

        # Determine reward and termination
        reward = (
            self._get_reward() - self.last_reward
        )  # difference between this reward and last reward
        terminated = self.puzzle.is_goal_state()
        if terminated:
            print("GOAL")

        # Construct the observation state.
        observation = self._get_obs()

        # Additional info to return, for debugging or other purposes.
        info = {}

        # Render the environment.
        if self.render_mode == "human":
            print(Moves(action))
            self.render()

        # return Gymnasium-required parameters. 4th param (truncated) is
        # not used for this environment.
        return (
            observation,
            reward,
            terminated,
            False,
            info,
        )

    def render(self):
        self.puzzle.render()

    def _sum_tile_distances(self):
        return sum(
            (
                _manhattan_distance(self.puzzle.tile_loc(i), self.puzzle.goal_loc(i))
                for i in range(self.n)
            )
        )

    def _get_obs(self):
        return self.puzzle.field

    def _get_reward(self):
        return 1000 if self.puzzle.is_goal_state() else -1 * self._sum_tile_distances()


class TicTacToeEnv(gym.Env):
    def __init__(self, frames_per_second=60):
        self.game = TicTacToe()
        self.action_space = Discrete(9)
        self.observation_space = Discrete(len(self.game.valid_states))
        self.screen = None
        self.frames_per_second = frames_per_second
        self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        self.game.reset()
        obs = self.get_obs()
        info = {}
        return obs, info

    def step(self, action):
        self.game.make_move(action)
        winner = self.game.check_winner()
        reward = 0
        done = False
        if winner is not None:
            if winner == self.game.X:
                reward = 1
            elif winner == self.game.O:
                reward = -1
            done = True
        return self.get_obs(), reward, done, {}

    def get_obs(self):
        return self.game.valid_states[self.game.board]

    def action_mask(self):
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
