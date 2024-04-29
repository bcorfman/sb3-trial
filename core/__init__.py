from gymnasium.envs.registration import register

register(id="NPuzzle-v1", entry_point="core.env:NPuzzleEnv")
register(id="TicTacToe-v1", entry_point="core.env:TicTacToeEnv")
