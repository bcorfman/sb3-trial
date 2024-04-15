from gymnasium.envs.registration import register

register(
    id="EightPuzzle-v0", entry_point="core.game:EightPuzzleEnv", max_episode_steps=100
)
