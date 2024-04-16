from gymnasium.envs.registration import register

register(id="NPuzzle-v0", entry_point="core.game:NPuzzleEnv", max_episode_steps=1000)
