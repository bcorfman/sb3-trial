import comet_ml
import gymnasium as gym
from comet_ml.integration.gymnasium import CometLogger
from stable_baselines3 import A2C, DQN

from core.game import NPuzzleEnv

env = gym.make("NPuzzle-v0")
experiment = comet_ml.Experiment(project_name="npuzzle", workspace="bcorfman")
env = CometLogger(env, experiment)

model = A2C("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=1000, progress_bar=True)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(10):
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = vec_env.step(action)
    print(action, str(obs))
    vec_env.render("human")
env.close()
experiment.end()
