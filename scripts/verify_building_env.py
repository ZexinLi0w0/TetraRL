"""Verify SustainGym BuildingEnv can be imported, instantiated, reset, and stepped."""

import numpy as np
from sustaingym.envs.building import BuildingEnv
from sustaingym.envs.building.utils import ParameterGenerator

params = ParameterGenerator(
    building="ApartmentHighRise",
    weather="Hot_Dry",
    location="Albuquerque",
    episode_len=288,
)
print(f"Generated params for ApartmentHighRise, keys: {len(params)}")

env = BuildingEnv(params)
print(f"BuildingEnv created: {type(env).__name__}")
print(f"action_space: {env.action_space}")
print(f"observation_space: {env.observation_space}")
print(f"reward_space (if MO): {getattr(env, 'reward_space', 'N/A')}")

obs, info = env.reset(seed=0)
print(f"obs shape: {np.array(obs).shape if hasattr(obs, '__len__') else type(obs)}")

for i in range(3):
    action = env.action_space.sample()
    result = env.step(action)
    print(f"step {i}: result has {len(result)} components, reward={result[1]:.4f}" if isinstance(result[1], (int, float)) else f"step {i}: result has {len(result)} components, reward={result[1]}")

env.close()
print("SustainGym Building env smoke OK")
