import torch 
import numpy as np
import gym
import random
from collections import namedtuple

env = gym.make('Bowling-v0').unwrapped 

actions  = env.get_action_meanings()



print(actions)
