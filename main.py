import torch 
import numpy as np
import gym
import random
from collections import namedtuple

env = gym.make('Bowling-v0').unwrapped 
'''

Results = namedtuple(
    'Results',
    ('original', 'output')
)

class Buffer():
    def __init__(self, capacity):
        self.capacity = capacity 
        self.memory = []
        self.push_count = 0
    
    def push(self, Results):
        if len(self.memory)< self.capacity:
            self.memory.append(Results)
        else:
            self.memory[self.push_count % self.capacity] = Results 
        self.push_count += 1

    def random_sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size 
    
    def empty(self):
        self.memory.clear()

buffer = Buffer(100)

for i in range(10):
    for j in range(10):
        buffer.push((i, j))
    k = buffer.random_sample(3)
    print(k) 
    buffer.empty()

'''
def MSE(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    z = float(imageA.shape[0] * imageA.shape[1])
    err /= z
    return z 

def new(i):
    i = np.array(i.squeeze(0).permute(1, 2, 0)) 
    return i

i = torch.rand(1, 3, 160, 160)
j = torch.rand(1, 3, 160, 160)

i = new(i)
j = new(j)

j = MSE(i, j)

print(j)