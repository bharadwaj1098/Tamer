import gym
import numpy as np 
import time
import random
import cv2
import matplotlib.pyplot as plt
import psutil
import gc
import subprocess
import warnings 
warnings.filterwarnings("ignore")
from pathlib import Path 
from collections import deque
from itertools import count
from IPython.display import clear_output
from pdb import set_trace

import torch 
import torchvision
import torchvision.utils 
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.transforms as T 
import torch.optim as optim

class Encoder(nn.Module):
    def __init__(self,):
        super(Encoder, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 1, 3)

        self.conv_bn1 = nn.BatchNorm2d(64)
        self.conv_bn2 = nn.BatchNorm2d(1)
        
        self.linear_1 = nn.Linear(64, 100)
        
    def forward(self, x):
        x = x
        x = F.max_pool2d(self.conv_bn1(self.conv1(x)), 2)
        x = F.max_pool2d(self.conv_bn1(self.conv2(x)), 2)
        x = F.max_pool2d(self.conv_bn1(self.conv2(x)), 2)
        x = F.max_pool2d(self.conv_bn2(self.conv3(x)), 2)
        x = x.view(x.size(0), -1)
        x = self.linear_1(x) #encoded states might come in "-ve" so no Relu or softmax
        return x

class Decoder(nn.Module):
    #Linear - Reshape - Upsample - Deconv + BatchNorm 
    def __init__(self):
        super(Decoder, self).__init__()

        self.linear_1 = nn.Linear(100,64)

        self.upsample_1 = nn.Upsample(scale_factor=2)
        self.upsample_2 = nn.Upsample(scale_factor=2.03)

        self.deconv_1 = nn.ConvTranspose2d(1, 64, 3)
        self.deconv_2 = nn.ConvTranspose2d(64, 64, 3)
        self.deconv_3 = nn.ConvTranspose2d(64, 3, 3)

        self.deconv_bn1 = nn.BatchNorm2d(64)
        self.deconv_bn2 = nn.BatchNorm2d(3)  

    def forward(self,x):
        x = x
        x = F.relu(self.linear_1(x)) 
        x = x.view(x.shape[0],1,8,8)
        x = self.deconv_bn1(self.deconv_1(self.upsample_1(x)))
        x = self.deconv_bn1( self.deconv_2(self.upsample_1(x)))
        x = self.deconv_bn1(self.deconv_2(self.upsample_2(x)))
        x = self.deconv_bn2(self.deconv_3(self.upsample_1(x))) 
        return x

class BufferArray():
    def __init__(self, size):
        self.memory = torch.zeros((size), dtype=torch.float32)
        self._mem_loc = 0
        self.push_count = 0

    def __len__(self):
        return self.push_count

    def push(self, tensor):
        type(self._mem_loc)
        if self._mem_loc == len(self.memory)-1:
          self._mem_loc = 0
        self.memory[self._mem_loc] = tensor.cpu()

        self._mem_loc += 1
        self.push_count += 1

    def random_sample(self, batch_size):
        rand_batch = np.random.randint(len(self.memory), size=batch_size)
        if len(rand_batch.shape) == 3:
           return torch.unsqueeze(self.memory[rand_batch], axis=1)
        return self.memory[rand_batch]
         

class BufferDeque():
    def __init__(self, size):
        self.memory = deque(maxlen=size)

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, index):
        if isinstance(index, tuple, list):
          pointer = list(self.memory)
          return [pointer[i] for i in index]
        return list(self.memory[index])

    def push(self, tensor):
        self.memory.append(tensor.cpu()) 

    def random_sample(self, batch_size):
        rand_batch = np.random.randint(len(self.memory),  size=batch_size)
        return torch.stack([self.memory[b] for b in rand_batch]) 

 

    

