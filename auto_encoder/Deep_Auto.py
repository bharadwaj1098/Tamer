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

env = gym.make("Bowling-v0").unwrapped 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_actions = env.action_space.n
no_of_episodes = 5000
batch_size = 32 
img_dims = (3, 160, 160)
agent_histroy = 2
buffer_size = (20000,) + img_dims  

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


def select_action():
    action = torch.tensor([[random.randrange(n_actions)]], device = device, dtype = torch.long)
    return action

def get_screen():
    screen = env.render(mode = "rgb_array").transpose((2,0,1))
    screen = np.ascontiguousarray(screen, dtype = np.float32)/255
    screen = torch.from_numpy(screen)
    resize = T.Compose([T.ToPILImage(),
                        T.Resize((img_dims[1:])),
                        T.ToTensor()])
    screen = resize(screen).to("cuda")#.unsqueeze(0) 
    return screen ##Returns Grayscale, PILIMAGE, TENSOR 

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map 

if __name__ == "__main__":
    # buffer = BufferDeque(buffer_size[0])
    buffer = BufferArray(buffer_size)
    loss_fn = nn.MSELoss(reduction = 'mean')
    loss_history = []

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    opt = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4, weight_decay = 1e-1 )


    def optimize():
        # Sample batch and preprocess
        state_batch = buffer.random_sample(batch_size)

        # Move state_batch to GPU and run through autoencoder
        state_batch = state_batch.to(device)
        output_batch = decoder(encoder(state_batch))

        # Compute loss
        loss = loss_fn(state_batch, output_batch).cpu()

        # Optimize
        opt.zero_grad()
        loss.backward()
        opt.step()

        return loss
    
    for episode in range(no_of_episodes):
        env.reset()
        state = get_screen()

        done = False
        for step in count():
            action = select_action() 
            _, _, done, _ = env.step(action.item())

            if not done:
                next_state = get_screen()
            else:
                next_state = None

            # Store trasition 
            buffer.push(state)

            # Set next state 
            state = next_state

            # Optimization/training
            # Dont start training until there are enough samples in memory
            if len(buffer.memory) > 1000:
                # Only train every certain number of steps
                if step % 16 == 0:
                    loss = optimize()
                    loss_history.append(loss.cpu()) 
                    print("Episode: {} Step: {} Loss: {:5f} GPU Memory: {} RAM: {:5f} Buffer Size: {}".format(
                                        episode, step, loss, get_gpu_memory_map()[0]/1000, 
                                        psutil.virtual_memory().available /  1024**3, len(buffer)
                                        ))
                    clear_output(wait=True)

            # Check if down
            if done:
                break 
    
    torch.save(encoder.state_dict(), "encoder_one_image.pt")
    torch.save(decoder.state_dict(), "decoder_one_image.pt")
    


