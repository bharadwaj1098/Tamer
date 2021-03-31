import gym
import numpy as np 
import time
import random
import cv2
import os 
import pathlib 
import matplotlib.pyplot as plt
#import psutil
import gc
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

from Autoencoder import Encoder, Decoder, BufferArray  

'''
This code is to make autoencoder train till episode is DONE.
each state is 1 RGB image.
'''

def main():

    env = gym.make("Bowling-v0").unwrapped 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_actions = env.action_space.n
    no_of_episodes = 1 #2000
    batch_size = 32 
    img_dims = (3, 160, 160) 
    buffer_size = (20000,) + img_dims  

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

    # buffer = BufferDeque(buffer_size[0])
    buffer = BufferArray(buffer_size)
    loss_fn = nn.MSELoss(reduction = 'mean')
    loss_history = []

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    opt = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-4, weight_decay = 1e-1 )

    def image_visual():
        batch = 3
        plt.figure(figsize = (25,30))
        i = buffer.random_sample(batch).to(device)
        o = decoder(encoder(i))

        interleaved_shape = (len(i) + len(o),) + i.shape[1:]
        interleaved = torch.empty(interleaved_shape)
        interleaved[0::2] = i
        interleaved[1::2] = o

        img_grid = torchvision.utils.make_grid(interleaved, nrow = 2)
        plt.imshow(img_grid.permute(1,2,0) )
        plt.savefig(os.path.join(pathlib.Path().absolute() , 'Type_1/result'))

    def loss_visual(x):
        plt.plot(x)
        plt.show()
        plt.savefig(os.path.join(pathlib.Path().absolute() , 'Type_1/loss_history'))

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
                    '''
                    print("Episode: {} Step: {} Loss: {:5f} GPU Memory: {} RAM: {:5f} Buffer Size: {}".format(
                                        episode, step, loss, get_gpu_memory_map()[0]/1000, 
                                        psutil.virtual_memory().available /  1024**3, len(buffer)
                                        ))
                    clear_output(wait=True)
                    '''
            # Check if down
            if done:
                break 

    image_visual()
    loss_visual(loss_history) 

    path_encoder = os.path.join(pathlib.Path().absolute() , 'Type_1/encoder.pt') 
    path_decoder = os.path.join(pathlib.Path().absolute() , 'Type_1/decoder.pt') 

    torch.save(encoder.state_dict(), path_encoder)
    torch.save(decoder.state_dict(), path_decoder)

if __name__ == "__main__":
    main()
