import numpy as np 
import gym 
import random
import time
import os
import datetime as dt 
from pathlib import Path 
from csv import DictWriter 

import matplotlib.pyplot as plt
#%matplotlib inline
from IPython import display

import asyncio
import warnings 
warnings.filterwarnings("ignore") 

import torch 
import torchvision
import torchvision.utils 
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.transforms as T 
import torch.optim as optim

torch.manual_seed(0)

BOWLING_ACTION_MAP = {0:'NOOP', 1:'FIRE', 2:'UP', 3:'DOWN', 4:'UPFIRE', 5:'DOWNFIRE'} 


#os.environ['DISPLAY'] = ':1'

class Encoder(nn.Module):
    def __init__(self):
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

class Head_net(nn.Module):
    def __init__(self):
        super(Head_net,self).__init__()

        self.linear_1 = nn.Linear(100,16)
        self.linear_2 = nn.Linear(16,4)
    
    def forward(self, x):
        x = x 
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        return x

class Agent:
    def __init__(self, env, num_episodes, 
                encoder, head, device,
                img_dims = (3, 160, 160), ts_len=0.2):

        self.env = env
        self.img_dims = img_dims
        self.num_episodes = num_episodes
        self.ts_len = ts_len
        self.device = device  
        self.encoder = encoder
        self.head = head
        self.buffer = []

    def get_screen(self):
        screen = self.env.render(mode='rgb_array')
        screen = screen.transpose((2,0,1))
        screen = np.ascontiguousarray(screen, dtype = np.float32)/255
        screen = torch.from_numpy(screen)
        resize = T.Compose([T.ToPILImage(),
                            T.Resize((self.img_dims[1:])),
                            T.ToTensor()])
        screen = resize(screen).to(self.device).unsqueeze(0)
        return screen 
    
    def _train_(self, disp):
        state = self.env.reset() 
        state = self.get_screen(state)
        for step in range(100):
            self.env.render('rgb_array') 
            state_ts = dt.datetime.now().time() #state start time 
            network_output = self.head(self.encoder(state.to(self.device)) ) #output of the network
            action = np.argmax( network_output.detach().numpy() ) #action

            disp.show_action(action) #display_the_action_taken 
            next_state, env_reward, done, info = self.env.step(action)  
            
            now = time.time() 
            while time.time() < now + self.ts_len: 
                time.sleep(0.01)

                human_reward = disp.get_scalar_feedback()
                feedback_ts = dt.datetime.now().time() 

                if human_reward != 0: 
                    self.buffer.append([state_ts, state, env_reward, feedback_ts, human_reward])
                    #print(buffer[:]) 

            state = self.get_screen()
    
    def train(self):
        self.env.render('rgb_array') 
        from interface import Interface
        disp = Interface(action_map = BOWLING_ACTION_MAP)
        for i in range(self.num_episodes):
            self._train_(disp) 


def main():
    env = gym.make("Bowling-v0").unwrapped 
    num_episodes = 1

    encoder = Encoder()
    head_net = Head_net()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt = optim.Adam( head_net.parameters(), lr=1e-4, weight_decay = 1e-1 ) 

    encoder.load_state_dict(torch.load("auto_encoder/Type_1/encoder.pt", map_location=device))
    encoder.eval()

    for name, params in encoder.named_parameters():
        params.requires_grad = False #no updating happens to the weights while training the Tamer

    agent = Agent(env, num_episodes, encoder = encoder, head = head_net , device = device, ts_len=0.3) 

    agent.train()

if __name__ == "__main__":
    main() 



                
