import numpy as np 
import gym 
import random
import time
import datetime as dt 
from pathlib import Path 
from csv import DictWriter 

import asyncio 

import torch 
import torchvision
import torchvision.utils 
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.transforms as T 
import torch.optim as optim

from interface import Interface  

torch.manual_seed(0)

env = gym.make("Bowling-v0").unwrapped 
img_dims = (3, 160, 160) 
BOWLING_ACTION_MAP = {0:'NOOP', 1:'FIRE', 2:'UP', 3:'DOWN', 4:'UPFIRE', 5:'DOWNFIRE'} 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
buffer = []
ts_len = 0.2 #len of time_step to train 
Done = False 
env.reset() # ressting_the_environment

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

encoder = Encoder().to(device)
head_net = Head_net().to(device)

opt = optim.Adam( head_net.parameters(), lr=1e-4, weight_decay = 1e-1 ) 

encoder.load_state_dict(torch.load("auto_encoder/Type_1/encoder.pt", map_location=device))
encoder.eval()

for name, params in encoder.named_parameters():
    params.requires_grad = False #no updating happens to the weights while training the Tamer



class Agent:
    def __init__(self, env, num_episodes, 
                encoder, head, device, display, 
                img_dims = (3, 160, 160), ts_len=0.2):

        self.env = env
        self.img_dims = img_dims
        self.num_episodes = num_episodes
        self.ts_len = ts_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        self.encoder = encoder().to(self.device)
        self.head = head().to(self.device)
        self.buffer = []

    def get_screen(self):
        screen = self.env.render(mode = "rgb_array").transpose((2,0,1))
        screen = np.ascontiguousarray(screen, dtype = np.float32)/255
        screen = torch.from_numpy(screen)
        resize = T.Compose([T.ToPILImage(),
                            T.Resize((self.img_dims[1:])),
                            T.ToTensor()])
        screen = resize(screen).to(device).unsqueeze(0)
        return screen 
    
    def train(self, disp):
        state = self.get_screen()
        for step in count():
            state_ts = dt.datetime.now().time() #state start time 
            network_output = self.head(self.encoder(state.to(device)) ) #output of the network
            action = np.argmax( network_output.detach().numpy() ) #action

            disp.show_action(action) #display_the_action_taken 
            _, env_reward, done, info = self.env.step(action)  
            
            now = time.time() 
            while time.time() < now + self.ts_len: 
                time.sleep(0.01)

                human_reward = disp.get_scalar_feedback()
                feedback_ts = dt.datetime.now().time() 

                if human_reward != 0:
                    self.buffer.append([state_ts, state, env_reward, feedback_ts, human_reward])
                    break 



                
