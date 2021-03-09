import gym
import numpy as np 
import time
import random
import cv2
from matplotlib import pyplot as plt
import warnings 
warnings.filterwarnings("ignore") 

import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.transforms as T 
import torch.optim as optim 

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        
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
        x = F.relu(self.linear_1(x))
        return x

class decoder(nn.Module):
    #Linear - Reshape - Upsample - Deconv + BatchNorm 
    def __init__(self):
        super(decoder, self).__init__()

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
'''
class policynet(nn.Module):
    def __init__(self):
        super(policynet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 1, 3)

        self.conv_bn1 = nn.BatchNorm2d(64)
        self.conv_bn2 = nn.BatchNorm2d(1)

        self.linear_1 = nn.Linear(64, 100)
        self.linear_2 = nn.Linear(100,16)
        self.linear_3 = nn.Linear(16,4)

    def forward(self, x):
        x = x
        x = F.max_pool2d(self.conv_bn1(self.conv1(x)), 2) 
        x = F.max_pool2d(self.conv_bn1(self.conv2(x)), 2) 
        x = F.max_pool2d(self.conv_bn1(self.conv2(x)), 2) 
        x = F.max_pool2d(self.conv_bn2(self.conv3(x)), 2) 
        x = x.view(x.size(0), -1)
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x = F.relu(self.linear_3(x))
        return x
'''

class Random_Agent:
    def __init__(self, env, device):

        self.current_step = 0
        self.env = gym.make(env)
        self.env.reset()
        self.current_screen = None
        self.done = False
        self.device = device
        self.action_space = self.env.action_space.n

    def reset(self):
        self.env.reset()
        self.current_screen = None
    
    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)
    
    def num_actions_available(self):
        return self.env.action_space.n
    
    def select_action(self):
        action = random.randrange(self.action_space)
        action = torch.tensor([action]).to(device)
        return action
    
    def just_starting(self):
        return self.current_screen is None

    def take_action(self, action):        
        _, reward, self.done, _ = self.env.step(action.item())
        #return torch.tensor([reward], device=self.device)
        return self.done

    def get_state(self):
        if self.just_starting() or self.done:
            self.current_screen = self.get_processed_screen()
            black_screen = torch.zeros_like(self.current_screen)
            return black_screen
        else:
            s1 = self.current_screen
            s2 = self.get_processed_screen()
            self.current_screen = s2
            return s2 - s1
    
    def get_processed_screen(self):
        screen = self.render('rgb-array').transpose((2, 0, 1)) 
        screen = self.transform_screen(screen)
        return screen  

    def transform_screen(self, screen):
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255 
        screen = torch.numpy(screen)

        resize = T.Compose([
        T.ToPILImage()
        ,T.Resize((160, 160))
        ,T.ToTensor()
        ])

        screen = resize(screen).unsqueeze(0).to(self.device)
        return screen 

if __name__ = '__main__':

    no_episodes = 100000
    count = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    e = encoder()
    d = decoder()
    agent = Random_Agent("Bowling-v0", device)

    for episode in range(no_episodes):
        agent.reset()
        state = agent.get_state()
        for step in range(count):
            action = agent.select_action()
            '''
            having problem writing thew loss functionaETQR1R:WQ
            
            '''
            if agent.take_action(action) = True:
                agent.close()
            else:
                next_state = agent.get_state()
                state = next_state
            