import gym
import numpy as np 
import time
import random
import cv2
import matplotlib.pyplot as plt
from collections import namedtuple
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

        self.optimizer = optim.Adam(self.parameters(), lr= 0.001)  
        
    def forward(self, x):
        x = x
        x = F.max_pool2d(self.conv_bn1(self.conv1(x)), 2)
        x = F.max_pool2d(self.conv_bn1(self.conv2(x)), 2)
        x = F.max_pool2d(self.conv_bn1(self.conv2(x)), 2)
        x = F.max_pool2d(self.conv_bn2(self.conv3(x)), 2)
        x = x.view(x.size(0), -1)
        x = self.linear_1(x) #encoded states might come in "-ve" so no Relu or softmax
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

        self.optimizer = optim.Adam(self.parameters(), lr= 0.001)

    def forward(self,x):
        x = x
        x = F.relu(self.linear_1(x))  
        x = x.view(x.shape[0],1,8,8)
        x = self.deconv_bn1(self.deconv_1(self.upsample_1(x)))
        x = self.deconv_bn1( self.deconv_2(self.upsample_1(x)))
        x = self.deconv_bn1(self.deconv_2(self.upsample_2(x)))
        x = self.deconv_bn2(self.deconv_3(self.upsample_1(x))) 
        return x

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

        self.optimizer = optim.Adam(self.parameters(), lr= 0.001)

    def forward(self, x):
        x = x
        x = F.max_pool2d(self.conv_bn1(self.conv1(x)), 2) 
        x = F.max_pool2d(self.conv_bn1(self.conv2(x)), 2) 
        x = F.max_pool2d(self.conv_bn1(self.conv2(x)), 2) 
        x = F.max_pool2d(self.conv_bn2(self.conv3(x)), 2) 
        x = x.view(x.size(0), -1)
        x = self.linear_1(x)
        x = self.linear_2(x)
        x = self.linear_3(x)
        return x

Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
    
    def push(self, experience):
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.push_count % self.capacity] = experience
        self.push_count +=1 
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size) 
    
    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

class EpsilonGreedy():
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end 
        self.decay = decay 
    
    def get_exploration_rate(self, current_step):
        rate = self.end + (self.start - self.end) * \
            math.exp(-1. * current_step * self.decay)
        return rate

class Agent():
    def __init__(self, strategy,num_actions,device):
        self.current_step = 0
        self .strategy = strategy
        self.num_actions = num_actions
        self.device = device 
    
    def select_action(self, state, policy_net):
        rate = strategy.get_exploration_rate(self.current_step)
        self.current_step += 1
        if rate > random.random():
            return random.randrange(self.num_actions)
        else:
            with torch.no_grad():
                return policy_net(state).argmax(dim =1).item()

class env_manager():
    def __init__(self, device):
        self.env = gym.make("Bowling-v0").unwrapped
        self.env.reset()
        self.device = device 
        self.done = False
    
    def get_state(self):
        screen = self.render(mode = "rgb_array").transpose((2,0,1))
        screen = np.ascontiguousarray(screen, dtype = np.float32)/255
        screen = torch.from_numpy(screen)
        resize = T.Compose([T.ToPILImage(),
                            T.Resize( ??, interpolation = Image.CUBIC),
                            T.ToTensor()])
        screen = resize(screen).unsqueeze(0).to(self.device) 
        

'''
if __name__ == '__main__':

    no_episodes = 100000
    count = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder()
    decoder = decoder()
    agent = Random_Agent(device)
    decoder_optimizer = optim.RMSprop(decoder.parameters(), lr=1e-4, alpha=0.99, eps=1e-8, 
                                weight_decay=0.0005, momentum=0.9)
    encoder_optimizer = optim.RMSprop(decoder.parameters(), lr=1e-4, alpha=0.99, eps=1e-8, 
                                weight_decay=0.0005, momentum=0.9) 

    for episode in range(no_episodes):
        agent.reset()
        state = agent.get_state()
        state = state.cuda()
        for step in range(count):
            action = agent.select_action()
            encoder_output = encoder(state.cuda())
            decoder_output = decoder(encoser_output)
            diff = (state - decoder_output)**2
            diff = np.array(diff)
            summed = np.sum(diff)
            loss_contrastive = torch.tensor(summed/(160**2) )
            loss_contrastive.backward()
            optimizer.step()

            if agent.take_action(action) == True:
                agent.close()
            else:
                next_state = agent.get_state()
                state = next_state.cuda() 

environment = "Bowling-v0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''