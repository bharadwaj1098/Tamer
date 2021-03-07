import torch as T 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 

import warnings 
warnings.filterwarnings("ignore") 

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
    '''
    Linear - Reshape - Upsample - Deconv + BatchNorm 
    '''
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

class Random_Agent:
    def __init__(self, env):
        self.state_size_h = environment.observation_space.shape[0]
        self.state_size_w = environment.observation_space.shape[1]
        self.state_size_c = environment.observation_space.shape[2]

        # Activation size for breakout env. Used as output size in network
        self.action_size = environment.action_space.n

        # Image pre process params
        self.target_h = 160  # Height after process
        self.target_w = 160  # Widht after process 

