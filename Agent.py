import numpy as np 
import gym 
import random 
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

env = gym.make("Bowling-v0").unwrapped
img_dims = (3, 160, 160) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder().to(device)
head_net = Head_net().to(device)

opt = optim.Adam(list(encoder.parameters()) + list(Head_net.parameters()), lr=1e-4, weight_decay = 1e-1 ) 

encoder.load_state_dict(torch.load("auto_encoder/Type_1/encoder.pt", map_location=device))
encoder.eval()

def get_screen():
    screen = env.render(mode = "rgb_array").transpose((2,0,1))
    screen = np.ascontiguousarray(screen, dtype = np.float32)/255
    screen = torch.from_numpy(screen)
    resize = T.Compose([T.ToPILImage(),
                        T.Resize((img_dims[1:])),
                        T.ToTensor()])
    screen = resize(screen).to(device).unsqueeze(0) 
    return screen ##Returns Grayscale, PILIMAGE, TENSOR

def optimize(state):
    encoder_out = encoder(state.to(device) )
    head_net_output = head_net(encoder_out)
    action = np.argmax( head_net_output.detach().numpy() )

    #opt.zero_grad()
    #loss.backward()
    #opt.step()

j = 0
k = 0
for i in range(1):
    state = get_screen()
    action = optimize(state)
    




















'''
async def div(inrange, div_by):
    print("finding the number in range {} divisible by {}".format(inrange, div_by))
    located = []
    for i in range(inrange):
        if i % div_by ==0:
            located.append(i)
        if i % 50000 == 0:
            await asyncio.sleep(0.0000001)
    print("Done with nums in range {} divisible by {}".format(inrange, div_by) )
    return located 
async def main():
    divs1 = loop.create_task(div(50800, 34113))
    divs2 = loop.create_task(div(100052, 3210))
    divs3 = loop.create_task(div(500, 3))
    await asyncio.wait([divs1, divs2, divs3])
    return divs1, divs2, divs3
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    d1, d2, d3 = loop.run_until_complete(main())
    print(d1.result(), d2.result(), d3.result())
    loop.close()
''' 
