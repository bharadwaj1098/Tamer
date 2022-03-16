from warnings import warn
from pdb import set_trace
from multiprocessing import Process, Queue
from functools import partial

from torch._C import device
from play import PyGymCallback, Player

import gym
import pygame
import os 
import numpy as np
import time 
import datetime as dt
from itertools import count 
from typing import Tuple
from collections import deque 

import scipy
import matplotlib.pyplot as plt
from scipy.stats import uniform, gamma, norm, exponnorm
import matplotlib.pyplot as plt 

import torch
import torch.nn.functional as F
import torchvision.transforms as T 
import torch.optim as optim
from torch import nn

torch.manual_seed(10)

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

class Head(nn.Module):
    def __init__(self):
        super(Head,self).__init__()

        self.linear_1 = nn.Linear(100,16)
        self.linear_2 = nn.Linear(16,4)
    
    def forward(self, x):
        x = x 
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        return x
        
class CreditAssignment():
    def __init__(self, dist: scipy.stats.rv_continuous):
        self.dist = dist
        
    def __call__(self, s_start: float, s_end: float, h_start: float) -> float:
        s_norm_start, s_norm_end = self._normalize(s_start, s_end, h_start)
        start_cdf = self.dist.cdf(s_norm_start)
        end_cdf = self.dist.cdf(s_norm_end)
        return start_cdf - end_cdf
        
    def _normalize(self, s_start: float, s_end: float, h_start: float) -> Tuple[float, float]: 
        s_norm_start =  h_start - s_start
        s_norm_end = h_start - s_end
        return s_norm_start, s_norm_end
    
    def show_dist(self, s_start: float, s_end: float, h_start: float):
        s_norm_start, s_norm_end = self._normalize(s_start, s_end, h_start)
        x = np.linspace(self.dist.ppf(.01), self.dist.ppf(.99))
        plt.plot(x, self.dist.pdf(x), 'r-')
        plt.vlines(s_norm_start,ymin=0, ymax=self.dist.pdf(s_norm_start), color='green')
        plt.vlines(s_norm_end, ymin=0, ymax=self.dist.pdf(s_norm_end), color='green')

class BufferDeque():
    def __init__(self, size):
        self.memory = deque(maxlen=size)

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, index):
        if isinstance(index, (tuple, list)):
          pointer = list(self.memory)
          return [pointer[i] for i in index]
        return list(self.memory[index])

    def push(self, tensor):
        self.memory.append(tensor) 

    def random_sample(self, batch_size):
        rand_idx = np.random.randint(len(self.memory),  size=batch_size)
        rand_batch = [self.memory[i] for i in rand_idx]
        state, action, feedback, credit =  [], [], [], []
        for s, a, f, c in rand_batch:
            state.append(s)
            action.append(a)
            feedback.append(f)
            credit.append(c)

        return torch.cat(state), torch.tensor(action), torch.tensor(feedback), torch.tensor(credit)

class NetworkController(PyGymCallback):
    '''
    state_start_time :- state start time is collected before step in before_step()
    state_end_time :- state end time is collected after the step in after_step()
    h_time :- time at which the feedback is recorded, which comes along with the feedback in a list in after_step function
    
    questions:-
        1. why random sample the credit?
        2. Are my assumptons about start and end of state time correct?
        3. Why perform the backward function in both after_set_action() and after_set() [as per psuedo code] isn't
            performing it in after_set_action() enough ?
        4. Not able to get the torch.stack(credit) to work in buffer.random_sample() ? 

    Yet-To-Do:-

    adding backward function
    '''
    def __init__(self, encoder, head, queue, img_dims = (3, 160, 160), ts_len = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.head = head
        self.queue = queue 
        self.img_dims = img_dims
        self.ts_len = ts_len
        self.dims = 10000
        self.buffer = BufferDeque(self.dims)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sliding_window = deque()
        self.opt = optim.Adam(list(self.head.parameters()), lr=1e-1, weight_decay = 1e-1 )

    def backward(self, state, action, feedback, credit):
        state = state.to(self.device)
        credit = credit.to(self.device)
        feedback = feedback.to(self.device)
        action = action.to(self.device)

        self.loss_list = []
        h_hat = self.head(self.encoder(state))
        h_hat_s_a = h_hat[:, action]
        # h_hat_s_a.requires_grad = True
        L = torch.mean(credit*(h_hat_s_a - feedback)**2)
        self.opt.zero_grad() 
        L.backward()
        self.opt.step()
        self.loss_list.append(L)
        # print(f"feedback : {feedback}")
        # print(f"loss: {L}, q_values: {h_hat}")

    def before_step(self):
        self.state_start_time = time.time()

    def before_set_action(self):
        state = self.env.render(mode='rgb_array').transpose((2,0,1))
        state = np.ascontiguousarray(state, dtype = np.float32)/255
        state = torch.from_numpy(state)
        resize = T.Compose([T.ToPILImage(),
                            T.Resize((self.img_dims[1:])),
                            T.ToTensor()])
        state = resize(state).to(self.device).unsqueeze(0) 
        return state  
    
    def set_action(self):
        self.play.state = self.before_set_action()
        self.network_output = self.head(self.encoder(self.play.state.to(self.device)))
        # self.play.action = np.argmax(self.network_output.cpu().detach().numpy())
        self.play.action = torch.argmax(self.network_output)#.cpu().detach().numpy())

    def after_set_action(self):
        batch=64
        loss_fn = nn.MSELoss(reduction = 'mean') 
        #only when buffer has 50 feedbacks
       
        if len(self.buffer) > 50:      
            # Only train every certain number of steps
            if self.t % 16 == 0: 
                #rand_batch = np.random.randint(len(self.buffer), size=batch_size) 
                state, action, feedback, credit = self.buffer.random_sample(batch)
                self.backward( state, action, feedback, credit) 
    
    def after_step(self):
        self.state_end_time = time.time()
        fb_dict = self.queue.get()
        # fb = torch.tensor(fb_dict["feedback"]).to(self.device)
        fb = fb_dict['feedback']
        h_time = fb_dict["h_time"]
        self.sliding_window.append(
            dict(
                state = self.state, 
                action = self.action, 
                feedback = fb, 
                s_start = self.state_start_time, 
                s_end = self.state_end_time
            )
        )

        if fb != 0:
            ca = CreditAssignment(uniform(0.2, 0.8))
            state, action, credit = [], [], []
            for win in self.sliding_window:
                credit_for_state = ca(s_start=win["s_start"], s_end=win["s_end"], h_start=h_time)
                if credit_for_state !=0:
                    state.append(win['state'])
                    action.append(win['action'])
                    # credit.append(torch.tensor(credit_for_state, dtype=torch.float32).to(self.device))
                    credit.append(credit_for_state) 
                    self.buffer.push([
                        win['state'], 
                        win['action'],
                        fb,
                        credit_for_state
                        # torch.Tensor(credit) 
                    ])

            state, action, credit = torch.cat(state), torch.tensor(action), torch.tensor(credit)
            feedback = torch.full(credit.size(), fb)
            self.backward( state, action, feedback, credit)

    def after_play(self):
        plt.title('Head_Network_Error')
        plt.plot(self.loss_list)
        plt.savefig('Test_Error')
        
class FeedbackListener(Process):
    def __init__(self,fb_queue,video_size=(200, 100)):
        super().__init__()
        self.video_size = video_size
        self.fb_queue = fb_queue
        
    def run(self, fps=60):
        self._init_pygames()
        self.listening = True
        while self.listening:
            fb, fill = self._do_pygame_events()
            self._update_screen(fill)
            #add feedback to queue is feeback =! 0
            self.clock.tick(fps)
            self.fb_queue.put(
                dict(
                    feedback = fb,
                    h_time = time.time()
                ))

    def _init_pygames(self):
        pygame.init()
        self.screen = pygame.display.set_mode(self.video_size, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self._update_screen()
    
    def _do_pygame_events(self):
        fb, fill = 0, None
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    fill = self.screen.fill((0, 255, 0))
                    fb = 1
                elif event.key == pygame.K_2:
                    fill = self.screen.fill((255, 0, 0))
                    fb = -1
            elif event.type == pygame.VIDEORESIZE:
                self.video_size = event.size
                self._update_screen(fill)
            elif event.type == pygame.QUIT:
                self.listening = False
        return fb, fill 
    
    def _update_screen(self, fill=None):
        if fill is None:
            fill = self.screen.fill((0, 0, 0))
            
        pygame.display.update(fill) 

def main():
    env = gym.make("Bowling-v0").unwrapped 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder().to(device)
    head_net = Head().to(device) 
    encoder.load_state_dict(torch.load("auto_encoder/Type_1/encoder.pt", map_location=device))
    
    # Freeze encoder weights
    
    for name, params in encoder.named_parameters():
        params.requires_grad = False
    
    # opt = torch.optim.Adam(head_net.parameters(), lr=1e-4, weight_decay=1e-1) 
    
    Feedback_queue = Queue()

    listener = FeedbackListener(Feedback_queue) #pass it to listener ()
    listener.start()
    player = Player(callbacks=[NetworkController(encoder= encoder, head=head_net, queue=Feedback_queue,
                                                    env=env, zoom=4, fps=60, human=True)]) #pass the queue
    player.play(n_episodes = 1)
    listener.join()
    

if __name__ == "__main__":
    main() 