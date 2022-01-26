from warnings import warn
from pdb import set_trace
from multiprocessing import Process, Queue
from functools import partial

import gym
import pygame
import os 
import numpy as np
import time 
import datetime as dt 

import torch
import torch.nn.functional as F
import torchvision.transforms as T 
import torch.optim as optim
from torch import nn

class Callback(object):
    play = None
    def __init__(self): pass
    
    def __call__(self, event_name):
        """ Runs a callback method if it exists """
        res = getattr(self, event_name)()
        return res
    
    def __getattr__(self, name):
        if hasattr(self.play, name):
            return getattr(self.play, name)
     
    def __setattr__(self, name, value):
        if hasattr(self.play, name):
            msg = f"You are shadowing an attribute ({name}) that exists in the GymPlayer. " \
                  f"Use `self.play.{name}` to avoid this."
            warn(msg)
        super().__setattr__(name, value)

    def __repr__(self):
        return type(self).__name__

class PyGymCallback(Callback):
    def __init__(self, env,transpose=True, fps=60, zoom=None, human=False):
        super().__init__()
        self.env = env
        self.transpose = transpose
        self.fps = fps
        self.zoom = zoom
        self.human = human

    def before_play(self):
        rendered = self.env.render(mode='rgb_array')
        self.video_size = [rendered.shape[1], rendered.shape[0]]
        
        if self.zoom is not None:
            self.video_size = int(self.video_size[0] * self.zoom), int(self.video_size[1] * self.zoom)
            
        if not self.human:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
            
        self.screen = pygame.display.set_mode(self.video_size, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        
        self.env.reset()
        
    def reset(self):
        self.play.state = self.env.reset()
        
    def before_step(self):
        self.clock.tick(self.fps)
        
    def set_action(self):
        self.play.action = self.env.action_space.sample()
    
    def step(self):
        self.play.next_state, self.play.reward, self.play.term, self.play.info = self.env.step(self.action)

    def after_step(self):
        # TODO: This might be an issue later on but for now we only pop the events 
        # from the Pygame queue that we care about. Other callbacks that pop specific 
        # events may want to store the events in a Player class variable for other callbacks
        # to have access to.
        # print('here in after_step')
        for event in pygame.event.get([pygame.QUIT, pygame.VIDEORESIZE]):
            if event.type == pygame.QUIT:
                self.play.term = True
                self.play.done = True
            elif event.type == pygame.VIDEORESIZE and self.human:
                self.video_size = event.size
                rendered = self.env.render(mode='rgb_array')
                self._update_screen(self.screen, rendered)
                pygame.display.update()
     
    def _update_screen(self, screen, arr):
        arr_min, arr_max = arr.min(), arr.max()
        arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
        pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1) if self.transpose else arr)
        pyg_img = pygame.transform.scale(pyg_img, self.video_size)
        screen.blit(pyg_img, (0,0))
    
    def render(self):
        if self.next_state is not None and self.human:
            rendered = self.env.render(mode='rgb_array')
            self._update_screen(self.screen, rendered)
            caption = f"{round(self.clock.get_fps())} {self.episode}"
            pygame.display.set_caption(caption)
            pygame.display.update()

    def after_play(self):
        pygame.quit()

class NetworkController(PyGymCallback):
    def __init__(self, encoder, head, queue, img_dims = (3, 160, 160), ts_len = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.head = head
        self.queue = queue 
        self.img_dims = img_dims
        self.ts_len = ts_len
        self.buffer = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def before_set_action(self):
        state = self.env.render(mode='rgb_array').transpose((2,0,1))
        state = np.ascontiguousarray(state, dtype = np.float32)/255
        state = torch.from_numpy(state)
        resize = T.Compose([T.ToPILImage(),
                            T.Resize((self.img_dims[1:])),
                            T.ToTensor()])
        self.state_ts = time.time()
        state = resize(state).to(self.device).unsqueeze(0) 
        return state  
    
    def set_action(self):
        self.play.state = self.before_set_action()
        self.network_output = self.head(self.encoder(self.play.state.to(self.device)))
        self.action = np.argmax(self.network_output.detach().numpy())
        self.play.action = self.action

        fb = self.queue.get()
        if  fb != 0:
            self.buffer.append([self.play.state, fb, np.amax(self.network_output.detach().numpy())])

    def after_set_action(self):
        '''
        should sample over buffer and perform SGD\
            but, how to update the weights of Head_network ?
        '''
        if len(self.buffer) > 10:
            print('working')
            for i in self.buffer:
                kk = f"state_shape : {i[0].shape} feedback : {i[1]} network_output : {i[2]}"
                print(kk)

class PyControllerCallback(PyGymCallback): 
    def __init__(self, keys_to_action=None, **kwargs):
        super().__init__(**kwargs)
        self.keys_to_action = None
        self.pressed_keys = []
        
    def before_play(self):
        super().before_play()
        
        if self.keys_to_action is None:
            if hasattr(self.env, 'get_keys_to_action'):
                self.keys_to_action = self.env.get_keys_to_action()
        elif hasattr(self.env.unwrapped, 'get_keys_to_action'):
            self.keys_to_action = self.env.unwrapped.get_keys_to_action()
        else:
            assert False, self.env.spec.id + " does not have explicit key to action mapping, " + \
                        "please specify one manually"
        
        print("Controls: {self.keys_to_action}")
        self.play.relevant_keys = set(sum(map(list, self.keys_to_action.keys()),[]))
    
    def set_action(self):
        self.play.action = self.keys_to_action.get(tuple(sorted(self.pressed_keys)), 0)

    def after_step(self):                    
        super().after_step()
        # TODO: Mouse events might be needed MOUSEMOTION, MOUSEBUTTONUP, MOUSEBUTTONDOWN
        self.play.key_events = pygame.event.get([pygame.KEYDOWN, pygame.KEYUP])
        for event in self.key_events:
            if event.type == pygame.KEYDOWN:
                if event.key in self.relevant_keys:
                    self.pressed_keys.append(event.key)
            elif event.type == pygame.KEYUP:
                if event.key in self.relevant_keys:
                    self.pressed_keys.remove(event.key)

class Player(Process):
    _callbacks = {
        'before_play': [],
        'before_episode': [],
        'before_reset': [],
        'reset': [],
        'after_reset': [],
        'before_step': [],
        'before_set_action': [],
        'set_action': [],
        'after_set_action': [],
        'step': [],
        'after_step': [],
        'before_render': [],
        'render': [],
        'after_render': [],
        'after_episode': [],
        'after_play': [],
        }
    
    def __init__(self, callbacks=[]):
        self._build_callbacks(callbacks)  

    def play(self, n_episodes=None, n_steps=None):
        self.n_episodes = n_episodes
        self.state = None
        self.reward = None
        self.action = None
        self.next_state = None
        self.term = False
        self.done = False
        self.episode = 0
        self.t = 0

        self._run_event(self._do_play, 'play')
        
    def _do_play(self):
        self._do_n_episodes()
    
    def _do_n_episodes(self):
        check_eps = lambda: self.n_steps is not None and self.episode < self.n_episodes
        while not self.done and check_eps:
            self._do_reset()
            self._run_event(self._do_episode, 'episode')
        
    def _do_reset(self):
        self.episode += 1
        self.state = None
        self.next_state = None
        self.term = False
        self._run_callback('reset')
        
    def _do_episode(self):
        check_steps = lambda: self.n_steps is not None and self.t < self.n_steps
        while not self.term and check_steps:
        
            if self.next_state is not None:
                self.state = self.next_state
            
            self._run_event(self._do_step, 'step')
            self._run_event(self._do_render, 'render')
        
    def _do_step(self):
        # if self.action is None:
        #     err = "Value action is set to None. Either no action callback is set or the " \
        #     "callback does not set an action."
        #     raise TypeError(err)
        self._run_event(self._set_action, 'set_action')
        self._run_callback('step')
        self.t += 1
        
    def _set_action(self):
        self._run_callback('set_action')
    
    def _do_render(self):
        self._run_callback('render')
                    
    def _run_callback(self, cb_name):
        cbs = self._callbacks[cb_name]
        if not isinstance(cbs, list):
            cbs(cb_name)
        else:
            [cb(cb_name) for cb in cbs]
    
    def _run_event(self, f, event_name):
        self._run_callback(f'before_{event_name}')
        f()
        self._run_callback(f'after_{event_name}')
    
    def _build_callbacks(self, callbacks):
        if not isinstance(callbacks, (list, tuple)):
            callbacks = [callbacks]

        for cb in callbacks: #cb is the input class to the player class
            if isinstance(cb, (type, partial)):
                cb = cb()
            cb.play = self
            cb_dir = cb.__dir__() #cb_dir is all the classes in the input cb class
            for cb_name in self._callbacks.keys():
                if cb_name in cb_dir:
                    self._callbacks[cb_name].append(cb)
   
# TODO: Add shared memory for storing feedback
# TODO: Create Training Callbacks  
def main():

    env = gym.make("Bowling-v0").unwrapped 

    player = Player(callbacks=[PyControllerCallback(env=env, zoom=4, fps=60, human=True)]) #pass the queue
    player.play(n_episodes=1)
    

if __name__ == "__main__":
    main() 