import numpy as np

class Logger:
    def __init__(self,mode='train'):
        self.mode = mode
        
        self.train_losses = []
        self.valid_losses = []
        
    def train(self):
        self.mode = 'train'
        
    def valid(self):
        self.mode = 'valid'
        
    
    def __call__(self,loss):
        if self.mode == 'train':
            self.train_losses.append(loss)
            
        elif self.mode == 'valid':
            self.valid_losses.append(loss)

