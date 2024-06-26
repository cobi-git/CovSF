import numpy as np

class Earlystopping:
    def __init__(self, patience=3, delta=0.0, mode='min', verbose=True):
        
        self.early_stop = False
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        
        self.best_score = np.Inf if mode == 'min' else 0
        self.mode = mode
        self.delta = delta
        
    def __call__(self,score):
        
        if self.best_score is None:
            self.best_score = score
            self.counter = 0
        elif self.mode == 'min':
            if score < (self.best_score - self.delta):
                self.counter = 0
                self.best_score = score
                # if self.verbose:
                    # print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f}')
            else:
                self.counter += 1
                
        elif self.mode == 'max':
            if score > (self.best_score + self.delta):
                self.counter = 0
                self.best_score = score
                # if self.verbose:
                    # print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f}')
                    
            else:
                self.counter += 1
                
        if self.counter >= self.patience:
            # if self.verbose:
                # print(f'[EarlyStop Triggered] Best Score: {self.best_score:.5f}')
            
            self.early_stop = True
        else:
            self.early_stop = False
                