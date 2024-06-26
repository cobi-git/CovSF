import sys
sys.path.append('/home/qotmd01/CovSF_2/')
sys.path.append('/home/qotmd01/CovSF_2/Data/')
sys.path.append('/home/qotmd01/CovSF_2/train/')

from pathlib import Path
import copy
import concurrent.futures

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.preprocessing import StandardScaler

from model import Seq2Seq
from loss import AsymmetricLoss

from earlystopping import Earlystopping
from logger import Logger
from utils import load_pickle,save_pickle

if torch.cuda.is_available:
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

MODEL = {
    'Seq2Seq' : Seq2Seq
}

MAX_WORKER = 5

class Trainer():
    def __init__(self,dataset_dir,save_dir,save_name,n_folds=10,model='Seq2Seq'):
        self.dataset_dir = Path(dataset_dir)
        self.save_name = save_name
        self.save_dir = Path(save_dir)/save_name
        self.save_dir.mkdir(exist_ok=True,parents=True)
        
        self.model = model

        self.n_folds = 10

    def train(self,model_args,train_args):
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKER) as executor:
            futures = []
            for k in range(1,self.n_folds+1):
                args = {
                    'model_args' : model_args,
                    'train_args' :  train_args,
                    'k' : k
                }
                futures.append(executor.submit(self._train,**args))
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f'{self.save_name} - Fold generated an exception : {exc}')
                
    def _train(self,model_args,train_args,k):
        # Load dataset, scaler fit
        train_set,valid_set = load_pickle(self.dataset_dir/f'{k}/train_set.pkl'),load_pickle(self.dataset_dir/f'{k}/valid_set.pkl')
        train_loader = DataLoader(train_set,batch_size=train_args['batch_size'],shuffle=True)
        valid_loader = DataLoader(valid_set,batch_size=1)
        scaler = self.scaler_init_fit(train_loader)

        # Prepare for train
        model = MODEL[self.model](**model_args).to(DEVICE)
        optimizer = optim.RAdam(model.parameters(), **train_args['optimizer'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,**train_args['scheduler'])

        train_loss_func = AsymmetricLoss().to(DEVICE)
        valid_loss_func = AsymmetricLoss().to(DEVICE)

        early_stopping = Earlystopping(**train_args['early_stopping'])
        logger = Logger()

        best_valid_loss = None
        best_model_state = None

        for epoch in range(1,train_args['epochs']+1):
            #Train
            model.train()
            model._train = True
            logger.train()

            train_loss = 0.

            for (x,y) in train_loader:
                optimizer.zero_grad()

                mini_batch,seq_len,feature_dim = x.shape
                x = scaler.transform(x.reshape(mini_batch * seq_len,feature_dim))
                x = torch.Tensor(x.reshape(mini_batch,seq_len,feature_dim)).float().to(DEVICE)

                y = y.to(DEVICE)

                outputs = model(x,y.to(torch.int64),train_args['tpro'])[:,:,1]

                loss = train_loss_func(outputs,y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            logger(train_loss)

            model.eval()
            model._train = False
            logger.valid()

            valid_loss = 0.

            with torch.no_grad():
                for (x,y) in valid_loader:
                    mini_batch,seq_len,feature_dim = x.shape
                    x = scaler.transform(x.reshape(mini_batch * seq_len,feature_dim))
                    x = torch.Tensor(x.reshape(mini_batch,seq_len,feature_dim)).float().to(DEVICE)

                    y = y.to(DEVICE)

                    outputs = model(x,y.to(torch.int64),0.0)[:,:,1]

                    loss = valid_loss_func(outputs,y)
                    valid_loss += loss.item()

            scheduler.step(valid_loss)

            early_stopping(valid_loss)
            logger(valid_loss)
            
            train_loss /= len(train_loader)
            valid_loss /= len(valid_loader)

            print(f'{self.save_name} | {k} Folds, epoch {epoch} : train_loss - {train_loss}, valid_loss - {valid_loss}')

            if best_valid_loss == None or valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_model_state = copy.deepcopy(model.state_dict())
            
            if early_stopping.early_stop:
                print(f'{self.save_name} | {k} Folds, epoch {epoch} : Early stopping triggered')
                break
        
        save_dir = self.save_dir/f'{k}/'
        save_dir.mkdir(parents=True,exist_ok=True)
        torch.save(best_model_state,save_dir/'model.pt')
        save_pickle(save_dir/'scaler.pkl',scaler)
        save_pickle(save_dir/'logger.pkl',logger)
        save_pickle(save_dir/'model_args.pkl',model_args)
        save_pickle(save_dir/'train_args.pkl',train_args)   

        return True

    def scaler_init_fit(self,data_loader):
        X = torch.cat([x for x,y in data_loader],dim=0)
        N,T,feature_dim = X.shape
        X = X.reshape(N * T, feature_dim)

        scaler = StandardScaler()
        scaler.fit(X)

        return scaler


        