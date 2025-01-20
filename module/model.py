import torch
from torch import nn
import torch.nn.functional as F

RECURRENT = {
    'rnn' : nn.RNN,
    'lstm' : nn.LSTM,
    'gru' : nn.GRU
}
NORM = {
    'batch' : nn.BatchNorm1d,
    'layer' : nn.LayerNorm
}

class Encoder(nn.Module):
    def __init__(
            self,
            recurrent_type,
            input_dim,
            hidden_dim,
            hidden_layers,
            recurrent_dropout
    ):
        super(Encoder,self).__init__()
        self.recurrent_type = recurrent_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.recurrent_dropout = recurrent_dropout
        self.recurrent = RECURRENT[recurrent_type](
            input_dim,
            hidden_dim,
            hidden_layers,
            dropout=recurrent_dropout,
            batch_first=True
        )
    
    def forward(self,x,DEVICE):
        batch_size,seq_len,input_dim = x.shape

        hidden = self.init_hidden(batch_size,DEVICE)
        
        output,hidden = self.recurrent(x,hidden)

        return output,hidden
    
    def init_hidden(self, batch_size,DEVICE):
        if self.recurrent_type == 'lstm':
            return (torch.zeros((self.hidden_layers, batch_size, self.hidden_dim), device=DEVICE),
                    torch.zeros((self.hidden_layers, batch_size, self.hidden_dim), device=DEVICE))
        else:
            return torch.zeros((self.hidden_layers, batch_size, self.hidden_dim), device=DEVICE)
        
class Decoder(nn.Module):
    def __init__(
            self,
            recurrent_type,
            output_dim,
            output_len,
            hidden_dim,
            hidden_layers,
            recurrent_dropout,
            fc_layers,
            fc_dropout,
            norm_method
    ):
        super(Decoder,self).__init__()
        self.recurrent_type = recurrent_type
        self.output_dim = output_dim
        self.output_len = output_len
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.recurrent_dropout = recurrent_dropout
        self.fc_layers = fc_layers
        self.fc_dropout = fc_dropout
        self.norm_method = norm_method
        
        self.recurrent = RECURRENT[recurrent_type](
            output_dim,
            hidden_dim,
            hidden_layers,
            dropout=recurrent_dropout,
            batch_first=True
        )

        prev_dim = self.hidden_dim
        fc = []
        for dim in self.fc_layers:
            fc.append(nn.Linear(prev_dim,dim))
            fc.append(NORM[norm_method](dim))
            fc.append(nn.GELU())    
            fc.append(nn.Dropout(self.fc_dropout))
            prev_dim = dim
        
        fc.append(nn.Linear(prev_dim,self.output_dim))
        self.fc = nn.Sequential(*fc)

    def forward(self,hidden,DEVICE):
        if self.recurrent_type == 'lstm' : layers,batch_size,hidden_dim = hidden[0].shape
        else : layers,batch_size,hidden_dim = hidden.shape
        
        output = None
        outputs = []
        for t in range(self.output_len):
            if t == 0: input = torch.zeros((batch_size,1,self.output_dim)).float().to(DEVICE)
            else: input = output.unsqueeze(1) # Auto regressive

            top_hidden,hidden = self.recurrent(input,hidden)
            top_hidden = top_hidden.squeeze(1)
            output = F.softmax(self.fc(top_hidden),dim=-1)
            outputs.append(output)
        
        outputs = torch.stack(outputs,dim=1)
        return outputs


class CovSF(nn.Module): #Seq2Seq CovSF
    def __init__(
            self,
            encoder_args,
            decoder_args):
        super(CovSF,self).__init__()

        self.encoder = Encoder(**encoder_args)
        self.decoder = Decoder(**decoder_args)

    def forward(self,x,DEVICE):
        hidden_last_layers,hidden_last_steps = self.encoder(x,DEVICE)
        outputs = self.decoder(hidden_last_steps,DEVICE)

        return outputs