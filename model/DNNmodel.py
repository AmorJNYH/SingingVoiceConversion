import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence,pack_sequence,pad_sequence

class DNNmodel(nn.Module):
    
    def __init__(self, input_size=1, hidden_size=1, output_size = 1, num_layers=1, dropout=0.1):
        """
        input_size: feature size
        hidden_size: 1
        num_layers: 1
        output_size: 1
        """
        
        super(DNNmodel, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, _x):
        x, _ = self.lstm(_x) # _x is input, size(seq_len, batch, input_size)
        s, b, h = x.shape # x is output, size(seq_len, batch, hidden_size)
        x = x.view(s*b, h)
        x = self.linear(x)
        x = x.view(s, b, -1)
        return x

if __name__ == "__main__":
    # device = torch.device('cuda:0')
    device = torch.device('mps')
    a = torch.randn((1,21000))
    #b = torch.randn((22,129))
    #c = torch.randn((33,129))
    train = pack_sequence([a]).to(device)
    net = DNNmodel().to(device)
    x = net(train)