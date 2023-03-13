import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence,pack_sequence,pad_sequence

class DNNmodel(nn.Module):
    
    def __init__(self, seq_len, embed_size, hidden_size, num_layers, dropout):
        """
        embeded_size: 128
        hidden_size: 1024
        num_layers: 1
        """
        
        super(DNNmodel, self).__init__()

        self.embed = nn.Embedding(seq_len, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, seq_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, h):
        x = self.embed(x)
        out, (h, c) = self.lstm(x, h)
        out = out.reshape(out.size(0)*out.size(1), out.size(2))
        out = self.linear(out)

        return out, (h, c)


if __name__ == "__main__":
    # device = torch.device('cuda:0')
    device = torch.device('mps')
    a = torch.randn((11,129))
    #b = torch.randn((22,129))
    #c = torch.randn((33,129))
    train = pack_sequence([a]).to(device)
    net = DPCL().to(device)
    x = net(train)