import torch 
import torch.nn as nn

class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers = 1, dropout = 0):
        super(GRUNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size=input_size, \
                          hidden_size=hidden_size,\
                          num_layers=num_layers,\
                          dropout=dropout,\
                          batch_first = True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        output = self.linear(output[:,-1,:])
        return output

    # def init_hidden(self):
    #     return torch.zeros(1, 1, self.hidden_size)

if __name__ == "__main__":
    gru = GRUNet(input_size=7, hidden_size=128, output_size=3)
    print(gru)