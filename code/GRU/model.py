import torch 
import torch.nn as nn

class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUNet, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        output = self.fc(output[-1])
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

if __name__ == "__main__":
    gru = GRUNet(input_size=7, hidden_size=128, output_size=3)
    print(gru)