import torch
import torch.nn as nn


class DummyNet(torch.nn.Module):
    def __init__(self, input_size, hidden_state, output_size, nonlin="relu"):
        super(DummyNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_state)
        self.layer2 = nn.Linear(hidden_state, output_size)

        if nonlin == "relu":
            self.nonlin = nn.ReLU(inplace=True)
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.layer2(self.nonlin(self.layer1(x)))
