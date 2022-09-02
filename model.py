# module
import torch
import torch.nn as nn
import torch.optim as optim

# definition of your neural network
'''
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        # linear transformations
        self.l1 = nn.Linear(input_size,hidden_size)
        self.l2 = nn.Linear(hidden_size,output_size)

    def forward(self, x):
        # the actual computation performed by the neural network
        x = self.l1(x)
        # activation function. tanh, sigmoid, relu etc.
        x = torch.tanh(x)
        #x = torch.sigmoid(x)
        #x = torch.relu(x)
        x = self.l2(x)
        return torch.softmax(x,dim=1)
'''

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer, output_size):
        super(MLP, self).__init__()
        self.l = nn.ModuleList()
        self.l.append(nn.Linear(input_size,hidden_size))
        for i in range(num_layer-1):
            self.l.append(nn.Linear(hidden_size, hidden_size))
        self.l.append(nn.Linear(hidden_size,output_size))

    def forward(self, x):
        num_layer = len(self.l)-1
        for i in range(num_layer):
            x = self.l[i](x)
            x = torch.tanh(x)
        x = self.l[num_layer](x)
        return x

class Setting():
    num_inputs = 784 # input unit
    num_outputs = 10 # output unit
    num_hidden = 32 # hidden unit
    num_layer = 1 # hidden layer
    num_epochs = 20 # epochs


    def __init__(self) -> None:
        pass

    def setSetting(self, num_hidden, num_layer):
        self.num_hidden = num_hidden
        self.num_layer = num_layer
