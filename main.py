import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt

import model
import data

def createGIF(path, name):
    files = sorted(glob.glob(str(path) + "*.png"))
    images = list(map(lambda file:Image.open(file), files))
    images[0].save(
        name, save_all=True, append_images=images[1:], duration=100, loop=0
    )

# Computing using the GPU if available. If something goes wrong, set device = 'cpu'.
device = torch.device('cuda')
#device = 'cpu'

# Creating a new MLP.
# The size of the neural network. hidden part can be modified.
num_inputs = 2 # the number of the variables
num_outputs = 1 # output is just a real value
num_layer = 3
num_hidden = 32
mynet = model.MLP(num_inputs,num_hidden,num_layer,num_outputs).to(device)

# How long to train. Please start with a small value and see how it goes.
num_epochs = 100 # perhaps, the number of epochs should be increased.

# The loss function.
criterion = nn.MSELoss()

# The learning algorithm. Please change the learning rate "lr"
optimizer = optim.SGD(params=mynet.parameters(), lr=0.1)

history_loss = []
history_eval = []

dataset = data.MultiFunction()

# Start Learning
for epoch in range(num_epochs):
    # First, switch the network to the learning mode.
    mynet.train()

    total_loss = 0.0
    for i, (data, target) in enumerate(dataset.train_loader):
        # data = data.view(-1, 28*28) this is only for images.

        # Initialize the derivative to zero.
        # Compute the output of the network, the loss function, and its derivative,
        # and run the learning algorithm using the computed derivative.
        optimizer.zero_grad()
        output = mynet(data.to(device))
        loss = criterion(output, target.to(device))
        loss.backward()
        optimizer.step()

        # Sum of the loss function for each mini batch.
        total_loss = total_loss + loss.cpu().item()
    total_loss = total_loss/(i+1)

    # Validation using the test data (different from the training data).
    # Because we do not need to run the training algorithm,
    # we switch the network to the evaluation mode (so that the derivative is not computed).

    mynet.eval()
    with torch.no_grad():
        eval_loss = 0.0
        for i, (data, target) in enumerate(dataset.test_loader):
            output = mynet(data.to(device))
            loss = criterion(output, target.to(device))
            eval_loss = eval_loss + loss.cpu().item()
        eval_loss = eval_loss/(i+1)

    history_loss.append(total_loss)
    history_eval.append(eval_loss)
    print("{}/{} training loss: {}, evaluation loss: {}".format(epoch,num_epochs,total_loss,eval_loss))