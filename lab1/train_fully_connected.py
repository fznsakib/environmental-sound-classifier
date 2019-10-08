# -*- coding: utf-8 -*-

import torch as torch
from torch import nn
from torch.nn import functional as F
from typing import Callable
from torch import optim
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

class MLP(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_layer_size: int,
                 output_size: int,
                 activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.relu):
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_layer_size)
        self.l2 = nn.Linear(hidden_layer_size, output_size)
        self.activation_fn = activation_fn
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.l1(inputs)
        x = self.activation_fn(x)
        x = self.l2(x)
        return x

def accuracy(probs: torch.FloatTensor, targets: torch.LongTensor) -> float:
    """
    Args:
        probs: A float32 tensor of shape ``(batch_size, class_count)`` where each value 
            at index ``i`` in a row represents the score of class ``i``.
        targets: A long tensor of shape ``(batch_size,)`` containing the batch examples'
            labels.
    """
    decisions = probs.argmax(1)
    targetsEqual = torch.eq(decisions, targets).sum().float()
    check = targetsEqual/targets.shape[0]
    return check


def check_accuracy(probs: torch.FloatTensor,
                   labels: torch.LongTensor,
                   expected_accuracy: float):
    actual_accuracy = accuracy(probs, labels)
    assert actual_accuracy == expected_accuracy, f"Expected accuracy to be {expected_accuracy} but was {actual_accuracy}"

# define GPU device for the computation
device = torch.device('cuda')

# initialise logger
summary_writer = SummaryWriter('logs', flush_secs=5)

# datasets are stored in a dictionary containing an array of features and targets
iris = datasets.load_iris()  

preprocessed_features = (iris['data'] - iris['data'].mean(axis=0)) / iris['data'].std(axis=0)

labels = iris['target']

# train_test_split takes care of the shuffling and splitting process
train_features, test_features, train_labels, test_labels = train_test_split(preprocessed_features, labels, test_size=1/3)

features = {
    'train': torch.tensor(train_features, dtype=torch.float32),
    'test': torch.tensor(test_features, dtype=torch.float32),
}
labels = {
    'train': torch.tensor(train_labels, dtype=torch.long),
    'test': torch.tensor(test_labels, dtype=torch.long),
}

# move features and labels to GPU
train_features = features["train"].to(device)
test_features = features["test"].to(device)
train_labels = labels["train"].to(device)
test_labels = labels["test"].to(device)

feature_count = 4
hidden_layer_size = 100
class_count = 3 

# Define the model to optimze
model = MLP(feature_count, hidden_layer_size, class_count)

# Move model to GPU
model = model.to(device)

# The optimizer we'll use to update the model parameters
optimizer = optim.SGD(model.parameters(), lr=0.05)

# Now we define the loss function.
criterion = nn.CrossEntropyLoss() 

# Now we iterate over the dataset a number of times. Each iteration of the entire dataset 
# is called an epoch.
for epoch in range(0, 100):
    # We compute the forward pass of the network
    logits = model.forward(train_features)
    # Then the value of loss function 
    loss = criterion(logits, train_labels)
    
    # How well the network does on the batch is an indication of how well training is 
    # progressing
    train_accuracy = accuracy(logits, train_labels) * 100

    summary_writer.add_scalar('accuracy/train', train_accuracy, epoch)
    summary_writer.add_scalar('loss/train', loss.item(), epoch)    

    print("epoch: {} train accuracy: {:2.2f}, loss: {:5.5f}".format(
        epoch,
        train_accuracy,
        loss.item()
    ))
    
    # Now we compute the backward pass, which populates the `.grad` attributes of the parameters
    loss.backward()
    # Now we update the model parameters using those gradients
    optimizer.step()
    # Now we need to zero out the `.grad` buffers as otherwise on the next backward pass we'll add the 
    # new gradients to the old ones.
    optimizer.zero_grad()
    
summary_writer.close()

# Finally we can test our model on the test set and get an unbiased estimate of its performance.    
logits = model.forward(test_features)    
test_accuracy = accuracy(logits, test_labels) * 100
print("test accuracy: {:2.2f}".format(test_accuracy))