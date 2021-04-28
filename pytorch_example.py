# Thanks to:
# https://ashleyy-czumak.medium.com/mnist-digit-classification-in-pytorch-302476b34e4f
# for the guide.

import torch as t
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt

# The transformation to apply to the dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),])

# Either download the datasets, or load them from
# file
mnist_trainset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = t.utils.data.DataLoader(mnist_trainset, batch_size=32, shuffle=True)
mnist_testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = t.utils.data.DataLoader(mnist_testset, batch_size=32, shuffle=True)

# Here is our network architecture
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.linear1 = nn.Linear(28*28, 128)
        self.linear2 = nn.Linear(128, 64)
        self.final = nn.Linear(64, 10) 
        self.relu = nn.ReLU()

    def forward(self, img):
        x = img.view(-1, 28*28)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.final(x)
        return x

# Create a network
net = Network()

# Define our loss function as well as gradient
# decent algorithm.
cross_el = nn.CrossEntropyLoss()
optimizer = t.optim.Adam(net.parameters(), lr=0.01)
epoch = 10

# Do the learning!
for epoch in range(epoch):
    net.train()

    for data in train_loader:
        x, y = data
        optimizer.zero_grad()
        output = net(x.view(-1, 28*28))
        loss = cross_el(output, y)
        loss.backward()
        optimizer.step()

correct = 0
total = 0

# How well did we do?
with t.no_grad():
    for data in test_loader:
        x, y = data
        output = net(x.view(-1, 28*28))
        for idx, i in enumerate(output):
            if t.argmax(i) == y[idx]:
                correct += 1
            total += 1

print(f'accuracy: {round(correct/total,3)}')