import numpy as np
from numpy.random import default_rng

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data_generation import get_datasets

def lin_relu(weights):
    layers = []
    for i in range(len(weights) - 1):
        layers.append(nn.Linear(weights[i], weights[i+1]))
        layers.append(nn.ReLU())

    return nn.Sequential(*layers)


class PredictorMkIII(nn.Module):
    def __init__(self, weights):
        super(PredictorMkIII, self).__init__()
        self.layer = lin_relu(weights)

    def forward(self, x):
        x = self.layer.forward(x)
        return x

class BasicLayer(nn.Module):
    def __init__(self, n):
        super(BasicLayer, self).__init__()
        self.l = nn.Sequential(
            nn.Linear(7, 7),
            nn.ReLU(),
            nn.Linear(7, 1)
        )

    def forward(self, x):
        return self.l.forward(x)

class Predictor_DropoutMKII(nn.Module):
    def __init__(self, in_features, out, p=0.0):
        super(Predictor_DropoutMKII, self).__init__()
        hidden = [10, 10]
        self.p = p
        self.l1 = nn.Linear(in_features, hidden[0])
        self.r1 = nn.ReLU()
        self.l2 = nn.Linear(hidden[0], hidden[1])
        self.r2 = nn.ReLU()
        self.l3 = nn.Linear(hidden[1], out)
        self.r3 = nn.ReLU()

    def forward(self, x):
        x = self.l1.forward(x)
        x = self.r1.forward(x)
        x = nn.functional.dropout(x, p=self.p, training=True)
        x = self.l2.forward(x)
        x = self.r2.forward(x)
        x = nn.functional.dropout(x, p=self.p, training=True)
        x = self.l3.forward(x)
        x = self.r3.forward(x)

        return x

def train_loader(model, dataloader, dataloader_val, epochs=100, lr=0.0001):
    """
        Training loop for the provided model on the data. Reports training and validation
        loss per epoch.
    """

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    num_data, num_val = len(dataloader), len(dataloader_val)

    for epoch in range(1, epochs + 1):
        epoch_loss = 0

        for x, y in dataloader:
            optimizer.zero_grad()

            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()

        with torch.no_grad():
            val_loss = 0
            for x, y in dataloader_val:
                val = model(x)
                loss = criterion(val, y)
                val_loss += loss.item()

        # print('epoch {}, train_loss {}'.format(epoch, epoch_loss/len(dataloader)))
        print('epoch {}, train_loss {}, val_loss {}'.format(epoch, epoch_loss/num_data, val_loss/num_val))

    if epoch_loss/len(dataloader) < 20:
        torch.save(model.state_dict(), "25k-regression-good.pt")


def train(model, x_train, y_train, x_val, y_val, dataloader, epochs=100, lr=0.0001):
    """
        Training loop for the provided model on the data. Reports training and validation
        loss per epoch.
    """

    criterion = nn.MSELoss()
    # optimizer = optim.Adagrad(model.parameters())

    if torch.cuda.is_available():
        inputs = Variable(torch.from_numpy(x_train).cuda())
        labels = Variable(torch.from_numpy(y_train).cuda())
        inputs_val = Variable(torch.from_numpy(x_val).cuda())
        labels_val = Variable(torch.from_numpy(y_val).cuda())
    else:
        inputs = Variable(torch.from_numpy(x_train))
        labels = Variable(torch.from_numpy(y_train))
        inputs_val = Variable(torch.from_numpy(x_val))
        labels_val = Variable(torch.from_numpy(y_val))

    l = len(inputs)
    batch_size = 4

    for epoch in range(1, epochs + 1):
        epoch_loss = 0

        # for x, y in zip(inputs, labels):
        for x, y in dataloader:
        # for i in range(l):
            # x, y = x_train[i:i+batch_size, :], y_train[i:i+batch_size, :]
            # x, y = x_train[i], y_train[i]

            optimizer.zero_grad()

            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()

        with torch.no_grad():
            val_loss = 0
            for x, y in zip(inputs_val, labels_val):
                val = model(x)
                loss = criterion(val, y)
                val_loss += loss.item()

        # print('epoch {}, train_loss {}'.format(epoch, epoch_loss/x_train.shape[0]))
        print('epoch {}, train_loss {}, val_loss {}'.format(epoch, epoch_loss/x_train.shape[0], val_loss/x_val.shape[0]))

if __name__ == "__main__":
    #credit: https://towardsdatascience.com/linear-regression-with-pytorch-eb6dedead817

    # Total number of samples to generate, and how many of them we use as validation
    num_samples = 25000
    num_val = 5000

    print(f"{num_samples=}, {num_val=}")

    num_features = 7
    hidden = 50
    out = 1

    train_set, val_set = get_datasets(num_samples, num_val)

    dataloader = DataLoader(train_set, batch_size=8, shuffle=True)
    dataloader_val = DataLoader(val_set)

    weights = [num_features, 10, 1]
    model = PredictorMkIII(weights)

    if torch.cuda.is_available():
        model.cuda()

    print(f"training {model}")
    train_loader(model, dataloader, dataloader_val, epochs=200)

    torch.save(model.state_dict(), "static-data.pt")

