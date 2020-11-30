import numpy as np
from numpy.random import default_rng

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data_generation import RegData, gen_data

class LinearOne(nn.Module):
    def __init__(self):
        super(LinearOne, self).__init__()

        self.l1 = nn.Linear(1, 1)

    def forward(self, x):
        x = self.l1.forward(x)

        return x

class Predictor(nn.Module):
    def __init__(self, in_features, hidden, out):
        super(Predictor, self).__init__()
        self.l1 = nn.Linear(in_features, hidden)
        self.r1 = nn.ReLU()
        self.l2 = nn.Linear(hidden, out)

    def forward(self, x):
        x = self.l1.forward(x)
        x = self.r1.forward(x)
        x = self.l2.forward(x)

        return x

class PredictorMkII(nn.Module):
    def __init__(self, in_features, hidden, out):
        super(PredictorMkII, self).__init__()
        hidden = [10, 10]
        self.l1 = nn.Linear(in_features, hidden[0])
        self.r1 = nn.ReLU()
        self.l2 = nn.Linear(hidden[0], hidden[1])
        self.r2 = nn.ReLU()
        self.l3 = nn.Linear(hidden[1], out)
        self.r3 = nn.ReLU()

    def forward(self, x):
        x = self.l1.forward(x)
        x = self.r1.forward(x)
        x = self.l2.forward(x)
        x = self.r2.forward(x)
        x = self.l3.forward(x)
        x = self.r3.forward(x)

        return x

class PredictorMkIII(nn.Module):
    def __init__(self, in_features, out):
        super(PredictorMkIII, self).__init__()
        hidden = [15, 10, 10]
        self.l1 = nn.Linear(in_features, hidden[0])
        self.r1 = nn.ReLU()
        self.l2 = nn.Linear(hidden[0], hidden[1])
        self.r2 = nn.ReLU()
        self.l3 = nn.Linear(hidden[1], hidden[2])
        self.r3 = nn.ReLU()
        self.l4 = nn.Linear(hidden[2], out)
        self.r4 = nn.ReLU()

    def forward(self, x):
        x = self.l1.forward(x)
        x = self.r1.forward(x)
        x = self.l2.forward(x)
        x = self.r2.forward(x)
        x = self.l3.forward(x)
        x = self.r3.forward(x)
        x = self.l4.forward(x)
        x = self.r4.forward(x)

        return x

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


def train(model, x_train, y_train, x_val, y_val, epochs=100, lr=0.0001):
    criterion = nn.MSELoss()
    optimizer = optim.Adagrad(model.parameters())

    # split = 1
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

    for epoch in range(1, epochs + 1):
        epoch_loss = 0

        for x, y in zip(inputs, labels):

            optimizer.zero_grad()

            outputs = model(x)
            loss = criterion(outputs, y)
            # if epoch > 10 and loss >= 10:
            #     print("bad")
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

    # create dummy data for training

    num_samples = 15000
    num_val = 1000
    dataset = RegData(num_samples)
    # dataloader = DataLoader(dataset, num_samples, True)
    # num_val = 5000
    x, y = gen_data(num_samples)
    rng = default_rng()

    xy = np.concatenate((x,y), axis=1)
    rng.shuffle(xy)
    x, y = xy[:, :-1], xy[:, -1:]

    x_train, y_train = x[:-num_val], y[:-num_val]
    x_val, y_val = x[-num_val:], y[-num_val:]


    in_size = x_train.shape[1]
    hidden = 50
    out = 1
    # model = Predictor(in_size, hidden, out)outputs

    # model = PredictorMkII(in_size, 50, 1)
    model = Predictor_DropoutMKII(in_size, 1, p=0.25)

    if torch.cuda.is_available():
        model.cuda()

    train(model, x_train, y_train, x_val, y_val, epochs=100)

    torch.save(model.state_dict(), "25k-regression")

