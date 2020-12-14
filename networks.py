import csv

import numpy as np
from numpy.random import default_rng

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data_generation import get_dataset, gen_data

train_seed = 15424
val_seed = 15624
test_seed = 15824

num_ensemble = 8


def lin_relu(weights):
    """
        Helper function to return a neural network composed of alternating Linear
        and ReLU layers.
    """
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
            nn.Linear(7, n),
            nn.ReLU(),
            nn.Linear(n, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.l.forward(x)
        x = nn.functional.dropout(x, p=0.05)
        return x

class Ensemble:
    """
        A class to wrap loading and running inference with our ensemble models.
    """
    def __init__(self, seed):
        self.models = get_ensemble(seed)

    def forward(self, x):
        """
            Returns the predictions of our ensemble, along with their mean, 
            median, and standard deviation. Assumes a single sample is passed in.
        """
        with torch.no_grad():
            preds = [m(x) for m in self.models]
            mean = np.mean(preds)
            med = np.median(preds)
            std = np.std(preds)
            return preds, mean, med, std


def get_ensemble(seed):
    """
        Returns a list of the ensemble models trained on the dataset generated
        with seed.
    """

    models = []
    weights = [7, 50, 1]
    for i in range(num_ensemble):
        model = get_model(f"./models/25k-{i+1}-{train_seed}.pt", PredictorMkIII, [weights])
        models.append(model)

    return models


def get_model(filename, model_class, params):
    """
        A quick helper method to load a model from a saved state dict.
    """

    model = model_class(*params)
    model.load_state_dict(torch.load(filename))

    return model


def train_loader(model, dataloader, dataloader_val, epochs=100):
    """
        Training loop for the provided model on the data. Reports the training 
        and validation loss per epoch. Trains the model using a Mean Squared Error
        loss function and the Adam optimizer.
    """

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    num_data, num_val = len(dataloader), len(dataloader_val)

    for epoch in range(1, epochs + 1):
        train_loss = 0

        for x, y in dataloader:
            optimizer.zero_grad()

            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        with torch.no_grad():
            val_loss = 0
            for x, y in dataloader_val:
                val = model(x)
                loss = criterion(val, y)
                val_loss += loss.item()
        
        train_loss /= num_data
        val_loss /= num_val
        print(f'{epoch=}, {train_loss=:0.3f}, {val_loss=:0.3f}')


def test_model(model, test_data):
    # cols = []
    with torch.no_grad():
        val_loss = 0
        x, y = test_data
        x, y = Variable(torch.from_numpy(x)), Variable(torch.from_numpy(y))
        for i in range(x.shape[0]):
            inputs, label = x[i], y[i]
            val = model(inputs)

            print(inputs, "true label: ", label, "predicted label: ", val)

def convert_sample(x):
    """
        Converts tensor to a list of values for later use, converts binary variables
        to ints.
    """
    x = x.numpy()[0]
    return [x[0]] + [int(x[i]) for i in [1,2,3,4]] + [x[5], + int(x[6])]

def test_uncertainty():
    weights = [7, 50, 1]
    num_models = 5

    num_test_examples = 1500
    # num_test_examples = 7000
    print(f"{num_test_examples=}, {test_seed=}")
    test_dataset = get_dataset(num_test_examples, regen_data=True, seed=test_seed)
    test_loader = DataLoader(test_dataset)

    ensemble = Ensemble(train_seed)

    num_simulations = 40
    dropout_model = get_model(f"./models/dropout-50-{train_seed}.pt", BasicLayer, [50])

    cols = ["stopwalk_dist",    #0
        "down street",          #1 
        "towards street",       #2
        "away street",          #3
        "up street",            #4
        "speed",                #5
        "red",                  #6
        "true label",           #7
        "ens mean",             #8
        "ens std",              #9
        "dropout mean",         #10
        "dropout std"           #11
        ]

    lines = []
    with torch.no_grad():
        for x, y in test_loader:
            preds, ens_mean, ens_med, ens_std = ensemble.forward(x)

            
            dropout_preds = [dropout_model(x) for _ in range(num_simulations)]
            dropout_mean = np.mean(dropout_preds)
            dropout_med = np.median(dropout_preds)
            dropout_std = np.std(dropout_preds)

            line = convert_sample(x)
            
            line.append(y.item())
            line.append(ens_mean)
            line.append(ens_std)
            line.append(dropout_mean)
            line.append(dropout_std)

            lines.append(line)

    np_lines = np.array(lines)
    print(f"avg ensemble std dev {np.mean(np_lines[:, 9]):.3f}")
    print(f"ensemble MSE {np.mean(np_lines[:, 7] - np_lines[:, 8]):.3f}")
    print(f"avg dropout std dev {np.mean(np_lines[:, 10]):.3f}")
    print(f"dropout MSE {np.mean(np_lines[:, 7] - np_lines[:, 10]):.3f}\n")

    percentages = []
    num_sigmas = 5
    print("percentage of predictions within x standard deviations of true answer")
    for i in range(num_sigmas):
        true_label = np_lines[:, 7]

        ens_pred = np_lines[:, 8]
        ens_std = np_lines[:, 9] * (i+1)

        dropout_pred = np_lines[:, 10]
        dropout_std = np_lines[:, 11] * (i+1)

        in_bounds_ens = (ens_pred - ens_std <= true_label) * (true_label <= ens_pred + ens_std)
        in_bounds_drop = (dropout_pred - dropout_std <= true_label) * (true_label <= dropout_pred + dropout_std)

        percent_ens = np.sum(in_bounds_ens) / num_test_examples
        percent_drop = np.sum(in_bounds_drop) / num_test_examples

        print(f"{i+1} standard deviations")
        print(f"ensemble: {percent_ens*100:2.2f}%\t dropout: {percent_drop*100:2.2f}%\n")

        percentages.append( (percent_ens, percent_drop) )

    write = False
    if write:
        with open("experiment-results.csv", "w") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for line in lines:
                w.writerow(line)


def make_and_train():
    num_samples = 25000
    num_val = 5000

    dropout_epochs = 60
    normal_epochs = 100


    num_features = 7
    num_neurons = 50
    out = 1

    regen = False

    print(f"{num_samples=}, {num_val=}, {regen=}")
    print(f"{train_seed=}, {val_seed=}")

    train_set = get_dataset(num_samples, regen_data=regen, seed=train_seed)
    val_set = get_dataset(num_val, regen_data=regen, seed=val_seed)

    dataloader = DataLoader(train_set, batch_size=8, shuffle=True)
    dataloader_val = DataLoader(val_set)

    weights = [num_features, num_neurons, out]

    train_dropout = True
    train_ensemble = False

    if train_dropout:
        basic = BasicLayer(50)
        # model = PredictorMkIII(weights)
        print("training dropout")
        train_loader(basic, dataloader, dataloader_val, epochs=dropout_epochs)
        torch.save(basic.state_dict(), f"./models/dropout-{num_neurons}-{train_seed}.pt")

    if train_ensemble:
        for i in range(num_ensemble):
            model = PredictorMkIII(weights)

            # cuda is broken on my machine at the moment
            # if torch.cuda.is_available():
            #     model.cuda()

            print(f"training ensemble model {i+1}")
            train_loader(model, dataloader, dataloader_val, epochs=normal_epochs)

            torch.save(model.state_dict(), f"./models/25k-{i+1}-{train_seed}.pt")


if __name__ == "__main__":
    make_and_train()
    # test_uncertainty()

