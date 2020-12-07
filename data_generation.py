import os
import csv

import numpy as np
from numpy.random import default_rng

import torch
from torch.utils.data import Dataset

data_dir = os.path.join(os.curdir, "data")

def time_to_cross(stopwalk_dist, direction, speed, red_clothes):
    """
        Gives time until a pedestrian with these attributes will cross the street
    """
    scale = 2
    epsilon = 0.00001

    #doesn't really need to make sense
    time_to_cross = scale * stopwalk_dist / (speed + epsilon)

    if direction == 1:
        time_to_cross /= 2
    elif direction == 2:
        time_to_cross *= 3

    if red_clothes:
        time_to_cross /= 1.5

    return time_to_cross

def get_one_hot(targets, nb_classes):
    """
        Converts index or array of indices to one-hot vectors

        courtesy of https://stackoverflow.com/a/42874726
    """
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def gen_data_linear(num_samples):
    """
        Used for verifying that training loop worked

        cite: https://towardsdatascience.com/linear-regression-with-pytorch-eb6dedead817
    """
    x_values = [i for i in range(num_samples)]
    x_train = np.array(x_values, dtype=np.float32)
    x_train = x_train.reshape(-1, 1)

    y_values = [2*i + 1 for i in x_values]
    y_train = np.array(y_values, dtype=np.float32)
    y_train = y_train.reshape(-1, 1)

    return x_train, y_train


def gen_data(num_samples):
    """
        Returns a tuple of (data, labels) of our synthetically generated data
    """
    features = ["dist_to_curb", "direction", "speed", "red"]
    label = ["time_to_cross"]

    cols = features + label
    gen = default_rng()

    # in meters, m/s
    max_dist = 4
    max_speed = 1

    # one hot encoding of cardinal direction of pedestrian
    # 0 = facing down street, away from us
    # 1 = facing towards street
    # 2 = facing away from street
    # 3 = facing towards us
    num_dirs = 4
    directions = gen.integers(0, num_dirs, num_samples)
    directions_onehot = get_one_hot(directions, num_dirs)

    distances = gen.beta(2, 2, (num_samples, 1)) * max_dist
    speeds = gen.beta(2, 2, (num_samples, 1)) * max_speed

    prob_red = .2
    red = gen.random((num_samples, 1)) >= (1 - prob_red)

    labels = np.empty((num_samples, 1))
    for i in range(num_samples):
        labels[i] = (time_to_cross(distances[i], directions[i], speeds[i], red[i]))

    labels = labels.astype(np.single)
    samples = np.concatenate((distances, directions_onehot, speeds, red), axis=1).astype(np.single)
    print_data = False

    if print_data:
        for i in range(20):
            print(f"{cols[0]}:{distances[i]}  {cols[1]}:{directions[i]}  {cols[2]}:{speeds[i]} {cols[3]}:{red[i]}  {cols[4]}:{labels[i]}")
    
    return samples, labels


def get_datasets(num_samples, num_val, regen_data=False):
    if regen_data:
        x_train, y_train = gen_data(num_samples)
        x_val, y_val = gen_data(num_val)

        np.save(os.path.join(data_dir, "x-"+str(num_samples)+".npy"), x_train)
        np.load(os.path.join(data_dir, "y-"+str(num_samples)+".npy"), y_train)
        np.load(os.path.join(data_dir, "x-"+str(num_samples)+".npy"), x_val)
        np.load(os.path.join(data_dir, "y-"+str(num_samples)+".npy"), y_val)
    else:
        x_train = np.load(os.path.join(data_dir, "x-"+str(num_samples)+".npy"))
        y_train = np.load(os.path.join(data_dir, "y-"+str(num_samples)+".npy"))
        x_val = np.load(os.path.join(data_dir, "x-"+str(num_samples)+".npy"))
        y_val = np.load(os.path.join(data_dir, "y-"+str(num_samples)+".npy"))

    train_set = RegData(num_samples, x_train, y_train)
    val_set = RegData(num_val, x_val, y_val)

    return train_set, val_set


class RegData(Dataset):
    """
        Pytorch dataset wrapper around our data generator
    """
    def __init__(self, num_samples, x=None, y=None):
        self.n = num_samples
        if x is None or y is None:
            self.samples, self.labels = gen_data(num_samples)
        else:
            self.samples, self.labels = x, y

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]
