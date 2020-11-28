import csv

import numpy as np
from numpy.random import default_rng

import torch

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

# courtesy of https://stackoverflow.com/a/42874726
def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def gen_data_linear(num_samples):
    x_values = [i for i in range(num_samples)]
    x_train = np.array(x_values, dtype=np.float32)
    x_train = x_train.reshape(-1, 1)

    y_values = [2*i + 1 for i in x_values]
    y_train = np.array(y_values, dtype=np.float32)
    y_train = y_train.reshape(-1, 1)

    return x_train, y_train

# def print_some_data(cols, samples, labels, n=5):
#     for i in range(n):
#         print(f"{cols[0]}:{samples[i][0]}  ")

def gen_data(num_samples):
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
    print_data = True

    if print_data:
        for i in range(20):
            print(f"{cols[0]}:{distances[i]}  {cols[1]}:{directions[i]}  {cols[2]}:{speeds[i]} {cols[3]}:{red[i]}  {cols[4]}:{labels[i]}")
    
    return samples, labels


if __name__ == "__main__":
    data = gen_data()
