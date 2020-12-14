import random

import torch

from data_generation import time_to_cross
from data_generation import gen_data
from networks import Ensemble


class Pedestrian:
    def __init__(self, time_to_cross, x=10, y=-10, size=1):
        self.vel = -y/time_to_cross
        self.x = x
        self.y = y
        self.size = size

    def step(self, dt):
        self.update_pos(dt)
        return self.get_pos()

    def update_pos(self, dt):
        self.y += self.vel * dt

    def get_pos(self):
        return self.x, self.y

class Car:
    def __init__(self, pedestrian, px, py, model=None, initial_vel=5, max_acc=5, max_brake=5, size=1, sigma=4, psize=1):
        self.x, self.y = 0, 0
        self.vel = initial_vel
        self.model = Ensemble(15424)
        self.size = size
        self.psize = psize
        self.sigma = sigma
        self.max_acc = max_acc
        self.max_brake = max_brake
        self.px = px
        self.py = py
        lower_bound, upper_bound = self.make_prediction(pedestrian)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.control(lower_bound, upper_bound)
    
    def make_prediction(self, pedestrian):
        prediction, mean, median, std = self.model.forward(pedestrian)
        estimate = mean
        uncertainty = std * self.sigma
        lower_bound = estimate - uncertainty
        upper_bound = estimate + uncertainty
        return lower_bound, upper_bound

    def control(self, lower_bound, upper_bound):
        braking_boundry = -0.5 * self.vel ** 2 * (1/(self.px - self.size - self.psize))
        acc = None
        if braking_boundry > -self.max_brake:
            acc = random.uniform(-self.max_brake, braking_boundry)
        max_ped_vel = -self.py/lower_bound
        worst_case_pos = self.vel*(self.lower_bound - ((self.psize+self.size)/max_ped_vel))
        if worst_case_pos > self.px + self.size + self.psize:
            test = random.randint(0,1)
            if acc == None:
                test = 1
            if test == 1:
                acc = random.uniform(0, self.max_acc)

        if acc == None:
            acc = -self.max_brake
            
        self.acc = acc

    def crash(self):
        if abs(self.x - self.px) < self.size + self.psize and abs(self.y - self.py) < self.size + self.psize:
            return True
        return False

    def step(self, dt, px, py):
        done = self.update_pos(dt)
        self.px = px
        self.py = py

        if self.x > px + self.size + self.psize:
            done = True
        return  done, self.crash()

    def update_pos(self, dt):
        if self.vel > 0:
            self.x += dt * self.vel + 0.5 * self.acc * dt**2
            self.vel += self.acc * dt
            return False
        else:
            return True

    def get_pos(self):
        return self.x, self.y


class Scene:
    def __init__(self, hz=30):
        pedestrian, time_to_cross = gen_data(1)
        pedestrian = torch.from_numpy(pedestrian)
        time_to_cross = torch.from_numpy(time_to_cross)
        self.ped = Pedestrian(time_to_cross=time_to_cross)
        self.car = Car(pedestrian=pedestrian, px=10, py=-10)
        self.hz = hz

    def run(self):
        dt = 1 / self.hz
        done = False
        crash = False
        while not done:
            px, py = self.ped.step(dt)
            done, crash = self.car.step(dt, px, py)
            if crash:
                break
        return crash

if __name__ == "__main__":
    runs = []
    for i in range(5000):
        s = Scene()
        runs.append(int(s.run()))

    print(f"num_crashes = {sum(runs)}")
