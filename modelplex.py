from data_generation import time_to_cross
from networks import Ensemble


class Pedestrian:
    def __init__(self, x=10, y=10):
        pass

    def update(self, d_t):
        pass

class Car:
    def __init__(self, model=None, x=5, y=5, initial_vel=5, max_acc=5, max_brake=5):
        self.x, self.y = x, y
        self.vel = initial_vel
        self.model = Ensemble()
    
    def make_decision(self, pedestrian):
        est_time = self.model.forward(pedestrian)

    def control(self):
        self.a = 5

    def step(self, dt):
        self.update_pos(dt)
        return self.get_pos()

    def update_pos(self, dt):
        self.y += dt * self.vel - 0.5 * self.a * dt**2

    def get_pos(self):
        return self.x, self.y


class Scene:
    def __init__(self, hz=4):
        self.car = Car()
        self.ped = Pedestrian()
        self.hz = 4

    def run(self):
        dt = 1 / self.hz

        while True:
            pass
