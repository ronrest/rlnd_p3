from collections import deque
import random

class ReplayBuffer:
    def __init__(self,size):
        self.size = size
        self.deque = deque(maxlen=self.size)

    def push(self,experience):
        """push into the buffer"""
        self.deque.append(experience)

    def sample(self, batchsize):
        """sample from the buffer"""
        samples = random.sample(self.deque, batchsize)
        return samples

    def __len__(self):
        return len(self.deque)
