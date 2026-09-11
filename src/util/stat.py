import time
import torch


class time_recorder:
    def __init__(self):
        self.time = 0
        self.snapshot = None

    def start(self):
        assert self.snapshot is None
        torch.cuda.synchronize()
        self.snapshot = time.time()

    def end(self):
        assert self.snapshot is not None
        torch.cuda.synchronize()
        self.time += time.time() - self.snapshot
        self.snapshot = None

    def clear(self):
        self.time = 0

    def item(self):
        return self.time
