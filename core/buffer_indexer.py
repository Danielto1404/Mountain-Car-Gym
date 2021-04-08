import numpy as np


class BufferIndexer:
    def __init__(self, size=1e4, batch_size=128):
        size = int(size)

        if batch_size > size:
            raise AssertionError('random sample size should be <= size')

        self.size = size
        self.batch_size = min(batch_size, self.size)
        self.batch_indices = np.arange(self.batch_size)
        self.elements_cnt = 0
        self.is_batch_ready = False

    def sample_indices(self):
        if not self.is_batch_ready:
            raise AssertionError('Buffer have not enough elements')

        return np.random.choice(min(self.elements_cnt, self.size), self.batch_size, replace=False)

    def increment_index(self):
        self.elements_cnt += 1

        if not self.is_batch_ready:
            self.is_batch_ready = self.elements_cnt == self.batch_size

    def index(self):
        return self.elements_cnt % self.size
