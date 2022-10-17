from multiprocessing import Process, Pool
from main import train
import multiprocessing
import random


class Tuner:
    def tune(self, config):
        self.result = train(config)

    def create_config(self):
        config = {'d_lr': random.uniform(0.001, 0.00001),
                  'g_lr': random.uniform(0.001, 0.00001),
                  'alpha': random.choice([i for i in range(0, 20, 5)]),
                  'epochs': random.choice(([100, 300, 500, 1000, 2000])),
                  'train_d': random.choice([i for i in range(0, 20, 5)])}
        return config


if __name__ == "__main__":
    random.seed(0)
    tuner = Tuner()
    cpu_count = multiprocessing.cpu_count()
    pool = Pool(1)
    results = pool.map(tuner.tune, [(tuner.create_config()) for _ in range(2)])
    for result in results:
        print(result[''])
