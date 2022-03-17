import torch
import numpy as np
import time

sigma = np.ones(1000)

def iso_dd( x, y):
    '''
    Iso+Line
    Ref:
    Vassiliades V, Mouret JB. Discovering the elite hypervolume by leveraging interspecies correlation.
    GECCO 2018
    '''
    a = torch.zeros_like(x).normal_(mean=0, std=1)
    b = np.random.normal(0, sigma)
    z = x+ a + b * (y - x).numpy()


def iso_dd_torch(x, y):
    a = torch.zeros_like(x).normal_(mean=0, std=1)
    b  = torch.normal(0, torch.from_numpy(sigma))
    z = x + a + b * (y - x)


if __name__ == '__main__':
    x, y = torch.ones(1000), torch.ones(1000)
    s1 = time.time()
    iso_dd(x, y)
    print(f'regular iso_dd is {(time.time() - s1) / 1000}')
    s2 = time.time()
    iso_dd_torch(x, y)
    print(f'torch iso_dd is {(time.time() - s2) / 1000}')