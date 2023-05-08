import random
import numpy as np


def random_element(array, need_index=False):
    length = len(array)
    assert length > 0, length
    rand_index = random.randint(0, length - 1)
    if need_index:
        return array[rand_index], rand_index
    else:
        return array[rand_index]


def random_elements(array, number):
    return np.random.choice(array, number, replace=False)
