"""
File for general usefull functions which are not specific to a certain module
"""

import numpy as np


class Config(dict):
    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)
        self.__dict__ = self


def np_int_to_one_hot(y, num_classes):
    num_points = len(y)
    y_one_hot = np.zeros((num_points, num_classes))
    y_one_hot[np.arange(num_points), y] = 1
    return y_one_hot
