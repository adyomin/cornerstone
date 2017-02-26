import problem_unittests as tests
from os.path import isfile, isdir
import helper
import numpy as np


def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """

    # get x min (darkest pixel in the image) per each color channel
    x_min = np.min(x, axis=(0, 1))
    # get x max (brightest pixel in the image) per each color channel
    x_max = np.max(x, axis=(0, 1))
    delta = x_max - x_min
    return (x - x_min) / delta


tests.test_normalize(normalize)

def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    batch_size = len(x)
    num_targets = 10
    output = np.zeros((batch_size, num_targets))
    output.flat[np.arange(batch_size)*num_targets + targets] = 1
    return output

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_one_hot_encode(one_hot_encode)

