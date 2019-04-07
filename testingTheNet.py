import sys
import scipy.misc
import numpy as np
import matplotlib.pyplot as plot


# read a PNG picture and turn it's color to greys.
img_array = scipy.misc.imread("mnist_dataset/my_handwriting.png", flatten = True)
img_data = 255.0 - img_array.reshape(784)
img_data = (img_data/255.0 * 0.99) + 0.01

