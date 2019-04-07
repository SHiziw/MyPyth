import sys
import scipy.misc
import numpy as np
import matplotlib.pyplot as plot
# helper to load data from PNG image files
import imageio
# glob helps select multiple files using patterns
import glob
from using_generated_w import after_trained

n = after_trained()
'''
# read a PNG picture and turn it's color to greys.
img_array = scipy.misc.imread("mnist_dataset/my_handwriting.png", flatten = True)
img_data = 255.0 - img_array.reshape(784)
img_data = (img_data/255.0 * 0.99) + 0.01
'''
# test the neural network with our own images
# our own image test data set
our_own_dataset = []

# load the png image data as test data set
for image_file_name in glob.glob('my_own_images/28228_my_own_?.png'):
    
    # use the filename to set the correct label
    label = int(image_file_name[-5:-4])
    
    # load image data from png files into an array
    print ("loading ... ", image_file_name)
    img_array = imageio.imread(image_file_name, as_gray=True)
    
    # reshape from 28x28 to list of 784 values, invert values
    img_data  = 255.0 - img_array.reshape(784)
    
    # then scale data to range from 0.01 to 1.0
    img_data = (img_data / 255.0 * 0.99) + 0.01
    print(np.min(img_data))
    print(np.max(img_data))
    
    # append label and image data  to test data set
    record = np.append(label,img_data)
    our_own_dataset.append(record)
    
    pass
# record to test
item = 0

# plot image
plot.imshow(our_own_dataset[item][1:].reshape(28,28), cmap='Greys', interpolation='None')


# correct answer is first value
correct_label = our_own_dataset[item][0]
# data is remaining values
inputs = our_own_dataset[item][1:]

# query the network
outputs = n.query(inputs)
print (outputs)

# the index of the highest value corresponds to the label
label = np.argmax(outputs)
print("network says ", label)
# append correct or incorrect to list
if (label == correct_label):
    print ("match!")
else:
    print ("no match!")
    pass
plot.show()