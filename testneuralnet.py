import numpy as np
import matplotlib.pyplot as plot
data_file = open("./mnist_dataset/mnist_train_100.csv", 'r') 
data_list = data_file.readlines()
data_file.close()
print(len(data_list))
print(data_list[0])
all_values = data_list[1].split(',')
image_array = np.asfarray(all_values[1:]).reshape((28,28))
plot.imshow(image_array,cmap='Greys', interpolation='none')
plot.show()