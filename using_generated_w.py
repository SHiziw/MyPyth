import numpy as np
# load the generated weight matrix CSV file into a list

weight_file = open("mnist_dataset/weight_matrix.csv", 'r')
weight_list = weight_file.readlines()
weight_file.close()

# reading the number of input, hidden and output nodes
first_line = weight_list[0].split(',')
#print(first_line)
input_nodes = int(first_line[0])
hidden_nodes = int(first_line[1])
output_nodes = int(first_line[2])
# learning rate
learning_rate = float(first_line[3])

w_ih = np.random.normal(0.0, pow(hidden_nodes, -0.5), (hidden_nodes, input_nodes))
w_ho = np.random.normal(0.0, pow(output_nodes, -0.5), (output_nodes, hidden_nodes))
# 生成容器装已经训练好的矩阵
for e in range(1, hidden_nodes):
    #避开第一行，从之后的 hidden_nodes 行开始
    all_values = weight_list[e].split(',')
    w_ih[e - 1] = np.asfarray(all_values[0:])
    pass
for e in range(1+hidden_nodes, 1 + hidden_nodes + output_nodes):
    # 再之后的 output_nodes 行
    all_values = weight_list[e].split(',')
    w_ho[e - 1 - hidden_nodes] = np.asfarray(all_values[0:])
    pass

print('in:', input_nodes, 'hid:', hidden_nodes, 'out:', output_nodes, 'rate:', learning_rate,'\n')
print(w_ho)
