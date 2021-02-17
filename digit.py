import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import csv


class neuralNetwork :
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningrate):
        self.inodes = inputNodes #输入层节点
        self.hnodes = hiddenNodes #隐藏层节点
        self.onodes = outputNodes #输出层节点

        self.lr = learningrate

        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)

        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)

        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)),np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        pass


    def query(self, inputs_list):

        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)

        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)

        final_outputs = self.activation_function(final_inputs)

        return final_outputs
        pass

input_nodes = 784
hidden_nodes = 500
output_nodes = 10

learning_rate = 0.1


n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data_file = open("train.csv", "r")
training_data_list = training_data_file.readlines()
training_data_file.close()
epochs = 10

for e in range(epochs):
    for record in training_data_list[1:]:
        all_values = record.split(",")
        inputs = (np.asfarray(all_values[1:])/ 255.0 * 0.99) + 0.01
        targets = np.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        inputs_plusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28, 28), 10, cval=0.01, order=1,reshape=False)
        n.train(inputs_plusx_img.reshape(784), targets)
        inputs_minusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28, 28), -10, cval=0.01, order=1,reshape=False)
        n.train(inputs_minusx_img.reshape(784), targets)
        pass
    pass
test_data_file = open("test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
scorecard = []
with open("sample_submission.csv","w",newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["ImageId","Label"])
    i = 1
    for record in test_data_list[1:]:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values) / 255.0 * 0.99) + 0.01
        outputs = n.query(inputs)
        label = np.argmax(outputs)
        writer.writerow([i,label])
        print(i,label)
        i += 1
        pass
