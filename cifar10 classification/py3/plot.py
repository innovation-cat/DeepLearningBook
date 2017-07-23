import matplotlib.pyplot as plt
import numpy

x = numpy.arange(300)

error_softmax, error_mlp, error_sda, error_cnn = [], [], [], []

plt.xlabel('epoch')
plt.ylabel('error rate')

with open("softmax.txt", "rb") as fin:
	for line in fin:
		error_softmax.append(float(line.strip()))
		
with open("mlp.txt", "rb") as fin:
	for line in fin:
		error_mlp.append(float(line.strip()))
		
with open("sda.txt", "rb") as fin:
	for line in fin:
		error_sda.append(float(line.strip()))
		
with open("cnn.txt", "rb") as fin:
	for line in fin:
		error_cnn.append(float(line.strip()))
		
		
plt.plot(x, error_softmax, 'r', label="softmax regression")
plt.plot(x, error_mlp, 'g', label="multilayer perceptron")
plt.plot(x, error_sda, 'purple', label="stacked denoising autoencoder")
plt.plot(x, error_cnn, 'b', label="convolutional neural network")

plt.legend()
plt.show()