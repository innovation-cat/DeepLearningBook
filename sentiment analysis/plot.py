
import matplotlib.pyplot as plt
import numpy

x = numpy.arange(30)

sgd, momentum, nesterov, adadelta = [], [], [], []

plt.title('Performance')
plt.xlabel('Epoch')
plt.ylabel('Error_Rate')

with open("sgd.txt", "rb") as fin:
	for line in fin:
		sgd.append(float(line.strip()))
		
with open("momentum.txt", "rb") as fin:
	for line in fin:
		momentum.append(float(line.strip()))
		
with open("nesterov_momentum.txt", "rb") as fin:
	for line in fin:
		nesterov.append(float(line.strip()))
		
with open("adadelta.txt", "rb") as fin:
	for line in fin:
		adadelta.append(float(line.strip()))
		
plt.plot(x, sgd, 'r', label="sgd")
plt.plot(x, momentum, 'g', label="momentum")
plt.plot(x, nesterov, 'b', label="nesterov_momentum")
plt.plot(x, adadelta, 'purple', label="adadelta")
plt.legend()
plt.show()