
import matplotlib.pyplot as plt
import numpy

x = numpy.arange(20)

a = 0.0419022056063
k3, k10, pk3, pk15, dynamic = [], [], [], [], []

plt.title('Performance')
plt.xlabel('Epoch')
plt.ylabel('Error_Rate')

with open("output_k3_lr0.01.txt", "rb") as fin:
	for line in fin:
		k3.append(float(line.strip())/a)
		
with open("output_k10_lr0.01.txt", "rb") as fin:
	for line in fin:
		k10.append(float(line.strip())/a)
		
with open("output_persistent_k3_lr0.01.txt", "rb") as fin:
	for line in fin:
		pk3.append(float(line.strip())/a)
		
with open("output_persistent_k15_lr0.01.txt", "rb") as fin:
	for line in fin:
		pk15.append(float(line.strip())/a)

with open("output_persistent_k3_lr0.1.txt", "rb") as fin:
	for line in fin:
		dynamic.append(float(line.strip())/a)
		
plt.plot(x, k3, 'r', label="lr=0.01, cd_k=3")
plt.plot(x, k10, 'g', label="lr=0.01, cd_k=10")
plt.plot(x, pk3, 'b', label="lr=0.01, cd_k=3, persistent")
plt.plot(x, pk15, 'purple', label="lr=0.01, cd_k=15, persistent")
plt.plot(x, dynamic, 'magenta', label="adaptive learning rate and cd_k")
plt.legend()
plt.show()