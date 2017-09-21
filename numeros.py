from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

neuralNetwork = buildNetwork(25, 15, 10, bias=True)

dataset = SupervisedDataSet(25,10)

dataset.addSample((0,1,1,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,1,1,0),(1,0,0,0,0,0,0,0,0,0)) 
dataset.addSample((0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0),(0,1,0,0,0,0,0,0,0,0)) 
dataset.addSample((0,1,1,1,0,0,0,0,1,0,0,1,1,1,0,0,1,0,0,0,0,1,1,1,0),(0,0,1,0,0,0,0,0,0,0)) 
dataset.addSample((0,1,1,1,0,0,0,0,1,0,0,0,1,1,0,0,0,0,1,0,0,1,1,1,0),(0,0,0,1,0,0,0,0,0,0)) 
dataset.addSample((0,1,0,1,0,0,1,0,1,0,0,1,1,1,0,0,0,0,1,0,0,0,0,1,0),(0,0,0,0,1,0,0,0,0,0)) 
dataset.addSample((0,1,1,1,0,0,1,0,0,0,0,1,1,1,0,0,0,0,1,0,0,1,1,1,0),(0,0,0,0,0,1,0,0,0,0)) 
dataset.addSample((0,1,0,0,0,0,1,0,0,0,0,1,1,1,0,0,1,0,1,0,0,1,1,1,0),(0,0,0,0,0,0,1,0,0,0)) 
dataset.addSample((0,1,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0),(0,0,0,0,0,0,0,1,0,0)) 
dataset.addSample((0,1,1,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,1,1,0),(0,0,0,0,0,0,0,0,1,0)) 
dataset.addSample((0,1,1,1,0,0,1,0,1,0,0,1,1,1,0,0,0,0,1,0,0,1,1,1,0),(0,0,0,0,0,0,0,0,0,1)) 


trainer = BackpropTrainer(neuralNetwork, dataset=dataset, learningrate=0.03, momentum=0.06)

for i in range(1, 2000):
    error = trainer.train()


    if i % 1000 == 0:
        print('Erro na interação ', i, " é: ", error)
        print(neuralNetwork.activate([0,1,1,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,1,1,0]))
        print(neuralNetwork.activate([0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0]))
        print(neuralNetwork.activate([0,1,1,1,0,0,0,0,1,0,0,1,1,1,0,0,1,0,0,0,0,1,1,1,0]))
        print(neuralNetwork.activate([0,1,1,1,0,0,0,0,1,0,0,0,1,1,0,0,0,0,1,0,0,1,1,1,0]))
        print(neuralNetwork.activate([0,1,0,1,0,0,1,0,1,0,0,1,1,1,0,0,0,0,1,0,0,0,0,1,0]))
        print(neuralNetwork.activate([0,1,0,0,0,0,1,0,0,0,0,1,1,1,0,0,1,0,1,0,0,1,1,1,0])) 
        print(neuralNetwork.activate([0,1,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0])) 
        print(neuralNetwork.activate([0,1,1,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,1,1,0])) 
        print(neuralNetwork.activate([0,1,1,1,0,0,1,0,1,0,0,1,0,1,0,0,1,0,1,0,0,1,1,1,0]))
        print(neuralNetwork.activate([0,1,1,1,0,0,1,0,1,0,0,1,1,1,0,0,0,0,1,0,0,1,1,1,0]))


print("\nSOLUCOES E RESULTADOS\n")


array = neuralNetwork.activate([0,1,0,1,0,0,1,0,1,0,0,1,1,1,0,0,0,0,1,0,0,0,0,1,0])
valor = ''
for i in range(0, 10):
	if(array[i] > 0.5):
		valor += '1'
	else:
		valor += '0'

print (valor)

if(valor == '1000000000'):
	print ("O número é 0!")

if(valor == '0100000000'):
	print ("O número é 1!")

if(valor == '0010000000'):
	print ("O número é 2!")

if(valor == '0001000000'):
	print ("O número é 3!")

if(valor == '0000100000'):
	print ("O número é 4!")

if(valor == '0000010000'):
	print ("O número é 5!")

if(valor == '0000001000'):
	print ("O número é 6!")

if(valor == '0000000100'):
	print ("O número é 7!")

if(valor == '0000000010'):
	print ("O número é 8!")

if(valor == '0000000001'):
	print ("O número é 9!")

