import numpy as np  
import math       
import itertools   
import io         
import scipy as sp     
from scipy import stats                                                                                           


class BaseNetwork:
    def __init__(self):
        return None    
    def getOutput(self, inputValues):
        return None    
    def getDerivatives(self, outputDerivatives):
        return None
    def trainNetwork(self, inputValues, correctOutputValues):
        return None
    def applyChanges(self, learningRate, biasLearningRate, regularization):
        return None


"""
Implements a vanilla feed-foward neural network, trained with batch-gradient-descent.
Uses l2-regularization to minimize overfitting. 
"""
class SimpleNeuronalNetwork(BaseNetwork):
    def __init__(self, size, activationFunction, activationDerivative, costFunction):
        BaseNetwork.__init__(self)

        if len(size) < 2 or min(size) == 0:
            raise ValueError("Size of network is not valid.")            
        
        self.InputNeuronsCount = size[0]
        self.OutputNeuronsCount = size[-1]
        self.Size = size
        
        self.ActivationFunction = activationFunction        
        self.ActivationDerivative = activationDerivative

        self.CostFunction = costFunction

        # Initialize network
        maxLayerSize = max(size)
        layerCount = len(size)
        self.LayerCount = layerCount

        self.Neurons = [np.zeros((x)) for x in size[:]]
        self.Bias = [np.zeros((x)) for x in size[:]]
        self.Weights = [np.random.randn(size[x],size[x+1]) for x in range(0, len(size)-1)]

        self.WeightsTrainable = [np.ones((size[x], size[x+1])) for x in range(0, len(size)-1)]

        self.NewWeights = [np.zeros((size[x],size[x+1])) for x in range(0, len(size)-1)]
        self.NewBias = [np.zeros((x)) for x in size[:]]
        self.BatchSize = 0

        self.Derivatives = [np.zeros((x)) for x in size[:]]        

    def load(self, path):
        # Load weights and bias from the files in 'path'
        BaseNetwork.__init__(self)

        self.Size = np.load(path + "/size.npy")
        self.Weights = np.load(path + "/weights.npy")
        self.Bias = np.load(path + "/bias.npy")        
        
        self.InputNeuronsCount = self.Size[0]
        self.OutputNeuronsCount = self.Size[-1]      

        print(self.Size) 

        # Initialize network
        layerCount = len(self.Size)
        self.LayerCount = layerCount

        self.NewWeights = [np.zeros((self.Size[x],self.Size[x+1])) for x in range(0, len(self.Size)-1)]
        self.NewBias = [np.zeros((x)) for x in self.Size[:]]
        self.Batchsize = 0

        self.Derivatives = [np.zeros((x)) for x in self.Size[:]] 

    def save(self, path):
        np.save(path + "/weights", self.Weights)
        np.save(path + "/bias", self.Bias)
        np.save(path + "/size", self.Size)

    def trainNetwork(self, inputValues, correctOutputValues):
        # feed forward
        self.getOutput(inputValues)

        # backpropagate and return gradient
        return self.getDerivatives(self.CostFunction.get_derivatives(self.Neurons[-1], correctOutputValues))

    """
    Calculates the output for a specific input.
    """
    def getOutput(self, inputValues):        
        self.Neurons[0] = inputValues 

        for i in range(self.LayerCount - 1):                                                           
            self.Neurons[i+1] = self.ActivationFunction(self.Weights[i].T.dot(self.Neurons[i]) + self.Bias[i+1])                        

        return self.Neurons[-1]

    """
    Backpropagates through the network, but doesn't yet change the weights.
    """
    def getDerivatives(self, outputDerivatives):
        self.Derivatives[-1] = self.ActivationDerivative(self.Neurons[-1]) * outputDerivatives                                                      
        
        # Change bias of output
        self.NewBias[-1] += self.Derivatives[-1]         
       
        for i in range(self.LayerCount - 2, 0, -1):    
            # Change weights
            self.NewWeights[i] += (self.Neurons[i][np.newaxis].T * self.Derivatives[i+1])                                                                                          

            # Collect derivatives
            self.Derivatives[i] = self.Weights[i].dot(self.Derivatives[i+1])
            # Compute derivative in respect to the input of the Neuron
            self.Derivatives[i] *= self.ActivationDerivative(self.Neurons[i]
                                                             )
            # Change bias
            self.NewBias[i] += self.Derivatives[i]                                    

        self.NewWeights[0] += (self.Neurons[0][np.newaxis].T * self.Derivatives[1])
        self.Derivatives[0] = self.Weights[0].dot(self.Derivatives[1])

        self.BatchSize += 1              
        return self.Derivatives[0]


    def unFreezeNeuron(self, layer, neuronIndex):
        self.WeightsTrainable[layer - 1][:, neuronIndex] = 1
        self.WeightsTrainable[layer][neuronIndex, :] = 1

    def freezeAllWeights(self):
        self.WeightsTrainable = [0 * x for x in self.WeightsTrainable]

    def unFreezeAllWeights(self):
        self.WeightsTrainable = [0 * x + 1 for x in self.WeightsTrainable]


    """
    Changes the weights and biases after a batch
    """
    def applyChanges(self, learningRate, biasLearningRate, regularization):
        self.Weights = [(w - nw * (learningRate / self.BatchSize) * tb - regularization * w)
                        for w, nw, tb in zip(self.Weights, self.NewWeights, self.WeightsTrainable)]
        self.Bias = [(b - nb * (biasLearningRate / self.BatchSize))
                     for b, nb in zip(self.Bias, self.NewBias)]

        self.NewWeights = [np.zeros(x.shape) for x in self.NewWeights]
        self.NewBias = [np.zeros(x.shape) for x in self.NewBias]                   

        self.BatchSize = 0            

    def evaluate(self, test_data):               
        test_results = [(np.argmax(self.getOutput(x[:,0])), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
