"""
Trains a simple neural network on MNIST classification, using my implementation.
Neuron-replacement happend after every epoch.
"""

from data.FashionMNIST import *
from helper.hyperparameter import *
from helper.NN import *
from helper.activations import *
from helper.losses import *
from helper.NeuralSleep import NeuralSleep
from helper.dataset_tools import *
import numpy as np

np.set_printoptions(linewidth=np.inf)

def train_model():
    data = load_data()
    (x_train, y_train), (x_test, y_test) = prepare_data_for_tooc(data)

    hp = Hyperparameter()
    hp.epochs = 50
    hp.lr = 2
    hp.r = 0.00001

    mse = MeanSquareCostFunction()
    classifier = SimpleNeuronalNetwork((784, 20, 10), sigmoidActivation, sigmoidDerivation, mse)

    sleep = NeuralSleep(classifier)

    for e in range(hp.epochs):
        shuffle_data((x_train, y_train))

        # neuron replacement
        sleep.sleep((x_train, y_train))

        for b in range(x_train.shape[0] // hp.batch_size):
            for s in range(hp.batch_size):
                classifier.trainNetwork(x_train[b * hp.batch_size + s], y_train[b * hp.batch_size + s])
            classifier.applyChanges(hp.lr, hp.lr, hp.r)

        accuracy = 0
        tests = 0

        for s in range(x_test.shape[0]):
            output = classifier.getOutput(x_test[s, : ])
            if np.argmax(output) == np.argmax(y_test[s, : ]):
                accuracy += 1
            tests += 1

        print(accuracy / tests, flush=True)

train_model()