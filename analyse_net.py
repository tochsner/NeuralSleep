"""
Implements several functions to analyze a trained neural network.
"""

from data.MNIST import *
from helper.NN import *
from helper.activations import *
from helper.losses import *
from helper.dataset_tools import *
from helper.NeuralSleep import NeuralSleep


"""
Saves the weights of model leaving the neurons from the layer-th layer in path.
"""
def save_weights(model, path, layer):
    with open(path + "/weights layer " + str(layer) + ".csv", "w+") as file:
        for neuron in range(model.Size[layer]):
            s = ""
            for weight in range(model.Size[layer + 1]):
                s += str(model.Weights[layer][neuron, weight]) + ","
            file.write(s + "\n")

"""
Saves the activations neurons in the layer-th layer for different labels.
"""
def determine_neuron_activations(model, path, layer):
    data = load_data()
    data = prepare_data_for_tooc(data)[1]
    grouped_data = group_data(data)
    num_labels = len(grouped_data)

    num_neurons = model.Size[layer]
    num_samples = [len(l) for l in grouped_data]

    activations = np.zeros((num_labels, num_neurons))

    for l in range(num_labels):
        for sample in grouped_data[l]:
            model.getOutput(sample)
            activations[l] += model.Neurons[layer] / num_samples[l]

    with open(path + "/activations of layer " + str(layer) + ".csv", "w+") as file:
        for n in range(num_neurons):
            s = ""
            for l in range(num_labels):
                s += str(activations[l, n]) + ","
            file.write(s + "\n")

"""
Saves the activations neurons, summed up with the outgoing weights, in the layer-th layer for different labels.
"""
def determine_neuron_activations_weighted(model, path, layer):
    data = load_data()
    data = prepare_data_for_tooc(data)[1]
    grouped_data = group_data(data)
    num_labels = len(grouped_data)

    num_neurons = model.Size[layer]
    num_samples = [len(l) for l in grouped_data]

    activations = np.zeros((num_labels, num_neurons))

    for l in range(num_labels):
        for sample in grouped_data[l]:
            model.getOutput(sample)
            activations[l] += model.Neurons[layer].dot(np.abs(model.Weights[layer].T)) / model.Size[layer + 1] / num_samples[l]

    with open(path + "/activations weighted of layer " + str(layer) + ".csv", "w+") as file:
        for n in range(num_neurons):
            s = ""
            for l in range(num_labels):
                s += str(activations[l, n]) + ","
            file.write(s + "\n")

"""
Saves the impact of the neurons in the layer-th layer for different labels.
"""
def determine_neuron_impact(model, path, layer):
    data = load_data()
    (x_test, y_test) = prepare_data_for_tooc(data)[1]
    num_samples = len(x_test)

    num_neurons = model.Size[layer]

    activations = np.zeros((num_samples, num_neurons))

    for s in range(num_samples):
        model.getOutput(x_test[s])
        activations[s] = model.Neurons[layer].dot(np.abs(model.Weights[layer].T)) / model.Size[layer + 1]

    std_of_activations = np.std(activations, axis=0)

    with open(path + "/impact of layer " + str(layer) + ".csv", "w+") as file:
        s = ""
        for n in range(num_neurons):
            s += str(std_of_activations[n]) + ","
        file.write(s)

"""
Saves the impact of the neurons in the layer-th layer for different labels.
"""
def determine_neuron_impact_between_labels(model, path, layer):
    data = load_data()
    data = prepare_data_for_tooc(data)[1]
    grouped_data = group_data(data)
    num_labels = len(grouped_data)

    num_neurons = model.Size[layer]
    num_samples = [len(l) for l in grouped_data]

    activations = np.zeros((num_labels, num_neurons))

    for l in range(num_labels):
        for sample in grouped_data[l]:
            model.getOutput(sample)
            activations[l] += model.Neurons[layer].dot(np.abs(model.Weights[layer].T)) / model.Size[layer + 1] / num_samples[l]

    std_of_activations = np.std(activations, axis=0)

    with open(path + "/impact (between-label) of layer " + str(layer) + ".csv", "w+") as file:
        s = ""
        for n in range(num_neurons):
            s += str(std_of_activations[n]) + ","
        file.write(s)

"""
Saves the impact of the neurons in the layer-th layer for the same label.
"""
def determine_neuron_impact_in_label(model, path, layer):
    data = load_data()
    data = prepare_data_for_tooc(data)[1]
    grouped_data = group_data(data)
    num_labels = len(grouped_data)

    num_neurons = model.Size[layer]
    num_samples = [len(l) for l in grouped_data]

    std_of_activations = np.zeros((num_labels, num_neurons))

    for l in range(num_labels):
        activations_of_label = np.zeros((num_samples[l], num_neurons))

        for s in range(num_samples[l]):
            model.getOutput(grouped_data[l][s])
            activations_of_label[s] = model.Neurons[layer].dot(np.abs(model.Weights[layer].T)) / model.Size[layer + 1]

        std_of_activations[l] = np.std(activations_of_label, axis=0)

    std_of_activations = np.mean(std_of_activations, axis = 0)

    with open(path + "/impact (in-label) in layer " + str(layer) + ".csv", "w+") as file:
        s = ""
        for n in range(num_neurons):
            s += str(std_of_activations[n]) + ","
        file.write(s)

def print_layerwise_entropy(model, layer):
    data = load_data()
    (x_train, y_train), (x_test, y_test) = prepare_data_for_tooc(data)

    neural_sleep = NeuralSleep(model)

    print(neural_sleep.calculate_average_layer_entropy(x_train, layer))
    print(neural_sleep.calculate_average_layer_entropy(x_test, layer))


model = SimpleNeuronalNetwork((784, 10, 10), sigmoidActivation, sigmoidDerivation, MeanSquareCostFunction())
model.load("saved_models/784-10-10 Baseline")
print_layerwise_entropy(model, 1)
print_layerwise_entropy(model, 2)
