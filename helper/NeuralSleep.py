import numpy as np
from .NN import SimpleNeuronalNetwork


class NeuralSleep:
    def __init__(self, model : SimpleNeuronalNetwork):
        self.model = model

    def sleep(self, samples):
        layer_count = len(self.model.Size)
        x_data, y_data = samples

        for l in range(1, layer_count - 1):
            neuron_information = self.calculate_information_per_neuron(x_data, l)

            sorted_indices = np.argsort(neuron_information)[:10]

            for index in sorted_indices:
                self.replace_neuron(l, index)

            self.pretrain_certain_neurons(l, sorted_indices, samples)

    def pretrain_certain_neurons(self, layer, neuron_indices, samples):
        x_data, y_data = samples

        self.model.freezeAllWeights()

        for index in neuron_indices:
            self.model.unFreezeNeuron(layer, index)

        epochs_factor = 20
        batch_size = 10
        lr = 0.2

        for b in range(x_data.shape[0] // 10 // epochs_factor):
            for s in range(batch_size):
                self.model.trainNetwork(x_data[b * batch_size + s], y_data[b * batch_size + s])
            self.model.applyChanges(lr, lr, 0)

        self.model.unFreezeAllWeights()

    def replace_neuron(self, layer, index):
        # calculate statistics about ingoing and outgoing weights in this layer
        ingoing_mean = np.mean(self.model.Weights[layer - 1])
        ingoing_std = np.std(self.model.Weights[layer - 1])

        outgoing_mean = np.mean(self.model.Weights[layer])
        outgoing_std = np.std(self.model.Weights[layer])

        bias_mean = np.mean(self.model.Bias[layer])
        bias_std = np.std(self.model.Bias[layer])

        # sample new ingoing and outgoing weights for replaced neuron
        self.model.Weights[layer - 1][:, index] = np.random.normal(ingoing_mean, ingoing_std, self.model.Size[layer - 1])
        self.model.Weights[layer][index, :] = np.random.normal(outgoing_mean, outgoing_std, self.model.Size[layer + 1])
        self.model.Bias[layer][index] = np.random.normal(bias_mean, bias_std, 1)

    def calculate_information_per_neuron(self, samples, layer):
        num_samples = len(samples)

        activations = np.zeros((num_samples, self.model.Size[layer]))

        for s in range(num_samples):
            self.model.getOutput(samples[s])
            activations[s] = self.model.Neurons[layer]

        # bin activations
        activations = np.round(np.minimum(activations, 0.9), 1)

        probabilities = np.apply_along_axis(self.calculate_probabilities, axis=0, arr=activations)
        information = -np.log10(probabilities)
        information[information == np.inf] = 0
        entropy = np.mean(information, axis=0)

        return entropy

    def calculate_probabilities(self, a):
        bins = np.arange(0, 1.1, 0.1)
        histogram = np.histogram(a, bins=bins)[0] / a.shape[0]
        p = histogram[(a * 10).astype(int)]
        return p
