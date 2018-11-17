import numpy as np
from .NN import SimpleNeuronalNetwork


class NeuralSleep:
    def __init__(self, model : SimpleNeuronalNetwork):
        self.model = model

    def get_probabilities(self, a):
        bins = np.arange(0, 1.1, 0.1)
        histogram = np.histogram(a, bins=bins)[0] / a.shape[0]
        p = histogram[(a * 10).astype(int)]
        return p

    def calculate_average_layer_entropy(self, samples, layer):
        num_samples = len(samples)

        activations = np.zeros((num_samples, self.model.Size[layer]))

        for s in range(num_samples):
            self.model.getOutput(samples[s])
            activations[s] = self.model.Neurons[layer]

        # bin activations
        activations = np.round(np.minimum(activations, 0.9), 1)

        probabilities = np.apply_along_axis(self.get_probabilities, axis=0, arr=activations)
        information = -np.log10(probabilities)
        information[information == np.inf] = 0
        entropy = np.mean(information, axis=0)

        return entropy
