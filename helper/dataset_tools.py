import numpy as np
import random

"""
Returns the dataset grouped after the different labels. returns (labels, num_samples_per_class, x) (as nested lists)
"""
def group_data(data):
    x_data, y_data = data

    num_labels = y_data.shape[1]
    num_samples = x_data.shape[0]

    grouped_data = [[] for label in range(num_labels)]

    for sample in range(num_samples):
        label = np.argmax(y_data[sample])

        grouped_data[label].append(x_data[sample])

    return grouped_data

def shuffle_data(data):
    x_data, y_data = data

    seed = random.randint(0, 1000)

    np.random.seed(seed)
    np.random.shuffle(x_data)
    np.random.seed(seed)
    np.random.shuffle(y_data)
