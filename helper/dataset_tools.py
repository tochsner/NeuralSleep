import numpy as np

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
