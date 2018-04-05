import matplotlib.pyplot as pyplot
import math


def copy_values(source, target):
    for i in range(len(source)):
        target[i] = source[i]


def distance(position1, position2):
    dist = 0

    for i in range(len(position1)):
        dist += (position1[i] - position2[i]) ** 2

    dist = math.sqrt(dist)
    return dist


def clear_clusters(clusters):
    for i in range(len(clusters)):
        clusters[i].data.clear()


def plot_neurons_path(neurons, color, value_x_index, value_y_index):
    x = []
    y = []
    for i in range(len(neurons)):
        x.append(neurons[i].weights[value_x_index])
        y.append(neurons[i].weights[value_y_index])
    pyplot.plot(x, y, color, marker='o', linestyle='solid', markersize=4)
    pyplot.plot(neurons[len(neurons) - 1].weights[value_x_index],
                neurons[len(neurons) - 1].weights[value_y_index],
                marker='o', markersize=6, color='red', linewidth=2)


def plot_centroid(cluster, color, value_x_index, value_y_index):
    pyplot.plot(cluster.centroid.position.values[value_x_index],
                cluster.centroid.position.values[value_y_index],
                color, marker='o')


def plot_cluster(clusters, color, neuron_color, value_x_index, value_y_index, neurons, index, xLabel, yLabel):
    x = []
    y = []
    for j in range(len(clusters[index].data)):
        x.append(clusters[index].data[j].values[value_x_index])
        y.append(clusters[index].data[j].values[value_y_index])
    pyplot.plot(x, y, color + 'x', markersize=5)
    pyplot.xlabel(xLabel)
    pyplot.ylabel(yLabel)


def plot_centroids(clusters, value_x_index, value_y_index):
    for index in range(len(clusters)):
        plot_centroid(clusters[index], 'r', value_x_index, value_y_index)


def plot_neurons(neurons, neuron_colors, value_x_index, value_y_index):
    for index in range(len(neurons[0])):
        neuron_positions = []
        for i in range(len(neurons)):
            neuron_positions.append(neurons[i][index])
        plot_neurons_path(neuron_positions, neuron_colors[index], value_x_index, value_y_index)


def plot_all_clusters(clusters, iteration, window_name, number_of_attributes, neurons, labels):
    pyplot.figure(window_name)
    colors = ['b', 'g', 'c', 'y', 'k', 'tab:olive', 'tab:pink', 'xkcd:coral', 'xkcd:indigo']
    neuron_colors = ['#d3d3d3', '#727272', '#8c95a3', '#bc93b9', '#000000']
    number_of_subplots = number_of_attributes / 2
    if number_of_attributes % 2 == 1:
        number_of_subplots += 0.5
    for i in range(0, number_of_attributes, 2):
        pyplot.suptitle('iteration no. ' + str(iteration + 1))
        pyplot.subplot(math.sqrt(number_of_subplots), 2, i / 2 + 1)
        for j in range(len(clusters)):
            if number_of_attributes == i + 1:
                plot_cluster(clusters, colors[j], neuron_colors[j], i - 1, i, neurons, j, labels[i - 1], labels[i])
            else:
                plot_cluster(clusters, colors[j], neuron_colors[j], i, i + 1, neurons, j, labels[i], labels[i + 1])
        if neurons is None:
            if number_of_attributes == i + 1:
                plot_centroids(clusters, i - 1, i)
            else:
                plot_centroids(clusters, i, i + 1)
        else:
            if number_of_attributes == i + 1:
                plot_neurons(neurons, neuron_colors, i - 1, i)
            else:
                plot_neurons(neurons, neuron_colors, i, i + 1)
        pyplot.grid(axis='both', color='black', which='major', linestyle='--', linewidth=1)
    pyplot.show()
