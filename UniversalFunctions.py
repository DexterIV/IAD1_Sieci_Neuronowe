import matplotlib.pyplot as pyplot
import math
# from numba import vectorize


# @vectorize(['float32(float32, float32, float32)'], target='cuda')
def dist_add_pow(a, b, res):
    return res + ((a - b) ** 2)


def distance_for_comparison(position1, position2):
    dist = 0

    if len(position1) != len(position2):
        raise ValueError

    for i in range(len(position1)):
        dist = dist_add_pow(position1[i], position2[i], dist)

    return dist


def distance(position1, position2):
    dist = 0

    if len(position1) != len(position2):
        return 0

    for i in range(len(position1)):
        dist = dist_add_pow(position1[i], position2[i], dist)

    dist = math.sqrt(dist)
    return dist


def copy_values(source):
    copy = []
    for i in range(len(source)):
        copy.append(source[i])
    return copy


def convert_3d_to_1d_list(three_dimensional_list):
    tmp = []
    for i in range(len(three_dimensional_list)):
        for j in range(len(three_dimensional_list[i])):
            if isinstance(three_dimensional_list[i][j], tuple):
                for k in range(len(three_dimensional_list[i][j])):
                    tmp.append(three_dimensional_list[i][j][k])
            else:
                tmp.append(three_dimensional_list[i][j])
    return tmp


def convert_1d_list_to_3d_chunk(one_dimensional_list, chunk_size):
    tmp = []
    for i in range(chunk_size):
        tmp1 = []
        for j in range(chunk_size):
            tmp2 = (one_dimensional_list[i * 3 + j * 12], one_dimensional_list[i * 3 + j * 12 + 1],
                    one_dimensional_list[i * 3 + j * 12 + 2])
            tmp1.append(tmp2)
        tmp.append(tmp1)
    return tmp


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


def plot_cluster(clusters, color, value_x_index, value_y_index, index, xLabel, yLabel):
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
        pyplot.suptitle(window_name + ' iteration no. ' + str(iteration + 1))
        pyplot.subplot(math.sqrt(number_of_subplots), 2, i / 2 + 1)
        for j in range(len(clusters)):
            if number_of_attributes == i + 1:
                plot_cluster(clusters, colors[j], i - 1, i, j, labels[i - 1], labels[i])
            else:
                plot_cluster(clusters, colors[j], i, i + 1, j, labels[i], labels[i + 1])
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


