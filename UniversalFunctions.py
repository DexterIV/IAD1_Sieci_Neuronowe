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


def plot_neuron(cluster, value_x_index, value_y_index):
    pyplot.plot(cluster.centroid.weights[value_x_index],
                cluster.centroid.weights[value_y_index],
                'ro')


def plot_centroid(cluster, value_x_index, value_y_index):
    pyplot.plot(cluster.centroid.position.values[value_x_index],
                cluster.centroid.position.values[value_y_index],
                'ro')


def plot_cluster(cluster, color, value_x_index, value_y_index, window_name):
    x = []
    y = []
    for j in range(len(cluster.data)):
        x.append(cluster.data[j].values[value_x_index])
        y.append(cluster.data[j].values[value_y_index])
    pyplot.plot(x, y, color + 'x')
    if window_name == 'KMeans algorithm':
        plot_centroid(cluster, value_x_index, value_y_index)
    else:
        plot_neuron(cluster, value_x_index, value_y_index)


def plot_all_clusters(clusters, iteration, window_name, number_of_attributes):
    pyplot.figure(window_name)
    colors = ['b', 'g', 'c', 'xkcd:orchid', 'y', 'k', 'tab:olive', 'tab:pink', 'xkcd:coral', 'xkcd:indigo']
    number_of_subplots = number_of_attributes / 2
    if number_of_attributes % 2 == 1:
        number_of_subplots += 1
    for i in range(0, number_of_attributes, 2):
        pyplot.suptitle('iteration no. ' + str(iteration + 1))
        pyplot.subplot(number_of_subplots, 1, i / 2 + 1)
        for j in range(len(clusters)):
            if number_of_attributes == i + 1:
                plot_cluster(clusters[j], colors[j], i - 1, i, window_name)
            else:
                plot_cluster(clusters[j], colors[j], i, i + 1, window_name)
            pyplot.grid (axis='both', color='black', which='major', linestyle='--', linewidth=1)
    pyplot.show()
