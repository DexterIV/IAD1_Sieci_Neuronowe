import matplotlib.pyplot as pyplot
import math


def copy_values(source, target):
    for i in range(len(source)):
        target[i] = source[i]


def distance(position1, position2):
    dist = 0

    if len(position1.values) != len(position2.values):
            return -1.0

    for i in range(len(position1.values)):
        dist += (position1.values[i] - position2.values[i]) ** 2

    dist = math.sqrt(dist)
    return dist


def clear_clusters(clusters):
    for i in range(len(clusters)):
        clusters[i].data.clear()


def plot_cluster(cluster, color, value_x_index, value_y_index):
    x = []
    y = []
    for j in range(len(cluster.data)):
        x.append(cluster.data[j].values[value_x_index])
        y.append(cluster.data[j].values[value_y_index])
    pyplot.plot(x, y, color + 'x')
    pyplot.plot(cluster.neuron.position.values[value_x_index],
                cluster.neuron.position.values[value_y_index],
                'ro')


def plot_all_clusters(clusters, iteration):
    pyplot.figure('KMeans algorithm')
    pyplot.subplot(211)
    colors = ['b', 'g', 'm']
    for j in range(len(clusters)):
        plot_cluster(clusters[j], colors[j], 0, 1)
    pyplot.title('iteration no. ' + str(iteration + 1))
    pyplot.grid(axis='both', color='black', which='major', linestyle='--', linewidth=1)
    pyplot.subplot(212)
    for j in range(len(clusters)):
        plot_cluster(clusters[j], colors[j], 2, 3)
    pyplot.grid(axis='both', color='black', which='major', linestyle='--', linewidth=1)
    pyplot.show()
