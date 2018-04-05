from KMeans.KMeans import KMeans
from Kohonen.Kohonen import Kohonen

kmeans_algorithm = KMeans(5)
kmeans_algorithm.initialize_data("Iris.csv")
kmeans_algorithm.initialize_centroids()
kmeans_algorithm.algorithm()

kohonen_algorithm = Kohonen(5)
kohonen_algorithm.initialize_data("Seeds.csv")
kohonen_algorithm.initialize_neurons()
kohonen_algorithm.algorithm()
