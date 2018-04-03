from KMeans.KMeans import KMeans
from Kohonen.Kohonen import Kohonen

kmeans_algorithm = KMeans(3)
kmeans_algorithm.initialize_data("Seeds.csv")
kmeans_algorithm.initialize_centroids()
kmeans_algorithm.algorithm(2)

kohonen_algorithm = Kohonen(3)
kohonen_algorithm.initialize_data("Seeds.csv")
kohonen_algorithm.initialize_neurons()
kohonen_algorithm.algorithm(2)
