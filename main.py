from KMeans.KMeans import KMeans
from Kohonen.Kohonen import Kohonen

kmeans_algorithm = KMeans(3)
#kmeans_algorithm.initialize_data("iris.csv")
#kmeans_algorithm.initialize_centroids()
#kmeans_algorithm.algorithm()

kohonen_algorithm = Kohonen(3)
kohonen_algorithm.initialize_data("iris.csv")
kohonen_algorithm.initialize_neurons()
kohonen_algorithm.algorithm()
