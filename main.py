from KMeans.KMeans import KMeans
from Kohonen.Kohonen import Kohonen

kmeans_algorithm = KMeans(3,64,0.01)
kmeans_algorithm.initialize_data("Abalone.csv")
kmeans_algorithm.initialize_centroids()
kmeans_algorithm.algorithm()

kohonen_algorithm = Kohonen(3,64,0.5,20,1.5,0.01)
kohonen_algorithm.initialize_data("Abalone.csv")
kohonen_algorithm.initialize_neurons()
kohonen_algorithm.algorithm()
