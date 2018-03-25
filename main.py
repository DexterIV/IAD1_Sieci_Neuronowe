from KMeans import KMeans


kmeans_algorithm = KMeans(3)
kmeans_algorithm.initialize_data("iris.csv")
kmeans_algorithm.initialize_centroids()
kmeans_algorithm.algorithm()