from KMeans.KMeans import KMeans
from Kohonen.Kohonen import Kohonen
from ImageCompresser.ImageCompresser import ImageCompresser


kmeans_algorithm = KMeans(3, True, 25)
kmeans_algorithm.initialize_data("Seeds.csv")
kmeans_algorithm.initialize_centroids()
kmeans_algorithm.algorithm()
kmeans_algorithm.show_error()

kohonen_algorithm = Kohonen()
kohonen_algorithm.initialize_data("Seeds.csv")
kohonen_algorithm.initialize_neurons()
kohonen_algorithm.algorithm()

#compress = ImageCompresser(64, 2, "SourceImages\Duck.png", "ResultImages\Duck64neur2iter2.png")
#compress.algorithm()
