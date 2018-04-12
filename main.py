from KMeans.KMeans import KMeans
from Kohonen.Kohonen import Kohonen
from ImageCompresser.ImageCompresser import ImageCompresser


#kmeans_algorithm = KMeans()
#kmeans_algorithm.initialize_data("Iris.csv")
#kmeans_algorithm.initialize_centroids()
#kmeans_algorithm.algorithm()

#kohonen_algorithm = Kohonen()
#kohonen_algorithm.initialize_data("Seeds.csv")
#kohonen_algorithm.initialize_neurons()
#kohonen_algorithm.algorithm()

compress = ImageCompresser(128, 32, "SourceImages\Chair.png", "ResultImages\Chair128neur32iter.png")
compress.algorithm()
