from PIL import Image
from KMeans.KMeans import KMeans
from UniversalFunctions import *
import numpy as np
import scipy.misc as smp


class ImageCompresser:
    def __init__(self, kmeans_clusters=64, kmeans_iterations=128, filename="lenna-uscsipi.png", chunk_size=4):
        self.dictionary = []        # its called dictionary but its actually a list
        self.chunks = []
        self.kmeans_clusters = kmeans_clusters
        self.kmeans_iterations = kmeans_iterations
        self.image = Image.open(filename)
        self.pixels = self.image.load()
        self.chunk_size = chunk_size
        self.compressed = []
        if self.image.size[0] % chunk_size != 0:
            raise ValueError

    def algorithm(self):
        self._read_chunks()
        self._fill_dictionary()
        print(self.dictionary)
        for i in range(len(self.chunks)):
            best_matching_dict_chunk = self._find_closest_entry_in_dictionary(self.chunks[i])
            self.compressed.append(best_matching_dict_chunk)
        new_pixels = self._save_chunks_as_pixels()
        img = smp.toimage(new_pixels)
        img.show()
        img.save("test.png", "PNG")

    def _read_chunks(self):
        width, height = self.image.size
        for width_iter in range(0, width, self.chunk_size):
            for height_iter in range(0, height, self.chunk_size):
                chunk = []
                for i in range(self.chunk_size):
                    chunk_row = []
                    for j in range(self.chunk_size):
                        chunk_row.append(self.pixels[width_iter + i, height_iter + j])
                    chunk.append(chunk_row)
                self.chunks.append(chunk)

    def _fill_dictionary(self):
        kmeans = KMeans(self.kmeans_clusters, True, self.kmeans_iterations)
        kmeans.initialize_data_with_chunks(self.chunks)
        kmeans.initialize_centroids()
        clusters = kmeans.algorithm()
        for i in range(len(clusters)):
            chunks = []
            for j in range(len(clusters[i].data)):
                chunks.append(convert_1d_list_to_3d_chunk(clusters[i].data[j].values, self.chunk_size))
            self.dictionary.append(self._average_similar_chunks(chunks))

    def _average_similar_chunks(self, chunks):
        chunk = chunks[0]
        for width_iter in range(len(chunks[0])):
            for height_iter in range(len(chunks[0][0])):
                avg = [0, 0, 0]
                for tuple_iter in range(len(chunks[0][0][0])):
                    for chunk_iter in range(len(chunks)):
                        avg[tuple_iter] += chunks[chunk_iter][width_iter][height_iter][tuple_iter]
                    avg[tuple_iter] /= len(chunks)
                chunk[width_iter][height_iter] = tuple(avg)
        return chunk

    def _find_closest_entry_in_dictionary(self, chunk):
        index = 0
        minimum = distance(convert_3d_to_1d_list(chunk), convert_3d_to_1d_list(self.dictionary[0]))
        for i in range(1, len(self.dictionary)):
            dist = distance(convert_3d_to_1d_list(chunk), convert_3d_to_1d_list(self.dictionary[i]))
            if dist < minimum:
                minimum = dist
                index = i
        return index

    def _save_chunks_as_pixels(self):
        width, height = self.image.size
        pixels = np.zeros((width, height, 3), dtype=np.uint8)
        for width_iter in range(0, width, self.chunk_size):
            for height_iter in range(0, height, self.chunk_size):
                for i in range(self.chunk_size):
                    for j in range(self.chunk_size):
                        chunk_index = int((width_iter - (width_iter % self.chunk_size))/self.chunk_size +
                                          (height_iter - (height_iter % self.chunk_size)) * (self.chunk_size ** 2.5))
                        tuple3 = self.dictionary[self.compressed[chunk_index]][i][j]
                        pixels[width_iter + i, height_iter + j] = tuple3
        return pixels
