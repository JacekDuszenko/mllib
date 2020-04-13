"""
kmeans algorithm implementation specific for data compression.
This module is also executable as a script, which serves the purpose
of compressing an image by implementing vector quantization which leverages
kmeans algorithm.
"""

import copy
import sys
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

RGBVector = List[int]
RGBTuple = Tuple[int, int, int]


def kmeans(vectors: np.array, k: int, initializer_func, metric_func, epsilon=.5):
    """
    Creates a mapping of every vector in the original image to its centroid.

    :param vectors: input image represented as dwo dimensional array of rgb vectors
    :param k: - number of clusters
    :param initializer_func: - function that creates the initial distribution of clusters centroids
    :param metric_func: - function that measures the distance between vectors.
    :param epsilon - This parameter is a boundary
    error value that determines when to end the iteration.

    :return quantized image as flattened ndarray
    """
    flat_vectors = np.concatenate(vectors)
    number_of_vectors = flat_vectors.shape[0]

    centroids = initializer_func(flat_vectors, k)
    prev_distortion = 0
    cur_distortion = 0
    while True:
        clusters = {tuple(c): [] for c in centroids}
        for idx, vec in enumerate(flat_vectors):
            _, best_centroid = find_best_centroid(vec, centroids, metric_func)
            clusters[tuple(best_centroid)].append((idx, vec))
        cur_distortion = calculate_distortion(clusters, number_of_vectors, metric_func)
        if abs(prev_distortion - cur_distortion) / cur_distortion < epsilon:
            return create_quantized_image(clusters, flat_vectors.shape, number_of_vectors)
        centroids = update_centroids(clusters)
        prev_distortion = cur_distortion


def create_quantized_image(clusters, original_shape, number_of_vectors):
    """
    reverse the dictionary of type [Centroid, List(Vector)] into dictionary of type [Vector, Centroid]
    it will be used later on to replace values in the original image with those obtained from quantization.

    MSE (mean square error) -> 1/N  *  (SUM{( x_n - y_n)^2 }, where n goes from 1 to N - for each pixel in the image
    SNR (signal to noise ratio) -> formula taken from "Handbook of Data Compression by D. Salomon, G. Motta"
    note that it is different from Maciej Gebala's lecture slide. Formula is

    """
    sqs = 0
    sigsum = 0
    quantized_vectors = np.empty(original_shape)
    for centroid, vecs in clusters.items():
        for index, vec in vecs:
            quantized_vectors[index] = centroid
            sqs += (centroid - vec) ** 2
            sigsum += vec ** 2
    mse = sum(sqs) / number_of_vectors
    rmse = np.sqrt(mse)
    avg_sigsum = sum(sigsum) / number_of_vectors
    r_avg_sigsum = np.sqrt(avg_sigsum)
    snr = r_avg_sigsum / rmse
    print('mse: ', mse)
    print('snr: ', snr)
    return quantized_vectors


def calculate_distortion(clusters, number_of_vectors, distance_f):
    """
    calculates the mse of quantization
    """
    sqs = 0
    for centroid, vecs in clusters.items():
        c_vec = np.array(centroid)
        for vec_and_index_tuple in vecs:
            vec_val = vec_and_index_tuple[0]
            sqs += distance_f(c_vec, vec_val)
    mse = int(sqs / number_of_vectors)
    return mse


def find_best_centroid(vec: RGBVector, centroids: List[RGBVector], distance_f):
    """
    finds the closest centroid for given vector
    can't use native min function - will repair in the future
    """
    return min([(distance_f(vec, c), c) for c in centroids], key=lambda x: x[0])


def update_centroids(clusters):
    """
    creates the most centric point for each of the cluster
    """
    new_centroids = []
    for _, vector_and_index_tuples in clusters.items():
        vectors = get_vects(vector_and_index_tuples)
        mean_centroid = np.mean(vectors, axis=0).astype(int)
        new_centroids.append(mean_centroid)
    return new_centroids


def get_vects(vec_index_tuple_list):
    return list(map(list, zip(*vec_index_tuple_list)))[1]


def k_random_vectors(flat_vectors, k):
    """
    Returns initial distribution of k vectors. Chooses random vectors from supplied set.
    """
    vectors_number = flat_vectors.shape[0]
    random_idxs = np.random.randint(vectors_number, size=k)
    random_vectors = flat_vectors[random_idxs, :]
    return random_vectors


def taxi(a, b):
    """
    taxi distance between two equally shaped vectors.
    """
    return np.sum(np.abs(a - b))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print('Usage: 5list.py IN_IMG OUT_IMG BITS_PER_COLOR')
        sys.exit(1)

    inf_path = sys.argv[1]
    of_path = sys.argv[2]
    bpc = int(sys.argv[3])
    if not 0 <= bpc <= 8:
        print("bpc must be in [0,8]")
        sys.exit(1)

    im = np.array(Image.open(inf_path))
    out_im = im.copy()
    K = 2 ** (3 * bpc)
    quantized = kmeans(im, K, k_random_vectors, taxi)
    originally_shaped = quantized.reshape(im.shape).astype(np.uint8)
    new_img = Image.fromarray(originally_shaped)
    new_img.save(of_path)
