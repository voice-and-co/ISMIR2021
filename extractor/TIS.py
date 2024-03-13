

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from astropy.convolution import convolve, Gaussian1DKernel
from scipy.spatial.distance import cosine, euclidean
np.seterr(all='raise')


class TIV:
    weights_symbolic = [2, 11, 17, 16, 19, 7]
    weights = [3, 8, 11.5, 15, 14.5, 7.5]

    def __init__(self, energy, vector):
        self.energy = energy
        self.vector = vector

    @classmethod
    def from_pcp(cls, pcp, symbolic=False):
        if not everything_is_zero(pcp):
            fft = np.fft.rfft(pcp, n=12)
            energy = fft[0]
            vector = fft[1:7]
            if symbolic:
                vector = ((vector / energy) * cls.weights_symbolic)
            else:
                vector = ((vector / energy) * cls.weights)
            return cls(energy, vector)
        else:
            return cls(complex(0), np.array([0, 0, 0, 0, 0, 0]).astype(complex))

    def get_vector(self):
        return np.array(self.vector)

    def dissonance(self):
        return 1 - (np.linalg.norm(self.vector) / np.sqrt(np.sum(np.dot(self.weights, self.weights))))

    def coefficient(self, ii):
        return self.mags()[ii] / self.weights[ii]

    def chromaticity(self):
        return self.mags()[0] / self.weights[0]

    def dyadicity(self):
        return self.mags()[1] / self.weights[1]

    def triadicity(self):
        return self.mags()[2] / self.weights[2]

    def diminished_quality(self):
        return self.mags()[3] / self.weights[3]

    def diatonicity(self):
        return self.mags()[4] / self.weights[4]

    def wholetoneness(self):
        return self.mags()[5] / self.weights[5]

    def mags(self):
        return np.abs(self.vector)

    def plot_tiv(self):
        titles = ["m2/M7", "TT", "M3/m6", "m3/M6", "P4/P5", "M2/m7"]
        tivs_vector = self.vector / self.weights
        i = 1
        for tiv in tivs_vector:
            circle = plt.Circle((0, 0), 1, fill=False)
            plt.subplot(2, 3, i)
            plt.subplots_adjust(hspace=0.4)
            plt.gca().add_patch(circle)
            plt.title(titles[i - 1])
            plt.scatter(tiv.real, tiv.imag)
            plt.xlim((-1.5, 1.5))
            plt.ylim((-1.5, 1.5))
            plt.grid()
            i = i + 1
        plt.show()

    @classmethod
    def euclidean(cls, tiv1, tiv2):
        return np.linalg.norm(tiv1.vector - tiv2.vector)

    @classmethod
    def cosine(cls, tiv1, tiv2):
        a = np.concatenate((tiv1.vector.real, tiv1.vector.imag), axis=0)
        b = np.concatenate((tiv2.vector.real, tiv2.vector.imag), axis=0)
        if everything_is_zero(a) or everything_is_zero(b):
            distance_computed = 0
        else:
            distance_computed = cosine(a, b)
        return distance_computed


zero_sequence = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
one_sequence = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


def compute_dissonance(chroma_audio, symbolic=False):
    dissonance = []
    for x in chroma_audio:
        if not np.array_equal(x, zero_sequence):
            dissonance.append(TIV.from_pcp(x, symbolic=symbolic).dissonance())
        else:
            dissonance.append(0.)
            #dissonance.append(zero_sequence)
    return dissonance


def parse_zeros(chroma_audio, symbolic):
    if not np.array_equal(chroma_audio, zero_sequence):
        TIV1 = TIV.from_pcp(chroma_audio, symbolic=symbolic)
    else:
        TIV1 = TIV(0, np.array([0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j, 0 + 0j]))
    return TIV1


def compute_cosine_similarity(chroma_audio1, chroma_audio2, symbolic=False):
    ans = []
    for c1, c2 in zip(chroma_audio1, chroma_audio2):
        TIV1 = parse_zeros(c1, symbolic)
        TIV2 = parse_zeros(c2, symbolic)
        distance_computed = 1 - TIV.cosine(TIV1, TIV2)
        # if TIV.cosine(TIV1, TIV2) > 1:
        #     print(TIV.cosine(TIV1, TIV2))
        ans.append(distance_computed)
    return ans


def compute_euclidean_similarity(chroma_audio1, chroma_audio2, symbolic=False):
    ans = []
    for c1, c2 in zip(chroma_audio1, chroma_audio2):
        TIV1 = parse_zeros(c1, symbolic)
        TIV2 = parse_zeros(c2, symbolic)
        distance_computed = 1 / (1 + TIV.euclidean(TIV1, TIV2))
        ans.append(distance_computed)
    return ans


def everything_is_zero(vector):
    for element in vector:
        if element != 0:
            return False
    return True


def complex_to_vector(vector):
    ans = []
    for i in range(0, vector.shape[1]):
        row1 = []
        row2 = []
        for j in range(0, vector.shape[0]):
            row1.append(vector[j][i].real)
            row2.append(vector[j][i].imag)
        ans.append(row1)
        ans.append(row2)
    return np.array(ans)


def tonal_interval_space(chroma, symbolic=False):
    centroid_vector = []
    for i in range(0, chroma.shape[1]):
        each_chroma = [chroma[j][i] for j in range(0, chroma.shape[0])]
        # print(each_chroma)
        if everything_is_zero(each_chroma):
            centroid = [0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j]
        else:
            tonal = TIV.from_pcp(each_chroma, symbolic)
            centroid = tonal.get_vector()
        centroid_vector.append(centroid)
    return complex_to_vector(np.array(centroid_vector))


def gaussian_blur(centroid_vector, sigma):
    centroid_vector = gaussian_filter(centroid_vector, sigma=sigma)
    return centroid_vector


def get_distance(centroids, dist):
    ans = [0]
    if dist == 'euclidean':
        for j in range(1, centroids.shape[1] - 1):
            sum = 0
            for i in range(0, centroids.shape[0]):
                sum += ((centroids[i][j + 1] - centroids[i][j - 1]) ** 2)
            sum = np.math.sqrt(sum)

            ans.append(sum)

    if dist == 'cosine':
        for j in range(1, centroids.shape[1] - 1):
            a = centroids[:, j - 1]
            b = centroids[:, j + 1]
            if everything_is_zero(a) or everything_is_zero(b):
                distance_computed = euclidean(a, b)
            else:
                distance_computed = cosine(a, b)
            ans.append(distance_computed)
    ans.append(0)

    return np.array(ans)


def get_peaks_hcdf(hcdf_function, rate_centroids_second, symbolic=False):
    changes = [0]
    hcdf_changes = []
    last = 0
    for i in range(2, hcdf_function.shape[0] - 1):
        if hcdf_function[i - 1] < hcdf_function[i] and hcdf_function[i + 1] < hcdf_function[i]:
            hcdf_changes.append(hcdf_function[i])
            if not symbolic:
                changes.append(i / rate_centroids_second)
            else:
              changes.append(i)
            last = i
    return np.array(changes), np.array(hcdf_changes)


def harmonic_change(chroma: list, window_size: int=2048, symbolic: bool=False,
                         sigma: int = 5, dist: str = 'euclidean'):
    chroma = np.array(chroma).transpose()
    centroid_vector = tonal_interval_space(chroma, symbolic=symbolic)

    # blur
    centroid_vector_blurred = gaussian_blur(centroid_vector, sigma)

    # harmonic distance and calculate peaks
    harmonic_function = get_distance(centroid_vector_blurred, dist)

    changes, hcdf_changes = get_peaks_hcdf(harmonic_function, window_size, symbolic)

    return changes, hcdf_changes, harmonic_function










