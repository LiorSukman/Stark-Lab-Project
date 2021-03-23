import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import math


def calc_temporal_histogram(time_lst, bins):
    ret = np.zeros(len(bins) - 1)
    for i in range(len(time_lst)):
        hist = np.histogram(time_lst - time_lst[i], bins=bins)
        ret += hist

    return hist


class TemporalFeatures(object):
    """
    TODO...
    """

    def __init__(self, resolution=2, upsample=8, bin_range=1500, cdf_range=30, jmp_min=50, jmp_max=1200):
        self.name = 'temporal features'
        self.resolution = resolution
        self.upsample = upsample
        self.bin_range = bin_range
        self.cdf_range = cdf_range
        self.jmp_min = jmp_min
        self.jmp_max = jmp_max

    def calculate_feature(self, time_lst):
        """
        inputs:
        time_lst: A list of spike timings.

        returns:
        TODO...
        """
        time_lst = np.array(time_lst)
        N = 2 * self.resolution * self.bin_range + 1
        bins = np.linspace(-self.bin_range, self.bin_range, N)
        histogram = calc_temporal_histogram(time_lst, bins)
        histogram = signal.resample(histogram, self.upsample * N)
        rhs = histogram[len(histogram) / 2:]
        start_band = rhs[:self.resolution * self.cdf_range]
        start_cdf = np.cumsum(start_band) / np.sum(start_band)

        uniform_cdf = np.linspace(0, 1, len(start_cdf))
        unif_dist = (start_cdf - uniform_cdf) / len(start_cdf)

        ach_rise_time = (start_cdf > 1 / math.e).argmax()

        mid_band = rhs[self.resolution * self.jmp_min: self.resolution * self.jmp_max + 1]
        jmp_line = np.linspace(mid_band[0], mid_band[-1], len(mid_band))
        # TODO after assuring this is ok change 5000 to number of samples and make the 50 part of the class
        ach_jmp = 50 * np.log(np.sum((mid_band - jmp_line) ** 2) / 5000)

        f, pxx = signal.periodogram(rhs, 20_000)
        centeroid = np.sum(f * pxx) / np.sum(pxx)  # TODO maybe it shoud be the || of pxx

        der_pxx = np.abs(np.gradient(pxx))  # TODO check if there really can be negative values here
        der_centeroid = np.sum(f * der_pxx) / np.sum(der_pxx)  # TODO maybe it shoud be the || of pxx

        uniform = np.ones(len(start_band)) / len(start_band)
        dkl = stats.entropy(start_cdf, uniform)  # TODO maybe on the mid-band as well?

        return unif_dist, ach_rise_time, ach_jmp, centeroid, der_centeroid, dkl

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return ["unif_dist", "rise_time", "jump", "psd_center", "der_psd_center", "dkl"]
