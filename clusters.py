import numpy as np
import matplotlib.pyplot as plt
import os
from constants import SAMPLE_RATE, NUM_CHANNELS, TIMESTEPS


class Spike(object):
    def __init__(self, data=None):
        self.data = data  # Will contain data from 8 channels each with multiple samples over time (originally 32)

    def is_punit(self):
        """
        The function checks if a spike is of a positive-unit and returns the result
        """
        median = 0  # reference value, it is assumed that the spikes are aligned to zero
        avg_spike = np.mean(self.data, axis=0)  # we look at all channels as one wave, should consider checking each
        # channel separately
        avg_spike = avg_spike[3:-3]  # in some cases the hyperpolarization at the edges was stronger than the
        # depolarization, causing wrong conclusion
        abs_diff = np.absolute(avg_spike - median)
        arg_max = np.argmax(abs_diff, axis=0)  # the axis specification has no effect, just for clarification
        if avg_spike[arg_max] > median:
            return True
        return False

    def fix_punit(self):
        """
        The function flips the spike
        """
        self.data = self.data * -1

    def get_data(self):
        return self.data

    def plot_spike(self):
        for i in range(NUM_CHANNELS):
            # we don't use constants to allow use after upsampling
            plt.plot(np.arange(self.data.shape[1]), self.data[i, :])
        plt.show()


class Cluster(object):
    def __init__(self, label=-1, filename=None, num_within_file=None, shank=None, spikes=None, timings=None):
        self.label = label
        self.filename = filename  # recording session
        self.num_within_file = num_within_file  # cluster ID
        self.shank = shank  # shank number
        self.spikes = spikes  # list of Spike
        self.np_spikes = None  # np array of spikes, used for optimization
        self.timings = timings

    def add_spike(self, spike, time):
        """
        The function receives a spike to append to the cluster (unit)
        """
        if self.spikes is not None:
            self.spikes.append(spike)
        else:
            self.spikes = [spike]

        if self.timings is not None:
            # we divide by the sample rate to get the time in seconds and multiply by 1000 to get the time in ms
            self.timings.append(1000 * time / SAMPLE_RATE)
        else:
            self.timings = [1000 * time / SAMPLE_RATE]

    def get_unique_name(self):
        """
        The function returns a unique name based on the cluster's fields
        """
        return self.filename + "_" + str(self.shank) + "_" + str(self.num_within_file)

    def calc_mean_waveform(self):
        """
        The function calculates the mean waveform (i.e. average spike of the cluster)
        """
        if self.np_spikes is None:  # for faster processing
            self.finalize_spikes()
        return Spike(data=self.np_spikes.mean(axis=0))

    def fix_punits(self):
        """
        The function checks if the cluster/unit is a positive-unit and flips it if so
        This is determined by the mean waveform
        """
        mean_spike = self.calc_mean_waveform()
        if mean_spike.is_punit():
            self.np_spikes = self.np_spikes * -1

    def finalize_spikes(self):
        """
        The function transforms the spike list to a single numpy array, this is used for faster processing later on
        """
        shape = (len(self.spikes), NUM_CHANNELS, TIMESTEPS)
        self.np_spikes = np.empty(shape)
        for i, spike in enumerate(self.spikes):
            self.np_spikes[i] = spike.get_data()

    def save_cluster(self, path):
        if self.np_spikes is None:  # for faster processing
            self.finalize_spikes()
        if not os.path.isdir(path):
            os.mkdir(path)
        np.save(path + self.get_unique_name() + '_' + str(self.label) + '_' + 'spikes', self.spikes)
        np.save(path + self.get_unique_name() + '_' + str(self.label) + '-' + 'timings', np.array(self.timings))

    def load_cluster(self, path):
        path_elements = path.split('\\')[-1].split('_')
        if path_elements[-1] == 'spikes':
            self.spikes = np.load(path)
            self.np_spikes = np.load(path)
            self.filename = path_elements[0]
            self.shank = path_elements[1]
            self.num_within_file = path_elements[2]
            self.label = path_elements[3]
        elif path_elements[-1] == 'timings':
            self.timings = np.load(path)

    def assert_legal(self):
        if self.filename is None or\
                self.num_within_file is None or\
                self.shank is None or\
                self.spikes is None or\
                self.np_spikes is None or\
                self.timings is None:
            return False
        return True
