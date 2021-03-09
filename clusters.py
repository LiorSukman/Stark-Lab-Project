import numpy as np
import matplotlib.pyplot as plt


class Spike(object):
    def __init__(self, data = None):
        self.data = data # Will contain data from 8 channels each with 32 samples

    def is_punit(self):
        """
        The function checks if a spike is of a positive-unit and returns the result
        """
        median = 0 # reference value, it is assumed that the spikes are alligned to zero
        avg_spike = np.mean(self.data, axis = 0) # we look at all channels as one wave, should consider checking each channel separately
        avg_spike = avg_spike[3:-3] # in some cases the hyperpolarization at the edges was stronger than the depolarization, causing wrong conclusion 
        abs_diff = np.absolute(avg_spike - median)
        arg_max = np.argmax(abs_diff, axis = 0) # the axis specification has no effect, just for clarification
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
        for i in range(8):
            plt.plot([j for j in range(32)], self.data[i, :])
        plt.show()

class Cluster(object):
    def __init__(self, label = -1, filename = None, numWithinFile = None, shank = None, spikes = []):
        self.label = label
        self.filename = filename # recording session
        self.numWithinFile = numWithinFile # cluster ID
        self.shank = shank # shank number
        self.spikes = spikes # list of Spike
        self.np_spikes = None # np array of spikes, used for optimization

    def add_spike(self, spike):
        """
        The function receives a spike to append to the cluster (unit)
        """
        self.spikes.append(spike)

    def get_unique_name(self):
        """
        The function returns a unique name based on the cluster's fields
        """
        return self.filename + "_" + str(self.shank) + "_" + str(self.numWithinFile)

    def calc_mean_waveform(self):
        """
        The function calculates the mean waveform (i.e. average spike of the cluster)
        """
        if self.np_spikes is None: # for faster processing
            self.finalize_spikes()
        return Spike(data = self.np_spikes.mean(axis = 0))

    def fix_punits(self):
        """
        The function checks if the cluster/unit is a positive-unit and flips it if so
        This is determined by the mean waveform
        """
        meanSpike = self.calc_mean_waveform()
        if meanSpike.is_punit():
            self.np_spikes = self.np_spikes * -1

    def finalize_spikes(self):
        """
        The function transforms the spike list to a single numpy array, this is used for faster processing later on
        """
        shape = (len(self.spikes), 8, 32)
        self.np_spikes = np.empty(shape)
        for i, spike in enumerate(self.spikes):
            self.np_spikes[i] = spike.get_data()
        
