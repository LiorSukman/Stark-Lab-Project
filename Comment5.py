import numpy as np
import matplotlib.pyplot as plt
from utils.upsampling import upsample_spike
from preprocessing_pipeline import load_cluster

TEMP_PATH_NO_LIGHT = './temp_state_minus_light/'
pyr_name = 'es25nov11_13_3_3'  # pyr

def com5(num_spikes=200):
    # print(np.argmax(mean_clu.max(axis=1) - mean_clu.min(axis=1)))
    clu = load_cluster(TEMP_PATH_NO_LIGHT, pyr_name).np_spikes[:num_spikes]

    mean_clu = clu.mean(axis=0)
    mean_up_clu = upsample_spike(mean_clu)[2]

    up_clu = np.array([upsample_spike(spike)[2] for spike in clu])
    up_mean_clu = up_clu.mean(axis=0)

    plt.plot(mean_up_clu)
    plt.plot(up_mean_clu)
    plt.show()

def com8():
    

if __name__ == '__main__':

