import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
from preprocessing_pipeline import load_cluster

temp_path = '..\\temp_state\\'
pyr_name = 'es25nov11_13_1_6'
pv_name = 'es25nov11_13_3_11'

pyr_clu = load_cluster(temp_path, pyr_name)
pv_clu = load_cluster(temp_path, pv_name)

pyr_timings = pyr_clu.timings
pv_timings = pv_clu.timings

timings = np.concatenate((pyr_timings, pv_timings))
classes = np.concatenate((np.ones(len(pyr_timings)), np.ones(len(pv_timings)) * 2))

sort_inds = np.argsort(timings)
timings = timings[sort_inds]
classes = classes[sort_inds]

mdic = {"timings": timings, "classes": classes}
# io.savemat("cc_mat_new.mat", mdic)


def calc_hist(spike_train, stims, bins):
    spike_train = spike_train * 20_000 / 1000
    ret = np.zeros(len(bins) - 1)
    for stim in stims:
        ref_time_list = spike_train - stim
        # print(bins[0], bins[-1])
        mask = (ref_time_list >= bins[0]) * (ref_time_list < bins[-1])
        #  print(mask.sum())
        ref_time_list = ref_time_list[mask]
        hist, _ = np.histogram(ref_time_list, bins=bins)
        ret += hist

    return 20_000 * ret / len(stims)


N = 2 * 1 * 50 + 2
offset = 1 / (2 * 1)
bins = np.linspace(-50 - offset, 50 + offset, N)

hist = calc_hist(pyr_timings, pv_timings, bins)
plt.bar(np.linspace(-50, 50, N - 1), hist)
plt.show()
