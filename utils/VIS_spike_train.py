import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np
from os import listdir
from os.path import isfile, join
import scipy.signal as signal

from clusters import Cluster, Spike
from preprocessing_pipeline import load_cluster
from light_removal import remove_light
from constants import PV_COLOR, LIGHT_PV, PYR_COLOR, LIGHT_PYR
from features.temporal_features_calc import calc_temporal_histogram
from features.spatial_features_calc import sp_wavelet_transform

TEMP_PATH = '..\\temp_state\\'
pyr_name = name = 'es25nov11_13_1_6'  # pyr
pv_name = 'es25nov11_13_3_11'  # pv
# name = 'es25nov11_13_1_11'  # pyr
# name = 'es25nov11_13_4_5'  # pv
# name = 'es25nov11_13_3_6'  # pyr

cluster = pyr_cluster = load_cluster(TEMP_PATH, pyr_name)
pv_cluster = load_cluster(TEMP_PATH, pv_name)

# Spike train
spike_train = cluster.timings
# _, _, _, stims = remove_light(cluster, True, data_path='../Data/')
# check indicated that it is way after the first 50 spikes so it is ignored in plotting

"""pyr_mask = (pyr_cluster.timings > 60000) * (pyr_cluster.timings < 63000)
pv_mask = (pv_cluster.timings > 60000) * (pv_cluster.timings < 63000)
pyr_train = pyr_cluster.timings[pyr_mask]
pv_train = pv_cluster.timings[pv_mask]

fig, ax = plt.subplots()
ax.axis('off')
ax.vlines(pyr_train, 0.05, 0.45, colors=PYR_COLOR)
ax.vlines(pv_train, 0.55, 0.95, colors=PV_COLOR)
scalebar = AnchoredSizeBar(ax.transData,
                           200, '200 ms', 'lower right',
                           pad=0.1,
                           color='k',
                           frameon=False,
                           size_vertical=0.01)

ax.add_artist(scalebar)
plt.show()"""

"""# Temporal features
bin_range = 50
N = 2 * bin_range + 2
offset = 1 / 2
bins = np.linspace(-bin_range - offset, bin_range + offset, N)
chunks = np.array([np.arange(len(spike_train))])
hist = calc_temporal_histogram(spike_train, bins, chunks)[0]
bin_inds = []
for i in range(len(bins) - 1):
    bin_inds.append((bins[i] + bins[i + 1]) / 2)
plt.bar(bin_inds, hist)
plt.show()

zero_bin_ind = len(hist) // 2
hist = (hist[:zero_bin_ind + 1:][::-1] + hist[zero_bin_ind:]) / 2
cdf = np.cumsum(hist) / np.sum(hist)
plt.bar(np.arange(len(cdf)), cdf)
lin = np.linspace(cdf[0], cdf[-1], len(cdf))
plt.plot(lin, c='k')
plt.vlines(np.arange(len(cdf)), np.minimum(lin, cdf), np.maximum(lin, cdf), colors='k', linestyles='dashed')
plt.show()"""

# Spatial features
NUM_CHANNELS = 8
TIMESTEPS = 32
UPSAMPLE = 1
chunks = pv_cluster.calc_mean_waveform()
if UPSAMPLE != 1:
    chunks = Spike(signal.resample_poly(chunks.data, UPSAMPLE ** 2, UPSAMPLE, axis=1, padtype='line'))

# constrained transformation
"""cons_wavelets = sp_wavelet_transform([chunks], True)[0].get_data()
fig, ax = plt.subplots(NUM_CHANNELS, 1, sharex=True, sharey=True)
for i, c_ax in enumerate(ax[::-1]):
    mean_channel = cons_wavelets[i]
    c_ax.plot(np.arange(TIMESTEPS), mean_channel)
    c_ax.plot(np.arange(TIMESTEPS), chunks.get_data()[i], c='k')
    fig.suptitle(f"Cluster {cluster.get_unique_name()} of type {'PYR' if cluster.label == 1 else 'IN' if cluster.label==0 else 'UT' }")
plt.show()"""

"""chunks = Spike(signal.resample_poly(chunks.data, UPSAMPLE ** 2, UPSAMPLE, axis=1, padtype='line'))
cons_wavelets = sp_wavelet_transform([chunks], True)[0].get_data()
fig, ax = plt.subplots(NUM_CHANNELS, 1, sharex=True, sharey=True)
for i, c_ax in enumerate(ax[::-1]):
    mean_channel = cons_wavelets[i]
    c_ax.plot(np.arange(TIMESTEPS * UPSAMPLE), mean_channel)
    c_ax.plot(np.arange(TIMESTEPS * UPSAMPLE), chunks.get_data()[i], c='k')
    fig.suptitle(f"Cluster {cluster.get_unique_name()} of type {'PYR' if cluster.label == 1 else 'IN' if cluster.label==0 else 'UT' }")
plt.show()"""

main_chn = np.argmin(chunks.get_data()) // (TIMESTEPS * UPSAMPLE)
dep = chunks.get_data()[main_chn].min()
fig, ax = plt.subplots(NUM_CHANNELS, 1, sharex=True, sharey=True, figsize=(6, 20))
for i, c_ax in enumerate(ax[::-1]):
    mean_channel = chunks.get_data()[i]
    if i == main_chn:
        c_ax.plot(np.arange(TIMESTEPS * UPSAMPLE), mean_channel, c=PYR_COLOR)
        c_ax.vlines(np.argmin(mean_channel), np.min(mean_channel), np.max(mean_channel), colors='k', linestyles='dashed')
    elif np.min(mean_channel) > 0.25 * dep:
        c_ax.plot(np.arange(TIMESTEPS * UPSAMPLE), mean_channel, c='k')
    else:
        c_ax.plot(np.arange(TIMESTEPS * UPSAMPLE), mean_channel, c=LIGHT_PYR)
        c_ax.vlines(np.argmin(mean_channel), np.min(mean_channel), np.max(mean_channel), colors=PYR_COLOR, linestyles='dashed')
        c_ax.vlines(np.argmin(chunks.get_data()[main_chn]), np.min(mean_channel), np.max(mean_channel), colors='k', linestyles='dashed')
    #fig.suptitle(f"Cluster {cluster.get_unique_name()} of type {'PYR' if cluster.label == 1 else 'IN' if cluster.label==0 else 'UT' }")
    c_ax.axis('off')
plt.show()

main_chn = np.argmin(chunks.get_data()) // (TIMESTEPS * UPSAMPLE)
dep = chunks.get_data()[main_chn].min()
fig, ax = plt.subplots(NUM_CHANNELS, 1, sharex=True, sharey=True)
for i, c_ax in enumerate(ax[::-1]):
    mean_channel = chunks.get_data()[i]
    if i == main_chn:
        c_ax.plot(np.arange(TIMESTEPS * UPSAMPLE), mean_channel, c='r')
    else:
        c = 'k' if mean_channel.min() > 0.5 * dep else 'b'
        c_ax.plot(np.arange(TIMESTEPS * UPSAMPLE), mean_channel, c=c)
    c_ax.hlines(dep * 0.5, 0, TIMESTEPS * UPSAMPLE - 1, colors='k', linestyles='dashed')
    fig.suptitle(f"Cluster {cluster.get_unique_name()} of type {'PYR' if cluster.label == 1 else 'IN' if cluster.label==0 else 'UT' }")
plt.show()

"""fig, ax = plt.subplots()
main_chn = np.argmin(chunks.get_data()) // (TIMESTEPS * UPSAMPLE)
dep = chunks.get_data()[main_chn].min()
for i, ch in enumerate(chunks.get_data()):
    mean_channel = ch
    if i == main_chn:
        ax.plot(np.arange(TIMESTEPS * UPSAMPLE), mean_channel, c=PV_COLOR)
    else:
        c = 'k' if mean_channel.min() > 0.5 * dep else LIGHT_PV
        ax.plot(np.arange(TIMESTEPS * UPSAMPLE), mean_channel, c=c)
ax.hlines(dep * 0.5, 0, TIMESTEPS * UPSAMPLE - 1, colors='k', linestyles='dashed')
# plt.title(f"Cluster {cluster.get_unique_name()} of type {'PYR' if cluster.label == 1 else 'IN' if cluster.label==0 else 'UT' }")
ax.axis('off')
plt.show()"""


