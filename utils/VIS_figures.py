import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np
import pandas as pd
import scipy.signal as signal
import os
import seaborn as sns
from sklearn.metrics import roc_curve, auc, plot_roc_curve, ConfusionMatrixDisplay


from clusters import Cluster, Spike
from preprocessing_pipeline import load_cluster
from light_removal import remove_light
from constants import PV_COLOR, LIGHT_PV, PYR_COLOR, LIGHT_PYR
from constants import SPATIAL, MORPHOLOGICAL, TEMPORAL
from features.temporal_features_calc import calc_temporal_histogram
from features.spatial_features_calc import sp_wavelet_transform
from utils.upsampling import upsample_spike
from ml.plot_tests import plot_results, plot_fet_imp

SAVE_PATH = '../../../data for figures/'
TEMP_PATH = '../temp_state/'
DATA_PATH = '../clustersData_no_light/0'
pyr_name = name = 'es25nov11_13_3_3'  # pyr
pv_name = 'es25nov11_13_3_11'  # pv

# Load wanted units
pyr_cluster = load_cluster(TEMP_PATH, pyr_name)
pv_cluster = load_cluster(TEMP_PATH, pv_name)

# Waveforms
pyr_cluster.plot_cluster(save=True, path=SAVE_PATH)
pv_cluster.plot_cluster(save=True, path=SAVE_PATH)

def clear():
    plt.clf()
    plt.cla()
    plt.close('all')


# Morphological features
NUM_CHANNELS = 8
TIMESTEPS = 32
UPSAMPLE = 8

clu = pyr_cluster
color = PYR_COLOR


def get_main(chunks):
    chunk_amp = chunks.max(axis=1) - chunks.min(axis=1)
    main_channel = np.argmax(chunk_amp)
    return main_channel


def t2p_fwhm(clu, color, name):
    chunks = clu.calc_mean_waveform()
    if UPSAMPLE != 1:
        chunks = upsample_spike(chunks.data)

    spike = chunks[get_main(chunks)]

    dep_ind = spike.argmin()
    dep = spike[dep_ind]
    hyp_ind = spike.argmax()
    hyp = spike[hyp_ind]
    print(
        f"{name} t2p is {1.6 * (hyp_ind - dep_ind + 1) / len(spike)} ms")  # 1.6 ms is the time of every spike (32 ts in 20khz)

    fwhm_inds = np.argwhere(spike <= dep / 2).flatten()
    print(f"{name} fwhm is {1.6 * (fwhm_inds[-1] - fwhm_inds[0] + 1) / len(spike)} ms")

    fig, ax = plt.subplots()
    ax.plot(spike, c=color)
    ax.plot(fwhm_inds, spike[fwhm_inds], c='k', linewidth=3)
    ax.plot([dep_ind, hyp_ind], [dep, hyp], marker='o', linestyle='None', c=color)

    plt.savefig(SAVE_PATH + f"{name}_t2p_fwhm.pdf", transparent=True)
    clear()


t2p_fwhm(pyr_cluster, PYR_COLOR, 'pyr')
t2p_fwhm(pv_cluster, PV_COLOR, 'pv')


def max_speed(clu, color, name):
    chunks = clu.calc_mean_waveform()
    if UPSAMPLE != 1:
        chunks = upsample_spike(chunks.data)

    spike = chunks[get_main(chunks)]

    spike_der = np.convolve(spike, [1, -1], mode='valid')
    max_speed_inds = np.argwhere(spike_der > spike_der[131]).flatten()
    print(f"{name} max_speed is {1.6 * (max_speed_inds[-1] - max_speed_inds[0] + 1) / len(spike)} ms")

    fig, ax = plt.subplots()
    ax.plot(spike_der, c=color)
    ax.plot(max_speed_inds, spike_der[max_speed_inds], c='k', linewidth=3)

    plt.savefig(SAVE_PATH + f"{name}_max_speed.pdf", transparent=True)
    clear()


max_speed(pyr_cluster, PYR_COLOR, 'pyr')
max_speed(pv_cluster, PV_COLOR, 'pv')


def load_df(features):
    df = None
    files = os.listdir(DATA_PATH)
    for file in sorted(files):
        if df is None:
            df = pd.read_csv(DATA_PATH + '/' + file)
        else:
            temp = pd.read_csv(DATA_PATH + '/' + file)
            df = df.append(temp)

    df = df.loc[df.label >= 0]
    df.label = df.label.map({1: 'PYR', 0: 'IN'})
    df = df[features]

    return df


features = ['Channels contrast', 'break_measure', 'fwhm', 'get_acc',
            'max_speed', 'peak2peak', 'trough2peak', 'rise_coef', 'smile_cry', 'label']
df = load_df(features)


def corr_mat(df, modality):
    correlation_matrix = df.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    plt.yticks(rotation=30)
    cmap = sns.color_palette("vlag", as_cmap=True)
    _ = sns.heatmap(correlation_matrix, annot=True, fmt='.2f', mask=mask, vmin=-1, vmax=1, cmap=cmap)
    plt.savefig(SAVE_PATH + f"{modality}_cor_mat.pdf", transparent=True)
    clear()


corr_mat(df, 'morph')


def density_plots(df, d_features, modality):
    palette = {"PYR": PYR_COLOR, "IN": PV_COLOR}

    for c in df.columns:
        if c not in d_features:
            continue
        # bw_adjust = (df[c].max() - df[c].min()) / 50
        sns.displot(data=df, x=c, hue="label", common_norm=False, kind="kde", fill=True,
                    palette=palette)
        plt.savefig(SAVE_PATH + f"{modality}_density_{c}.pdf", transparent=True)
        clear()


d_features = ['fwhm', 'trough2peak', 'max_speed']
density_plots(df, d_features, 'morph')

def get_results(modality, chunk_size=0, res_name='results_rf'):
    results = pd.read_csv(f'../ml/{res_name}.csv', index_col=0)
    complete = results[results.restriction == 'complete']
    complete = complete[complete.modality == modality]

    grouped_complete = complete.groupby(by=['restriction', 'modality', 'chunk_size'])

    plot_results(grouped_complete.mean(), grouped_complete.sem(), 'complete', acc=True, chunk_size=chunk_size, name=modality)
    clear()

    plot_results(grouped_complete.mean(), grouped_complete.sem(), 'complete', acc=False, chunk_size=chunk_size, name=modality)
    clear()
    d = {'spatial': SPATIAL, 'morphological': MORPHOLOGICAL, 'temporal': TEMPORAL}
    modalities = [(modality, d[modality])]
    plot_fet_imp(grouped_complete.mean(), grouped_complete.sem(), 'complete', chunk_size=chunk_size, name=modality, modalities=modalities)
    clear()

    fig, ax = plt.subplots()
    display_labels = ['PYR', 'PV']
    means = grouped_complete.mean()
    tp = means.xs(chunk_size, level="chunk_size").pyr_acc[0]
    tn = means.xs(chunk_size, level="chunk_size").in_acc[0]
    fp = 100 - tn
    fn = 100 - tp
    cm = np.array([[tp, fn], [fp, tn]])

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    disp.plot(
        include_values=True,
        ax=ax,
        xticks_rotation='horizontal',
        colorbar=True,
        cmap=plt.cm.RdYlGn,
    )
    plt.savefig(SAVE_PATH + f"{modality}_conf_mat.pdf", transparent=True)
    clear()


get_results('morphological', chunk_size=0)

# Spatial features
def spd(clu, name):
    color_main = PYR_COLOR if name == 'pyr' else PV_COLOR
    color_second = LIGHT_PYR if name == 'pyr' else LIGHT_PV
    chunks = clu.calc_mean_waveform()
    if UPSAMPLE != 1:
        chunks = Spike(upsample_spike(chunks.data, UPSAMPLE, 20_000))

    chunks = chunks.get_data()

    main_chn = get_main(chunks)
    dep = chunks[main_chn].min()
    fig, ax = plt.subplots()
    for i in range(NUM_CHANNELS):
        mean_channel = chunks[i]
        if i == main_chn:
            ax.plot(np.arange(TIMESTEPS * UPSAMPLE), mean_channel, c=color_main)
        else:
            c = color_second if mean_channel.min() <= 0.5 * dep else 'k'
            ax.plot(np.arange(TIMESTEPS * UPSAMPLE), mean_channel, c=c)
    ax.hlines(dep * 0.5, 0, TIMESTEPS * UPSAMPLE - 1, colors='k', linestyles='dashed')

    print(f"{name} spatial dispersion count is {np.sum(chunks.min(axis=1) <= chunks.min() * 0.5)}")

    plt.savefig(SAVE_PATH + f"{name}_spatial_dispersion.pdf", transparent=True)
    clear()


spd(pyr_cluster, 'pyr')
spd(pv_cluster, 'pv')

def da(clu, name):
    color_main = PYR_COLOR if name == 'pyr' else PV_COLOR
    color_second = LIGHT_PYR if name == 'pyr' else LIGHT_PV
    chunks = clu.calc_mean_waveform()
    if UPSAMPLE != 1:
        chunks = Spike(upsample_spike(chunks.data, UPSAMPLE, 20_000))

    chunks = chunks.get_data()
    main_chn = get_main(chunks)
    #fig, ax = plt.subplots()
    median = np.median(chunks)

    direction = chunks >= median
    counter = np.sum(direction, axis=0)

    # Iterating over the channels and calculating a direction agreeableness value
    for ind in range(counter.shape[0]):
        temp = counter[ind]
        counter[ind] = temp if temp <= NUM_CHANNELS // 2 else NUM_CHANNELS - temp

    cmap = sns.color_palette("gray", as_cmap=True)
    ax = sns.heatmap(np.array([counter]).repeat(10, axis=0), cmap=cmap, cbar=False)
    ax.axis('off')
    plt.savefig(SAVE_PATH + f"{name}_da_bg.pdf", transparent=True)

    min = chunks.min()
    max = chunks.max()
    chunks = (-chunks / (max - min)) * 6 + 4
    median = np.median(chunks)
    for i in range(NUM_CHANNELS):
        mean_channel = chunks[i]
        c = color_main if i == main_chn else color_second
        ax.plot(np.arange(TIMESTEPS * UPSAMPLE), mean_channel, c=c)
    ax.hlines(median, 0, TIMESTEPS * UPSAMPLE - 1, colors='k', linestyles='dashed')
    print(f"{name} direction agrreablensess is {np.sum(counter ** 2)}")
    plt.savefig(SAVE_PATH + f"{name}_da.pdf", transparent=True)
    clear()

features = ['dep_red', 'dep_sd', 'graph_avg_speed', 'graph_slowest_path', 'graph_fastest_path',
            'geometrical_avg_shift', 'geometrical_shift_sd', 'geometrical_max_dist', 'spatial_dispersion_count',
            'spatial_dispersion_sd', 'da', 'da_sd', 'Channels contrast', 'label']
df = load_df(features)

corr_mat(df, 'spatial')
d_features = ['spatial_dispersion_count', 'da']

density_plots(df, d_features, 'spatial')

da(pyr_cluster, 'pyr')
da(pv_cluster, 'pv')

get_results('spatial', chunk_size=0)


"""# Temporal features

# Spike train
# _, _, _, stims = remove_light(pyr_cluster, True, data_path='../Data/')
# check indicated that it is way after the first 50 spikes so it is ignored in plotting

# Spike Train
pyr_mask = (pyr_cluster.timings > 3509050) * (pyr_cluster.timings < 3510050)
pv_mask = (pv_cluster.timings > 3509050) * (pv_cluster.timings < 3510050)
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
plt.savefig(SAVE_PATH + "spike_train.pdf", transparent=True)

# ACHs
def ach(bin_range, name, clu):
    c = PV_COLOR if name == 'pv' else PYR_COLOR
    N = 2 * bin_range + 2
    offset = 1 / 2
    bins = np.linspace(-bin_range - offset, bin_range + offset, N)
    chunks = np.array([np.arange(len(clu.timings))])
    hist = calc_temporal_histogram(clu.timings, bins, chunks)[0]
    hist_up = signal.resample_poly(hist, UPSAMPLE ** 2, UPSAMPLE, padtype='line')
    hist_up = np.where(hist_up >= 0, hist_up, 0)

    bin_inds = np.linspace(bins[0] + offset, bins[-1] - offset, UPSAMPLE * (N - 1))

    fig, ax = plt.subplots()
    ax.bar(bin_inds, hist_up, color=c)
    plt.savefig(SAVE_PATH + f"{name}_ACH_{bin_range}.pdf", transparent=True)
    clear()

    return hist


hist_pv = ach(50, 'pv', pv_cluster)
hist_pyr = ach(50, 'pyr', pyr_cluster)
ach(1000, 'pv', pv_cluster)
ach(1000, 'pyr', pyr_cluster)

# unif_dist
def unif_dist(hist, name):
    c = PV_COLOR if name == 'pv' else PYR_COLOR
    zero_bin_ind = len(hist) // 2
    hist = (hist[:zero_bin_ind + 1:][::-1] + hist[zero_bin_ind:]) / 2
    hist_up = signal.resample_poly(hist, UPSAMPLE ** 2, UPSAMPLE, padtype='line')
    hist_up = np.where(hist_up >= 0, hist_up, 0)
    cdf = np.cumsum(hist_up) / np.sum(hist_up)

    fig, ax = plt.subplots()
    ax.bar(np.arange(len(cdf)), cdf, color=c)
    lin = np.linspace(cdf[0], cdf[-1], len(cdf))
    dists = abs(lin - cdf)
    ax.plot(lin, c='k')
    ax.vlines(np.arange(len(cdf))[::UPSAMPLE], np.minimum(lin, cdf)[::UPSAMPLE], np.maximum(lin, cdf)[::UPSAMPLE],
              colors='k', linestyles='dashed')
    print(f"{name} unif_dist is {np.sum(dists)}")
    plt.savefig(SAVE_PATH + f"{name}_unif_dist.pdf", transparent=True)
    clear()


features = ['d_kl', 'jump', 'psd_center', 'der_psd_center', 'rise_time', 'unif_dist', 'label']
df = load_df(features)

corr_mat(df, 'temporal')
d_features = ['unif_dist']

density_plots(df, d_features, 'temporal')

unif_dist(hist_pv, 'pv')
unif_dist(hist_pyr, 'pyr')

get_results('temporal', chunk_size=0)"""


# modified waveforms

def plot_delta(clu, name):
    color_main = PYR_COLOR if name == 'pyr' else PV_COLOR
    color_second = LIGHT_PYR if name == 'pyr' else LIGHT_PV
    chunks = clu.calc_mean_waveform()
    if UPSAMPLE != 1:
        chunks = Spike(upsample_spike(chunks.data, UPSAMPLE, 20_000))
    chunks = chunks.get_data()
    chunks = chunks/ (-chunks.min())
    delta = np.zeros(chunks.shape)
    delta[np.arange(NUM_CHANNELS), chunks.argmin(axis=1)] = chunks.min(axis=1)

    fig, ax = plt.subplots(NUM_CHANNELS, 1, sharex=True, sharey=True, figsize=(6, 20))
    for i, c_ax in enumerate(ax[::-1]):
        c_ax.plot(chunks[i], c=color_second)
        c_ax.plot(delta[i], c=color_main)
        c_ax.axis('off')

    plt.savefig(SAVE_PATH + f"{name}_delta.pdf", transparent=True)


plot_delta(pyr_cluster, 'pyr')
plot_delta(pv_cluster, 'pv')


def plot_perm(clu, name):
    color_main = PYR_COLOR if name == 'pyr' else PV_COLOR
    chunks = clu.calc_mean_waveform()
    if UPSAMPLE != 1:
        chunks = Spike(upsample_spike(chunks.data, UPSAMPLE, 20_000))
    chunks = chunks.get_data()
    inds = np.arange(chunks.shape[-1])
    np.random.shuffle(inds)

    fig, ax = plt.subplots(NUM_CHANNELS, 1, sharex=True, sharey=True, figsize=(6, 20))
    for i, c_ax in enumerate(ax[::-1]):
        c_ax.plot(chunks[i][inds], c=color_main)
        c_ax.axis('off')

    plt.savefig(SAVE_PATH + f"{name}_perm.pdf", transparent=True)


plot_perm(pyr_cluster, 'pyr')
plot_perm(pv_cluster, 'pv')

def get_part_results(modality, chunk_size=0, res_name='results_rf'):
    results = pd.read_csv(f'../ml/{res_name}.csv', index_col=0)
    complete = results[results.restriction == 'complete']
    complete = complete[complete.modality == modality]

    grouped_complete = complete.groupby(by=['restriction', 'modality', 'chunk_size'])

    plot_results(grouped_complete.mean(), grouped_complete.sem(), 'complete', acc=True, chunk_size=chunk_size, name=modality)
    clear()

    plot_results(grouped_complete.mean(), grouped_complete.sem(), 'complete', acc=False, chunk_size=chunk_size, name=modality)
    clear()


get_part_results('perm_morph', res_name='results_rf_perm_morph_no_fwhm')



