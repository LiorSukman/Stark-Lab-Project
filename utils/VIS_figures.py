import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats
import scipy.io as io
from sklearn.preprocessing import StandardScaler
import os
from os import listdir
from os.path import isfile, join
import seaborn as sns
import math

from clusters import Spike
from preprocessing_pipeline import load_cluster
from constants import PV_COLOR, LIGHT_PV, PYR_COLOR, LIGHT_PYR
from constants import SPATIAL, MORPHOLOGICAL, TEMPORAL, TRANS_MORPH, feature_names_org
from constants import COORDINATES
from features.temporal_features_calc import calc_temporal_histogram
from features.spatial_features_calc import DELTA_MODE
from features.spatial_features_calc import calc_pos as calc_pos_imp
from utils.upsampling import upsample_spike
from ml.plot_tests import plot_fet_imp, plot_conf_mats, plot_cf_thr, plot_roc_curve, \
    plot_test_vs_dev_bp, plot_auc_chunks_bp
# Note that plot_conf_mats requires path for the correct dataset

import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

SAVE_PATH = '../../../data for figures/Thesis/'  # make sure to change in plot_tests as well
TEMP_PATH_NO_LIGHT = '../temp_state_minus_light/'
DATA_PATH = '../cluster_data/clusterData_no_light_29_03_22/0'
SPATIAL_DATA_PATH = '../cluster_data/clusterData_no_light_29_03_22/25'
pyr_name = 'es25nov11_13_3_3'  # pyr
pv_name = 'es25nov11_13_3_11'  # pv

NUM_CHANNELS = 8
TIMESTEPS = 32
UPSAMPLE = 8
NUM_MOMENTS = 5
NUM_EVENTS = 3

BEST_WF_CHUNK = 50
BEST_SPATIAL_CHUNK = 25
BEST_TEMPORAL_CHUNK = 1600

# the following values are required to transform the spikes from arbitrary units to Voltage
# this values are based on the corresponding XML file, change them accordingly if changing units
VOL_RANGE = 8
AMPLIFICATION = 1000
NBITS = 16

feature_mapping = {feature_names_org[i]: i for i in range(len(feature_names_org))}


def clear():
    plt.clf()
    plt.cla()
    plt.close('all')


def pie_chart(df):
    labels = ['PYR CA1', 'PYR nCX', 'PV nCX', 'PV CA1']
    colors = [PYR_COLOR, LIGHT_PYR, LIGHT_PV, PV_COLOR]

    df.region = df.region * 2
    df['Label X Region'] = df.label + df.region

    x = df['Label X Region'].to_numpy()

    sizes = [np.count_nonzero(x == 3), np.count_nonzero(x == 1), np.count_nonzero(x == 0), np.count_nonzero(x == 2)]
    total = sum(sizes)

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct=lambda p: '{:.0f}'.format(p * total / 100), colors=colors, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.savefig(SAVE_PATH + f"pie.pdf", transparent=True)
    clear()


def trans_units(cluster):
    data = cluster.spikes
    data = (data * VOL_RANGE * 1e6) / (AMPLIFICATION * (2 ** NBITS))  # 1e6 for micro V
    cluster.spikes = data
    cluster.np_spikes = data


def get_main(chunks):
    chunk_amp = chunks.max(axis=1) - chunks.min(axis=1)
    main_channel = np.argmax(chunk_amp)
    return main_channel


def t2p_fwhm(clu, color, sec_color, name):
    chunks = clu.calc_mean_waveform()
    if UPSAMPLE != 1:
        chunks = upsample_spike(chunks.data)

    spike = chunks[get_main(chunks)]
    spike = spike / abs(spike.min())

    dep_ind = spike.argmin()
    dep = spike[dep_ind]
    hyp_ind = spike.argmax()
    hyp = spike[hyp_ind]
    t2p = 1.6 * (hyp_ind - dep_ind) / len(spike)
    print(f"{name} t2p is {t2p} ms")  # 1.6 ms is the time of every spike (32 ts in 20khz)

    fwhm_inds = spike <= (dep / 2)
    fwhm = 1.6 * (fwhm_inds.sum()) / len(spike)
    print(f"{name} fwhm is {fwhm} ms")

    fig, ax = plt.subplots()
    ax.plot(spike, c=color)
    rect = patches.Rectangle((np.argwhere(fwhm_inds).flatten()[0] - 5, dep / 2), fwhm_inds.sum() + 10, dep / 2,
                             facecolor=sec_color, alpha=0.2)
    ax.add_patch(rect)
    ax.plot([dep_ind, hyp_ind], [dep, hyp], marker='o', linestyle='None', c=color)

    plt.savefig(SAVE_PATH + f"{name}_t2p_fwhm.pdf", transparent=True)
    clear()

    return t2p, fwhm


def acceleration(clu, color, sec_color, name):
    chunks = clu.calc_mean_waveform()
    if UPSAMPLE != 1:
        chunks = upsample_spike(chunks.data)

    spike = chunks[get_main(chunks)]
    spike = spike / abs(spike.min())

    spike_der = np.convolve(np.convolve(spike, [1, -1], mode='valid'), [1, -1], mode='valid')
    dep_ind = np.argmin(spike)
    roi = spike_der[dep_ind + 14: dep_ind + 40]

    ret = np.sum(roi ** 2)

    print(f"{name} acceleration is {ret}")

    fig, ax = plt.subplots()
    ax.plot(spike_der, c=color)
    rect = patches.Rectangle((dep_ind + 14, roi.max()), 26,
                             roi.min() - roi.max(), facecolor=sec_color, alpha=0.2)
    ax.add_patch(rect)
    plt.savefig(SAVE_PATH + f"{name}_acceleration.pdf", transparent=True)
    clear()

    return ret


def load_df(features, trans_labels=True):
    df = None
    files = os.listdir(DATA_PATH)
    for file in sorted(files):
        if df is None:
            df = pd.read_csv(DATA_PATH + '/' + file)
        else:
            temp = pd.read_csv(DATA_PATH + '/' + file)
            df = df.append(temp)

    df = df.loc[df.label >= 0]
    df.region = df.region.map({0: 0, 1: 1, 3: 1, 4: 1})
    if trans_labels:
        df.label = df.label.map({1: 'PYR', 0: 'PV'})
    df = df[features]

    return df


def corr_mat(df, modality, order):
    pvals = io.loadmat('../statistics/spearman.mat')['pvals']
    inds = [feature_mapping[fet_name] for fet_name in order]
    pvals = pvals[inds][:, inds]
    if modality == 'temporal':
        print("correlation significance of dkl_start and unif_dist", pvals[0, 1])
    annotations = np.where(pvals < 0.05, '*', '')
    annotations = np.where(pvals < 0.01, '**', annotations)
    annotations = np.where(pvals < 0.001, '***', annotations)

    correlation_matrix = df[order].corr(method="spearman")
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    plt.yticks(rotation=30)
    cmap = sns.color_palette("vlag", as_cmap=True)
    _ = sns.heatmap(correlation_matrix, annot=annotations, mask=mask, fmt='s', vmin=-1, vmax=1, cmap=cmap,
                    annot_kws={"fontsize": 6})
    plt.savefig(SAVE_PATH + f"{modality}_cor_mat.pdf", transparent=True)
    clear()


def density_plots(df, d_features, modality, values):
    palette = {"PYR": PYR_COLOR, "PV": PV_COLOR}
    for c in df.columns:
        if c not in d_features:
            continue
        if c in ['fwhm', 'trough2peak']:
            df[c] = 1.6 * df[c] / 256  # transform index to time
        if c in ['fzc_sd']:
            df[c] = 1600 * df[c] / 256
        if c in ['fzc_red']:
            df[c] = df[c] * ((1600 / 256) ** 2)  # transform index to time
        if c in ['fzc_graph_avg_speed']:
            df[c] = 256 * df[c] / 1.6  # transform index to time

    for c, vals in zip(d_features, values):
        col_pyr = df[c][df.label == 'PYR'].to_numpy()
        col_pv = df[c][df.label == 'PV'].to_numpy()

        pyr_med = np.median(col_pyr)
        pv_med = np.median(col_pv)

        statistic, p_val = stats.kstest(col_pyr, col_pv)
        print(f"KS statistical test results for feature {c} are p-value={p_val} (statistic={statistic})")

        ax_cdf = sns.ecdfplot(data=df, x=c, hue="label", palette=palette)

        ax_cdf.scatter(vals[0], (np.sum(col_pyr < vals[0]) + np.sum(col_pyr <= vals[0])) / (2 * len(col_pyr)),
                       color=PYR_COLOR, s=20)
        ax_cdf.scatter(vals[1], (np.sum(col_pv < vals[1]) + np.sum(col_pv <= vals[1])) / (2 * len(col_pv)),
                       color=PV_COLOR,
                       s=20)

        if c in ['d_kl_mid', 'get_acc']:
            ax_cdf.set_xscale('log')

        ax_cdf.axhline(0.5, c='k', alpha=0.3, lw=0.5)
        ax_cdf.axvline(pyr_med, c=PYR_COLOR, lw=2, ls='--')
        ax_cdf.axvline(pv_med, c=PV_COLOR, lw=2, ls='--')

        ax_cdf.set_ylim(ymin=0, ymax=1.1)
        ax_cdf.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax_cdf.set_yticklabels(['0', '', '0.5', '', '1'])
        ax_cdf.spines['top'].set_visible(False)
        ax_cdf.spines['right'].set_visible(False)
        plt.savefig(SAVE_PATH + f"{modality}_cdf_{c}.pdf", transparent=True)
        clear()


def thr_confusion_mats(preds_path='../ml/predictions/preds_rf_290322_fix_imp.npy'):
    preds = np.load(preds_path)
    NUM_CHUNKS = 8
    NUM_MODALITIES = 3
    MOD_IND = 2
    CHUNK_IND = 0
    assert (CHUNK_IND == 0)
    preds_0 = preds[CHUNK_IND::NUM_CHUNKS]
    preds_wf = preds_0[MOD_IND::NUM_MODALITIES]

    NUM_PYR = 83
    NUM_PV = 21
    labels = np.concatenate((np.ones(NUM_PYR), np.zeros(NUM_PV)))

    fig, c_ax = plt.subplots()
    thr = 0.1
    plot_cf_thr(preds_wf, thr, labels, c_ax)
    plt.savefig(SAVE_PATH + f"thr{thr}_WF_conf_mat.pdf", transparent=True)
    clear()

    fig, c_ax = plt.subplots()
    thr = 0.9
    plot_cf_thr(preds_wf, thr, labels, c_ax)
    plt.savefig(SAVE_PATH + f"thr{thr}_WF_conf_mat.pdf", transparent=True)
    clear()

    fig, c_ax = plt.subplots()
    thr = 0.45
    plot_cf_thr(preds_wf, thr, labels, c_ax)
    plt.savefig(SAVE_PATH + f"thr{thr}_WF_conf_mat.pdf", transparent=True)
    clear()


def get_results(modality, chunk_size, res_name='results_rf_combined', base_name='results_rf_combined_chance_balanced'):
    results = pd.read_csv(f'../ml/{res_name}.csv', index_col=0)
    complete = results[results.restriction == 'complete']
    complete = complete[complete.modality == modality]

    complete = complete[complete.chunk_size.isin([chunk_size])]
    complete = complete.dropna(how='all', axis=1)

    grouped_complete = complete.groupby(by=['restriction', 'modality', 'chunk_size'])

    base = pd.read_csv(f'../ml/{base_name}.csv', index_col=0)
    base = base[base.restriction == 'complete']
    base = base[base.modality == modality]

    base = base[base.chunk_size.isin([chunk_size])]
    base = base.dropna(how='all', axis=1)

    grouped_base = base.groupby(by=['restriction', 'modality', 'chunk_size'])

    d = {'spatial': SPATIAL, 'morphological': MORPHOLOGICAL, 'temporal': TEMPORAL}
    modalities = [(modality, d[modality])]

    plot_fet_imp(grouped_complete.median(), grouped_complete.quantile(0.75), 'complete', grouped_base.median(),
                 chunk_size=[chunk_size], name=modality, modalities=modalities, semsn=grouped_complete.quantile(0.25))
    clear()

    chunk_size_list = [chunk_size] if chunk_size == 0 else [0, chunk_size]

    complete_roc = results[results.restriction == 'complete']
    complete_roc = complete_roc[complete_roc.modality == modality]

    complete_roc = complete_roc[complete_roc.chunk_size.isin(chunk_size_list)]
    complete_roc = complete_roc.dropna(how='all', axis=1)

    plot_roc_curve(complete_roc, name=modality, chunk_size=chunk_size_list, modalities=modalities)
    clear()


def spd(clu_spd, name_spd):
    color_main = PYR_COLOR if name_spd == 'pyr' else PV_COLOR
    color_second = LIGHT_PYR if name_spd == 'pyr' else LIGHT_PV
    chunks = clu_spd.calc_mean_waveform()
    if UPSAMPLE != 1:
        chunks = Spike(upsample_spike(chunks.data, UPSAMPLE, 20_000))

    chunks = chunks.get_data()
    chunks = chunks / abs(chunks.min())

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

    spd_val = np.sum(chunks.min(axis=1) <= chunks.min() * 0.5)
    print(f"{name_spd} spatial dispersion count is {spd_val}")
    chunk_norm = chunks.min(axis=1) / chunks.min()
    chunk_norm.sort()
    print(f"{name_spd} spatial dispersion sd is {np.std(chunk_norm)}, vector {chunk_norm}")

    plt.savefig(SAVE_PATH + f"{name_spd}_spatial_dispersion.pdf", transparent=True)
    clear()

    return spd_val


def fzc_time_lag(clu_fzc_tl, name_fzc_tl):
    color_main = PYR_COLOR if name_fzc_tl == 'pyr' else PV_COLOR
    color_second = LIGHT_PYR if name_fzc_tl == 'pyr' else LIGHT_PV

    chunks = clu_fzc_tl.calc_mean_waveform()
    if UPSAMPLE != 1:
        chunks = Spike(upsample_spike(chunks.data, UPSAMPLE, 20_000))

    chunks = chunks.get_data()

    main_c = get_main(chunks)
    amps = chunks.max(axis=1) - chunks.min(axis=1)

    chunks = chunks / (-chunks.min())
    delta = np.zeros(chunks.shape)
    inds = []
    med = np.median(chunks)
    for i in range(len(chunks)):
        sig_m = np.convolve(np.where(chunks[i] <= med, -1, 1), [-1, 1], 'same')
        sig_m[0] = sig_m[-1] = 1
        ind = calc_pos_imp(sig_m, chunks[i].argmin(), DELTA_MODE.F_ZCROSS)
        inds.append(ind)
    delta[np.arange(NUM_CHANNELS), inds] = chunks.min(axis=1)
    shift = chunks.shape[-1] // 2 - delta[main_c].argmin()
    cent_delta = np.roll(delta, shift, axis=1)

    dep_inds = delta.argmin(axis=1)
    cent_dep_inds = cent_delta.argmin(axis=1)
    fzc_ss = (((cent_dep_inds - 128) * (1600 / 256)) ** 2)[amps >= 0.25 * amps.max()].mean()
    print(f"{name_fzc_tl} FZC time lag SS value is {fzc_ss}")
    fzc_sd = np.std(((cent_dep_inds - 128) * (1600 / 256))[amps >= 0.25 * amps.max()])
    print(f"{name_fzc_tl} FZC time lag SD value is {fzc_sd}")

    micro = '\u03BC'

    fig, ax = plt.subplots(NUM_CHANNELS, 1, sharex=True, sharey=True, figsize=(6, 9))
    valid = amps >= 0.25 * amps[main_c]
    window_l = abs((-dep_inds[main_c] + dep_inds[valid]).min()) + 1
    window_r = abs((-dep_inds[main_c] + dep_inds[valid]).max()) + 1
    window = window_l + window_r
    for i, c in enumerate(chunks):
        color = 'k' if not valid[i] else (color_main if i == main_c else color_second)
        i_ax = NUM_CHANNELS - i - 1
        ax[i_ax].plot(c[dep_inds[main_c] - window_l: dep_inds[main_c] + window_r], color=color, alpha=0.5)
        ax[i_ax].axhline(y=med, xmin=0, xmax=2 * window, c='k', alpha=0.3, lw=0.5)
        ax[i_ax].axvline(x=window_l, ymin=-1, ymax=1, color='k', linestyle='--')
        ax[i_ax].axvline(x=window_l - dep_inds[main_c] + dep_inds[i], ymin=-1, ymax=1, color=color)
        ax[i_ax].axis('off')
        ax[i_ax].set_xlim(0, window)
        if valid[i]:
            ax[i_ax].annotate(f"d={'X' if not valid[i] else 1.6e3 * (-dep_inds[main_c] + dep_inds[i]) / 256} {micro}s",
                              xy=(window * 0.75, 0))

    plt.savefig(SAVE_PATH + f"{name_fzc_tl}_fzc_time.pdf", transparent=True)
    clear()

    return fzc_ss, fzc_sd


def time_lags_graph(df):
    conversion = 1600 / 256
    labels = df.label.to_numpy()
    fmc = df.fzc_sd.to_numpy() * conversion
    smc = df.szc_sd.to_numpy() * conversion
    neg = df.dep_sd.to_numpy() * conversion

    fig, ax = plt.subplots()
    xs = np.asarray([1, 2, 3])
    ax.set_ylabel("SD")
    ax.set_xlabel('Event')
    counter = 0
    x_shift = 0.1
    ax.set_xticks(xs)
    ax.set_xticklabels(['FMC', 'NEG', 'SMC'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for label, marker, c in [('PV', '--o', PV_COLOR), ('PYR', '--v', PYR_COLOR)]:
        inds = labels == label
        fmc_temp, smc_temp, neg_temp = fmc[inds], smc[inds], neg[inds]
        stacked = np.vstack((fmc_temp, neg_temp, smc_temp))
        meds = np.median(stacked, axis=1)
        q25 = np.quantile(stacked, 0.25, axis=1)
        q75 = np.quantile(stacked, 0.75, axis=1)
        ax.errorbar(xs + x_shift * counter - 0.5 * x_shift, meds, yerr=[meds - q25, q75 - meds], fmt=marker, c=c)
        # ax.plot(xs + x_shift * counter - 0.5 * x_shift, meds, c=c)
        counter += 1

        print(f'{label} Wilcoxon for FMC->NEG:', stats.wilcoxon(fmc_temp, neg_temp))
        print(f'{label} Wilcoxon for NEG->SMC:', stats.wilcoxon(neg_temp, smc_temp))
        print(f'{label} Wilcoxon for FMC->SMC:', stats.wilcoxon(fmc_temp, smc_temp))

    plt.savefig(SAVE_PATH + f"time_lag_graph.pdf", transparent=True)
    clear()


def euclidean_dist(point_a, point_b):
    """
    inputs:
    pointA: (x,y) tuple representing a point in 2D space
    pointB: (x,y) tuple representing a point in 2D space

    returns:
    The euclidean distance between the points
    """
    return math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)


def calculate_distances_matrix(coordinates):
    """
    inputs:
    coordinates: a list of (x, y) tuples representing the coordinates of different channels

    returns:
    A 2D matrix in which a cell (i, j) contains the distance from coordinate i to coordinate j
    """
    distances = np.zeros((NUM_CHANNELS, NUM_CHANNELS))
    for i in range(NUM_CHANNELS):
        for j in range(NUM_CHANNELS):
            distances[i, j] = euclidean_dist(coordinates[i], coordinates[j])

    return distances


def graph_vals(clu, clu_delta, name):
    chunks = clu.calc_mean_waveform()
    if UPSAMPLE != 1:
        chunks = Spike(upsample_spike(chunks.data, UPSAMPLE, 20_000))

    chunks = chunks.get_data()

    coordinates = COORDINATES
    dists = calculate_distances_matrix(coordinates)

    amp = chunks.max(axis=1) - chunks.min(axis=1)

    arr = clu_delta
    threshold = 0.25 * amp.max()  # Setting the threshold to be self.thr the size of max depolarization

    g_temp = []
    for i in range(NUM_CHANNELS):
        max_dep_index = arr[i].argmin()
        if amp[i] >= threshold:
            g_temp.append((i, max_dep_index))
    g_temp.sort(key=lambda x: x[1])
    assert len(g_temp) > 0

    velocities = []
    for i, (channel1, timestep1) in enumerate(g_temp):
        for channel2, timestep2 in g_temp[i + 1:]:
            if timestep2 != timestep1:
                velocity = dists[channel1, channel2] / (1.6 * (timestep2 - timestep1) / 256)
                print(f"{name} edge weight between channel {channel1} ({timestep1}) and channel {channel2} "
                      f"({timestep2}) is {velocity}")
                velocities.append(velocity)

    avg_vel = np.mean(velocities)
    print(f"{name} average edge weight is {avg_vel}")
    return avg_vel


def ach(bin_range, name, clu, min_range=0):
    N = 2 * bin_range + 1
    offset = 1 / 2
    bins = np.linspace(-bin_range - offset, bin_range + offset, N + 1)
    c = PV_COLOR if 'pv' in name else PYR_COLOR

    try:
        hist = np.load(f'./ach_{name}_{bin_range}.npy')
    except FileNotFoundError:
        chunks = np.array([np.arange(len(clu.timings))])
        hist = calc_temporal_histogram(clu.timings, bins, chunks)[0]

    # decided to remove upsampling for adobe illustrator complexity reasons
    """hist_up = signal.resample_poly(hist, UPSAMPLE ** 2, UPSAMPLE, padtype='line')
    hist_up = np.where(hist_up >= 0, hist_up, 0)

    bin_inds = np.linspace(bins[0] + offset, bins[-1] - offset, UPSAMPLE * (N - 1))"""

    bin_inds = np.convolve(bins, [0.5, 0.5], mode='valid')
    hist_up = hist

    fig, ax = plt.subplots()
    ax.bar(bin_inds, hist_up, color=c, width=bins[1] - bins[0])
    ax.set_ylim(0, 60)
    plt.savefig(SAVE_PATH + f"{name}_ACH_{bin_range}.pdf", transparent=True)
    clear()

    np.save(f'./ach_{name}_{bin_range}.npy', hist)

    center_ind = len(hist_up) // 2
    side = (hist_up + hist_up[::-1])[center_ind + min_range:] / 2
    bin_inds_side = bin_inds[center_ind + min_range:] + min_range

    fig, ax = plt.subplots()
    ax.bar(bin_inds_side, side, color=c, width=bins[1] - bins[0])
    # ax.set_ylim(0, 60)
    plt.savefig(SAVE_PATH + f"{name}_ACH_{min_range}-{bin_range}.pdf", transparent=True)

    return hist


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
    ax.set_ylim(ymin=0, ymax=1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ret = np.sum(dists) / len(cdf)
    print(f"{name} unif_dist is {ret}")
    plt.savefig(SAVE_PATH + f"{name}_unif_dist.pdf", transparent=True)
    clear()

    return ret


def dkl_mid(hist, name):
    color_main = PYR_COLOR if name == 'pyr' else PV_COLOR
    color_second = LIGHT_PYR if name == 'pyr' else LIGHT_PV
    resolution = 2
    zero_bin_ind = len(hist) // 2
    hist = (hist[:zero_bin_ind + 1:][::-1] + hist[zero_bin_ind:]) / 2
    # decided to remove upsampling for adobe illustrator complexity reasons
    """hist_up = signal.resample_poly(hist, UPSAMPLE ** 2, UPSAMPLE, padtype='line')
    hist_up = np.where(hist_up >= 0, hist_up, 0)
    mid = hist_up[50 * resolution * UPSAMPLE:]"""
    mid = hist[50 * resolution:]
    probs = mid / np.sum(mid)
    unif = np.ones(probs.shape) / len(probs)
    dkl = stats.entropy(probs, unif)
    print(f"{name} mid_dkl value is {dkl}")
    inds = probs == 0
    probs[inds] = 1
    dkl_func = probs * np.log(probs / unif)
    dkl_func[inds] = 0
    fig, ax = plt.subplots()
    x = np.linspace(50, 1000, len(dkl_func))
    ax.plot(x, dkl_func, c=color_main)
    ax.fill_between(x, dkl_func, alpha=0.2, color=color_second)
    plt.savefig(SAVE_PATH + f"{name}_dkl_mid.pdf", transparent=True)
    clear()

    return dkl


def plot_delta(clu, name):
    color_main = PYR_COLOR if name == 'pyr' else PV_COLOR
    color_second = LIGHT_PYR if name == 'pyr' else LIGHT_PV
    chunks = clu.calc_mean_waveform()
    main_c = get_main(chunks.data)
    if UPSAMPLE != 1:
        chunks = Spike(upsample_spike(chunks.data, UPSAMPLE, 20_000))
    chunks = chunks.get_data()
    chunk_amp = chunks.max(axis=1) - chunks.min(axis=1)
    amp_thr = (chunk_amp >= 0.25 * chunk_amp.max())
    chunks = chunks / (-chunks.min())
    delta = np.zeros(chunks.shape)
    inds = []
    med = np.median(chunks)
    for i in range(len(chunks)):
        sig_m = np.convolve(np.where(chunks[i] <= med, -1, 1), [-1, 1], 'same')
        sig_m[0] = sig_m[-1] = 1
        ind = calc_pos_imp(sig_m, chunks[i].argmin(), DELTA_MODE.F_ZCROSS)
        inds.append(ind)
    delta[np.arange(NUM_CHANNELS), inds] = chunks.min(axis=1)

    fig, ax = plt.subplots(NUM_CHANNELS, 1, sharex=True, sharey=True, figsize=(3, 20))
    for i, c_ax in enumerate(ax[::-1]):
        c_ax.plot(chunks[i], c=color_second if amp_thr[i] else '#505050')
        c_ax.plot(delta[i], c=color_main if amp_thr[i] else '#505050')
        c_ax.axis('off')

    plt.savefig(SAVE_PATH + f"{name}_delta.pdf", transparent=True)
    clear()

    shift = chunks.shape[-1] // 2 - delta[main_c].argmin()
    ret = np.roll(delta, shift, axis=1)

    fig, ax = plt.subplots(NUM_CHANNELS, 1, sharex=True, sharey=True, figsize=(6, 20))
    for i, c_ax in enumerate(ax[::-1]):
        c_ax.plot(delta[i], c=color_second if amp_thr[i] else '#505050')
        c_ax.plot(ret[i], c=color_main if amp_thr[i] else '#505050')
        c_ax.annotate(text='', xy=(delta[i].argmin(), delta[i].min() / 2), xytext=(ret[i].argmin(), ret[i].min() / 2),
                      arrowprops=dict(arrowstyle='<-'))

        c_ax.axis('off')

    plt.savefig(SAVE_PATH + f"{name}_delta_cent.pdf", transparent=True)
    clear()

    return ret


def get_delta_results(modality, chunk_size=[0], res_name='results_rf_030422_trans_wf'):
    results = pd.read_csv(f'../ml/{res_name}.csv', index_col=0)
    complete = results[results.restriction == 'complete']
    complete = complete[complete.modality == modality]
    complete = complete.dropna(how='all', axis=1)

    d = {'spatial': SPATIAL, 'morphological': MORPHOLOGICAL, 'temporal': TEMPORAL, 'trans_wf': TRANS_MORPH}
    modalities = [(modality, d[modality])]

    plot_conf_mats(complete, 'complete', name='delta', chunk_size=chunk_size, modalities=modalities,
                   data_path='../Datasets/data_sets_030422_trans_wf/complete_0/trans_wf/0_0.800.2/')
    clear()

    plot_roc_curve(complete, name='delta', chunk_size=chunk_size, modalities=modalities)
    clear()


def calc_pos(arr, start_pos):
    pos = start_pos
    while pos >= 0:
        if arr[pos] == 0:
            pos -= 1
        else:
            break
    return pos


def match_spike(channel, med):
    pos = channel.argmin()
    spike = np.zeros(channel.shape)
    sig_m = np.convolve(np.where(channel <= med, -1, 1), [-1, 1], 'same')
    sig_m[0] = sig_m[-1] = 1
    pos = calc_pos(sig_m, pos)
    spike[pos] = channel.min()

    return spike, pos


def match_chunk(wf, amp):
    ret = np.zeros(wf.shape)
    main_c = amp.argmax()
    roll_val = 0
    med = np.median(wf)
    for i, channel in enumerate(wf):
        ret[i], pos = match_spike(channel, med)
        if i == main_c:
            roll_val = wf.shape[-1] // 2 - pos

    ret = np.roll(ret, roll_val, axis=1)
    chunk = ret / abs(ret.min())

    return chunk


def wf_heatmap(labels, path='../cluster_data/clusterData_no_light_030422_wf_trans/mean_spikes/'):
    waveforms = []
    files = [path + f for f in listdir(path) if isfile(join(path, f))]
    for f in files:
        waveforms.append(np.load(f))

    pv_inds = np.argwhere(labels == 'PV').flatten()
    pyr_inds = np.argwhere(labels == 'PYR').flatten()

    if UPSAMPLE != 1:
        waveforms_up = np.asarray([upsample_spike(wf, UPSAMPLE, 20_000) for wf in waveforms])
    else:
        waveforms_up = waveforms

    main_channels = np.asarray([wf[get_main(wf)] for wf in waveforms_up])
    main_channels = main_channels / np.expand_dims(abs(main_channels.min(axis=1)), axis=1)

    reg_pv_order = pv_inds[np.argsort(main_channels[pv_inds].argmin(axis=1))]
    reg_pyr_order = pyr_inds[np.argsort(main_channels[pyr_inds].argmin(axis=1))]
    reg_inds = np.concatenate((reg_pv_order, reg_pyr_order))
    reg_sorted_wf = main_channels[reg_inds]

    fig, ax = plt.subplots()
    # cmap = sns.color_palette("Spectral_r", as_cmap=True)
    # ax = sns.heatmap(reg_sorted_wf, vmin=-1, vmax=1, cmap=cmap)
    shw = ax.imshow(reg_sorted_wf, vmin=-1, vmax=reg_sorted_wf.max(), cmap='gray', interpolation='none')
    ax.hlines(y=len(pv_inds), xmin=0, xmax=256, color='k')
    plt.colorbar(shw)
    plt.savefig(SAVE_PATH + f"heatmap_org.pdf", transparent=True)

    trans_wfs = []
    for wf in waveforms_up:
        amp = wf.max(axis=1) - wf.min(axis=1)
        wavelets_fzc = match_chunk(wf, amp)
        trans_wfs.append(wavelets_fzc)
    trans_main_channels = np.asarray([wf[get_main(wf)] for wf in trans_wfs])

    trans_pv_order = pv_inds[np.argsort(trans_main_channels[pv_inds].argmin(axis=1))]
    trans_pyr_order = pyr_inds[np.argsort(trans_main_channels[pyr_inds].argmin(axis=1))]
    trans_inds = np.concatenate((trans_pv_order, trans_pyr_order))
    trans_sorted_wf = trans_main_channels[trans_inds]

    fig, ax = plt.subplots()
    # ax = sns.heatmap(trans_sorted_wf, vmin=-1, vmax=1, cmap=cmap)
    shw = ax.imshow(trans_sorted_wf, vmin=-1, vmax=reg_sorted_wf.max(), cmap='gray', interpolation='none')
    ax.hlines(y=len(pv_inds), xmin=0, xmax=256, color='k')
    plt.colorbar(shw)

    plt.savefig(SAVE_PATH + f"heatmap_trans.pdf", transparent=True)
    clear()


def chunk_fig(clu, name, cz):
    color = PYR_COLOR if name == 'pyr' else PV_COLOR
    spikes = clu.spikes.copy()
    np.random.shuffle(spikes)
    spikes_show = [0, 1, cz - 1, cz, cz + 1, 2 * cz - 1, -cz, -cz + 1, -1]
    for spike in spikes_show:
        fig, ax = plt.subplots(NUM_CHANNELS, 1, sharex=True, sharey=True, figsize=(6, 20))
        Spike(data=spikes[spike]).plot_spike(ax=ax, c=color)
        for c_ax in ax:
            c_ax.axis('off')
        plt.savefig(SAVE_PATH + f"{name}_chunk_fig_{spike}.pdf", transparent=True)
        clear()
    for i in range(3):
        fig, ax = plt.subplots(NUM_CHANNELS, 1, sharex=True, sharey=True, figsize=(6, 20))
        start, end = spikes_show[3 * i], spikes_show[3 * i + 2]
        spike_avg = spikes[start: end].mean(axis=0)
        Spike(data=spike_avg).plot_spike(ax=ax, c=color)
        for c_ax in ax:
            c_ax.axis('off')
        plt.savefig(SAVE_PATH + f"{name}_chunk_fig_chunk{i}.pdf", transparent=True)
        clear()


def chunk_results(res_name='results_rf_combined'):
    results = pd.read_csv(f'../ml/{res_name}.csv', index_col=0)
    complete = results[results.restriction == 'complete']

    drop = [col for col in complete.columns.values if col not in ['modality', 'chunk_size', 'seed', 'auc']]

    complete = complete.drop(columns=drop)

    plot_auc_chunks_bp(complete, name='chunks_rf')
    clear()


def get_comp_results(chunk_sizes=(BEST_WF_CHUNK, BEST_TEMPORAL_CHUNK, BEST_SPATIAL_CHUNK),
                     res_name='results_rf_030822_rich_region'):
    results = pd.read_csv(f'../ml/{res_name}.csv', index_col=0)
    complete = results[results.restriction == 'complete']

    drop = [col for col in complete.columns.values if col not in ['restriction', 'modality', 'chunk_size', 'seed',
                                                                  'acc', 'pyr_acc', 'in_acc', 'dev_acc',
                                                                  'dev_pyr_acc', 'dev_in_acc', 'auc', 'fpr', 'tpr',
                                                                  'dev_auc', 'dev_fpr', 'dev_tpr']]

    complete = complete[complete.chunk_size.isin(chunk_sizes)]
    complete = complete.drop(columns=drop)

    for mod_name, mod_fets, cs in [('spatial', SPATIAL, BEST_SPATIAL_CHUNK),
                                   ('temporal', TEMPORAL, BEST_TEMPORAL_CHUNK),
                                   ('morphological', BEST_WF_CHUNK, 50)]:
        # ROC curves
        plot_roc_curve(complete, name='NCX', chunk_size=[cs], use_dev=False, modalities=[(mod_name, mod_fets)])
        clear()
        plot_roc_curve(complete, name='CA1', chunk_size=[cs], use_dev=True, modalities=[(mod_name, mod_fets)])
        clear()

    #  AUC
    # chunk sizes correspond to morphological, temporal spatial
    plot_test_vs_dev_bp(complete, chunk_sizes=(BEST_WF_CHUNK, BEST_TEMPORAL_CHUNK, BEST_SPATIAL_CHUNK),
                        name='ncx_vs_ca1_auc')
    clear()
    plot_test_vs_dev_bp(complete, chunk_sizes=(BEST_WF_CHUNK, BEST_TEMPORAL_CHUNK, BEST_SPATIAL_CHUNK),
                        name='ncx_vs_ca1_auc', diff=True)
    clear()


def main_figs():
    # Load wanted units - no light
    pyr_cluster = load_cluster(TEMP_PATH_NO_LIGHT, pyr_name)
    pv_cluster = load_cluster(TEMP_PATH_NO_LIGHT, pv_name)

    trans_units(pyr_cluster)
    trans_units(pv_cluster)

    # Waveforms
    pyr_cluster.plot_cluster(save=True, path=SAVE_PATH, name='no_light')
    clear()
    pv_cluster.plot_cluster(save=True, path=SAVE_PATH, name='no_light')
    clear()

    # ACH
    _ = ach(30, 'pv', pv_cluster)
    _ = ach(30, 'pyr', pyr_cluster)

    df = load_df(['region', 'label'], trans_labels=False)
    pie_chart(df)

    # Morphological features - figure 2

    pyr_t2p, pyr_fwhm = t2p_fwhm(pyr_cluster, PYR_COLOR, LIGHT_PYR, 'pyr')
    pv_t2p, pv_fwhm = t2p_fwhm(pv_cluster, PV_COLOR, LIGHT_PV, 'pv')

    pyr_acceleration = acceleration(pyr_cluster, PYR_COLOR, LIGHT_PYR, 'pyr')
    pv_acceleration = acceleration(pv_cluster, PV_COLOR, LIGHT_PV, 'pv')

    features = ['break_measure', 'fwhm', 'get_acc', 'max_speed', 'peak2peak', 'trough2peak', 'rise_coef', 'smile_cry',
                'label']
    df = load_df(features)

    d_features = ['fwhm', 'trough2peak', 'get_acc']
    density_plots(df, d_features, 'morph',
                  [[pyr_fwhm, pv_fwhm], [pyr_t2p, pv_t2p], [pyr_acceleration, pv_acceleration]])

    get_results('morphological', BEST_WF_CHUNK)

    thr_confusion_mats()

    # Temporal features - figure 2

    # Spike train
    # _, _, _, stims = remove_light(pyr_cluster, True, data_path='../Data/')
    # check indicated that it is way after the first 50 spikes so it is ignored in plotting

    # Spike Train
    pyr_mask = (pyr_cluster.timings > 3509050) * (pyr_cluster.timings < 3510050)
    pv_mask = (pv_cluster.timings > 3509050) * (pv_cluster.timings < 3510050)
    pyr_train = pyr_cluster.timings[pyr_mask]
    pv_train = pv_cluster.timings[pv_mask]

    fig, ax = plt.subplots(figsize=(10, 1))
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
    clear()

    # ACHs

    hist_pv = ach(50, 'pv_no_light', pv_cluster)
    hist_pyr = ach(50, 'pyr_no_light', pyr_cluster)
    hist_pv_long = ach(1000, 'pv_no_light', pv_cluster, min_range=50)
    hist_pyr_long = ach(1000, 'pyr_no_light', pyr_cluster, min_range=50)

    # unif_dist

    features = ['d_kl_start', 'd_kl_mid', 'jump', 'psd_center', 'der_psd_center', 'rise_time', 'unif_dist',
                'firing_rate', 'label']
    df = load_df(features)

    pv_unif_dist = unif_dist(hist_pv, 'pv')
    pyr_unif_dist = unif_dist(hist_pyr, 'pyr')

    pv_dkl_mid = dkl_mid(hist_pv_long, 'pv')
    pyr_dkl_mid = dkl_mid(hist_pyr_long, 'pyr')

    d_features = ['unif_dist', 'd_kl_mid']
    density_plots(df, d_features, 'temporal', [[pyr_unif_dist, pv_unif_dist], [pyr_dkl_mid, pv_dkl_mid]])

    get_results('temporal', BEST_TEMPORAL_CHUNK)

    # modified waveforms - fig 4

    wf_heatmap(df.label.to_numpy())

    pyr_fzc_delta = plot_delta(pyr_cluster, 'pyr')
    pv_fzc_delta = plot_delta(pv_cluster, 'pv')

    get_delta_results('trans_wf')

    # Spatial features - figure 5 - 6

    pyr_spd = spd(pyr_cluster, 'pyr')
    pv_spd = spd(pv_cluster, 'pv')

    pyr_fzc, pyr_fzc_sd = fzc_time_lag(pyr_cluster, 'pyr')
    pv_fzc, pv_fzc_sd = fzc_time_lag(pv_cluster, 'pv')

    df = load_df(['dep_sd', 'fzc_sd', 'szc_sd', 'label'])
    time_lags_graph(df)

    pyr_graph_avg = graph_vals(pyr_cluster, pyr_fzc_delta, 'pyr')
    pv_graph_avg = graph_vals(pv_cluster, pv_fzc_delta, 'pv')

    features = ['spatial_dispersion_count', 'spatial_dispersion_sd', 'spatial_dispersion_area', 'dep_red', 'dep_sd',
                'fzc_red', 'fzc_sd', 'szc_red', 'szc_sd', 'dep_graph_avg_speed', 'dep_graph_slowest_path',
                'dep_graph_fastest_path', 'fzc_graph_avg_speed', 'fzc_graph_slowest_path', 'fzc_graph_fastest_path',
                'szc_graph_avg_speed', 'szc_graph_slowest_path', 'szc_graph_fastest_path', 'label']
    df = load_df(features)

    d_features = ['spatial_dispersion_count', 'fzc_red', 'fzc_sd', 'fzc_graph_avg_speed']
    density_plots(df, d_features, 'spatial',
                  [[pyr_spd, pv_spd], [pyr_fzc, pv_fzc], [pyr_fzc_sd, pv_fzc_sd], [pyr_graph_avg, pv_graph_avg]])

    get_results('spatial', BEST_SPATIAL_CHUNK)

    # chunks - fig 6

    # chunk_fig(pyr_cluster, 'pyr', BEST_SPATIAL_CHUNK)

    chunk_results()

    # nCX vs CA1 - fig 7

    get_comp_results()


def costume_imp(modality, chunk_size, res_name, base_name, imp_inds, name_mapping, name):
    results = pd.read_csv(f'../ml/{res_name}.csv', index_col=0)
    complete = results[results.restriction == 'complete']
    complete = complete[complete.modality == modality]

    complete = complete[complete.chunk_size.isin([chunk_size])]
    complete = complete.dropna(how='all', axis=1)

    grouped_complete = complete.groupby(by=['restriction', 'modality', 'chunk_size'])

    base = pd.read_csv(f'../ml/{base_name}.csv', index_col=0)
    base = base[base.restriction == 'complete']
    base = base[base.modality == modality]

    base = base[base.chunk_size.isin([chunk_size])]
    base = base.dropna(how='all', axis=1)

    grouped_base = base.groupby(by=['restriction', 'modality', 'chunk_size'])

    modalities = [(modality, imp_inds)]

    plot_fet_imp(grouped_complete.median(), grouped_complete.quantile(0.75), 'complete', grouped_base.median(),
                 chunk_size=[chunk_size], name=f'{modality}_{name}', modalities=modalities,
                 semsn=grouped_complete.quantile(0.25),
                 fet_names_map=name_mapping)
    clear()

def chunks_comp(res_name_a='results_rf_combined', res_name_b='results_rf_290322_fix_imp'):
    results_a = pd.read_csv(f'../ml/{res_name_a}.csv', index_col=0)
    complete_a = results_a[results_a.restriction == 'complete']

    drop = [col for col in complete_a.columns.values if col not in ['modality', 'chunk_size', 'seed', 'auc']]

    complete_a = complete_a.drop(columns=drop)

    results_b = pd.read_csv(f'../ml/{res_name_b}.csv', index_col=0)
    complete_b = results_b[results_b.restriction == 'complete']
    complete_b = complete_b.drop(columns=drop)

    for mod in ['spatial', 'temporal', 'morphological']:
        ax = plot_auc_chunks_bp(complete_a[complete_a.modality == mod], plot=False, shift=-0.3)

        plot_auc_chunks_bp(complete_b [complete_b.modality == mod], name='chunks_rf_comp', ax_inp=ax, edge_color='r', shift=0.3)
        clear()

def spatial_var(path):
    arr = []
    labels = []
    files = os.listdir(path)
    df = None

    for file in sorted(files):
        if df is None:
            df = pd.read_csv(path + '/' + file)
        else:
            temp = pd.read_csv(path + '/' + file)
            df = df.append(temp)
    df = np.array(df.to_numpy()[:, SPATIAL[:-1]], dtype=np.float64)
    scaler = StandardScaler()
    scaler.fit(df)

    for file in sorted(files):
        df_temp = pd.read_csv(path + '/' + file)
        df_np = np.array(df_temp.to_numpy()[:, SPATIAL[:-1]], dtype=np.float64)
        df_np = scaler.transform(df_np)

        std = np.std(df_np, axis=0)
        arr.append(std)
        label = df_temp.label.to_numpy()[0]
        labels.append(int(label))

    columns = df_temp.columns.values[SPATIAL[:-1]]

    arr, labels = np.asarray(arr), np.asarray(labels)

    arr_pyr = arr[labels == 1]
    arr_pv = arr[labels == 0]

    p = stats.mannwhitneyu(arr_pyr.flatten(), arr_pv.flatten()).pvalue
    print(f'MW for PYR compared to PV in general: p-val={p}')

    arr_pyr_mean = arr_pyr.mean(axis=0)
    arr_pv_mean = arr_pv.mean(axis=0)

    p = stats.wilcoxon(arr_pyr_mean, arr_pv_mean).pvalue
    print(f'Wilcoxon for PYR compared to PV in general: p-val={p}')

    for pyr, pv, pyr_all, pv_all, col in zip(arr_pyr_mean, arr_pv_mean, arr_pyr.T, arr_pv.T, columns):
        p = stats.mannwhitneyu(pyr_all, pv_all).pvalue
        print(f"{col}: pyr={round(pyr, 3)}, pv={round(pv, 3)}. p-val={p}")
        plt.plot([pyr, pv], c='k', marker='o')

    plt.bar([0, 1], [arr_pyr.mean(), arr_pv.mean()], color='#A0A0A0')

    plt.xticks(ticks=[0, 1], labels=['PYR', 'PV'])

    plt.savefig(SAVE_PATH + f"spatial_var.pdf", transparent=True)
    clear()

def appendix_figs():
    chunks_comp()
    raise AssertionError

    # Appendix A
    # Correlation matrices
    # Waveform
    features = ['break_measure', 'fwhm', 'get_acc', 'max_speed', 'peak2peak', 'trough2peak', 'rise_coef', 'smile_cry',
                'label']
    df = load_df(features)

    order_morph = ['trough2peak', 'peak2peak', 'fwhm', 'rise_coef', 'max_speed', 'break_measure', 'smile_cry',
                   'get_acc']
    corr_mat(df, 'morph', order_morph)

    # Spike-timing
    features = ['d_kl_start', 'd_kl_mid', 'jump', 'psd_center', 'der_psd_center', 'rise_time', 'unif_dist',
                'firing_rate', 'label']
    df = load_df(features)

    order_temp = ['unif_dist', 'd_kl_start', 'rise_time', 'jump', 'd_kl_mid', 'psd_center', 'der_psd_center',
                  'firing_rate']
    corr_mat(df, 'temporal', order_temp)

    # Spatial
    features = ['spatial_dispersion_count', 'spatial_dispersion_sd', 'spatial_dispersion_area', 'dep_red', 'dep_sd',
                'fzc_red', 'fzc_sd', 'szc_red', 'szc_sd', 'dep_graph_avg_speed', 'dep_graph_slowest_path',
                'dep_graph_fastest_path', 'fzc_graph_avg_speed', 'fzc_graph_slowest_path', 'fzc_graph_fastest_path',
                'szc_graph_avg_speed', 'szc_graph_slowest_path', 'szc_graph_fastest_path', 'label']
    df = load_df(features)

    order_spat = ['dep_red', 'dep_sd', 'fzc_red', 'fzc_sd', 'szc_red', 'szc_sd', 'dep_graph_avg_speed',
                  'dep_graph_slowest_path', 'dep_graph_fastest_path', 'fzc_graph_avg_speed', 'fzc_graph_slowest_path',
                  'fzc_graph_fastest_path', 'szc_graph_avg_speed', 'szc_graph_slowest_path', 'szc_graph_fastest_path',
                  'spatial_dispersion_count', 'spatial_dispersion_sd', 'spatial_dispersion_area']
    corr_mat(df, 'spatial', order_spat)

    # Appendix B

    # Appendix C
    # Moments importances
    imp_inds = np.arange(NUM_MOMENTS + 2)  # extra +1 as a workaround for usual case where last value is -1
    name_mapping = ['Original', 'Mean', 'SD', 'Q25%', 'Median', 'Q75%']
    res_name = 'results_rf_moments'
    base_name = 'results_rf_moments_chance_balanced'
    name = 'moments'
    # Waveform
    costume_imp('morphological', BEST_WF_CHUNK, res_name, base_name, imp_inds, name_mapping, name)
    # Spike-timing
    costume_imp('temporal', BEST_TEMPORAL_CHUNK, res_name, base_name, imp_inds, name_mapping, name)
    # Spatial
    costume_imp('spatial', BEST_SPATIAL_CHUNK, res_name, base_name, imp_inds, name_mapping, name)

    # Event importances
    imp_inds = np.arange(NUM_EVENTS + 1)  # extra +1 as a workaround for usual case where last value is -1
    name_mapping = ['FMC', 'NEG', 'SMC']
    res_name = 'results_rf_events'
    base_name = 'results_rf_events_chance_balanced'
    name = 'events'
    costume_imp('spatial', BEST_SPATIAL_CHUNK, res_name, base_name, imp_inds, name_mapping, name)

    spatial_var(f'../cluster_data/clusterData_no_light_29_03_22/{BEST_SPATIAL_CHUNK}')


if __name__ == '__main__':
    # main_figs()
    appendix_figs()
