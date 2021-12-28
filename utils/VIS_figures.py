import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats
import os
import seaborn as sns

from clusters import Spike
from preprocessing_pipeline import load_cluster
from constants import PV_COLOR, LIGHT_PV, PYR_COLOR, LIGHT_PYR
from constants import SPATIAL, MORPHOLOGICAL, TEMPORAL
from features.temporal_features_calc import calc_temporal_histogram
from features.spatial_features_calc import DELTA_MODE, calc_pos
from utils.upsampling import upsample_spike
from ml.plot_tests import plot_results, plot_fet_imp, plot_conf_mats, plot_roc_curve, plot_test_vs_dev, plot_acc_vs_auc
# Note that plot_conf_mats requires path for the correct dataset

SAVE_PATH = '../../../data for figures/New/'  # make sure to change in plot_tests as well
TEMP_PATH = '../temp_state/'
DATA_PATH = '../clustersData_no_light_FINAL/0'
pyr_name = name = 'es25nov11_13_3_3'  # pyr
pv_name = 'es25nov11_13_3_11'  # pv

NUM_CHANNELS = 8
TIMESTEPS = 32
UPSAMPLE = 8

# the following values are required to transform the spikes from arbitrary units to V
# this values are based on the corresponding XML file, change them accordingly if changing units
VOL_RANGE = 8
AMPLIFICATION = 1000
NBITS = 16

def clear():
    plt.clf()
    plt.cla()
    plt.close('all')


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
    rect = patches.Rectangle((fwhm_inds[0] - 5, dep / 2), fwhm_inds[-1] - fwhm_inds[0] + 10, dep / 2,
                             facecolor=sec_color, alpha=0.2)
    ax.add_patch(rect)
    ax.plot([dep_ind, hyp_ind], [dep, hyp], marker='o', linestyle='None', c=color)

    plt.savefig(SAVE_PATH + f"{name}_t2p_fwhm.pdf", transparent=True)
    clear()


def max_speed(clu, color, sec_color, name):
    chunks = clu.calc_mean_waveform()
    if UPSAMPLE != 1:
        chunks = upsample_spike(chunks.data)

    spike = chunks[get_main(chunks)]

    spike_der = np.convolve(spike, [1, -1], mode='valid')
    max_speed_inds = np.argwhere(spike_der > spike_der[131]).flatten()
    print(f"{name} max_speed is {1.6 * (max_speed_inds[-1] - max_speed_inds[0] + 1) / len(spike)} ms")

    fig, ax = plt.subplots()
    ax.plot(spike_der, c=color)
    rect = patches.Rectangle((max_speed_inds[0] - 5, spike_der[131]), max_speed_inds[-1] - max_speed_inds[0] + 10,
                             spike_der.max() - spike_der[131], facecolor=sec_color, alpha=0.2)
    ax.add_patch(rect)
    plt.savefig(SAVE_PATH + f"{name}_max_speed.pdf", transparent=True)
    clear()


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
    if trans_labels:
        df.label = df.label.map({1: 'PYR', 0: 'IN'})
    df = df[features]

    return df


def corr_mat(df, modality):
    correlation_matrix = df.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    plt.yticks(rotation=30)
    cmap = sns.color_palette("vlag", as_cmap=True)
    _ = sns.heatmap(correlation_matrix, annot=True, fmt='.2f', mask=mask, vmin=-1, vmax=1, cmap=cmap,
                    annot_kws={"fontsize": 5})
    plt.savefig(SAVE_PATH + f"{modality}_cor_mat.pdf", transparent=True)
    clear()


def density_plots(df, d_features, modality):
    palette = {"PYR": PYR_COLOR, "IN": PV_COLOR}

    for c in df.columns:
        if c not in d_features:
            continue
        # bw_adjust = (df[c].max() - df[c].min()) / 50
        _ = sns.displot(data=df, x=c, hue="label", common_norm=False, kind="kde", fill=True,
                        palette=palette)
        plt.savefig(SAVE_PATH + f"{modality}_density_{c}.pdf", transparent=True)
        clear()
        ax = sns.ecdfplot(data=df, x=c, hue="label", palette=palette)
        ax.set_ylim(ymin=0, ymax=1.1)
        plt.savefig(SAVE_PATH + f"{modality}_cdf_{c}.pdf", transparent=True)
        clear()


def get_results(modality, chunk_size=[0], res_name='results_rf'):
    results = pd.read_csv(f'../ml/{res_name}.csv', index_col=0)
    complete = results[results.restriction == 'complete']
    complete = complete[complete.modality == modality]

    complete = complete[complete.chunk_size.isin(chunk_size)]
    complete = complete[complete.modality == modality]
    complete = complete.dropna(how='all', axis=1)

    grouped_complete = complete.groupby(by=['restriction', 'modality', 'chunk_size'])

    plot_results(grouped_complete.mean(), grouped_complete.sem(), 'complete', acc=True, chunk_size=chunk_size,
                 name=modality)
    clear()

    plot_results(grouped_complete.mean(), grouped_complete.sem(), 'complete', acc=False, chunk_size=chunk_size,
                 name=modality)
    clear()
    d = {'spatial': SPATIAL, 'morphological': MORPHOLOGICAL, 'temporal': TEMPORAL}
    modalities = [(modality, d[modality])]

    plot_fet_imp(grouped_complete.mean(), grouped_complete.sem(), 'complete', chunk_size=chunk_size, name=modality,
                 modalities=modalities)
    clear()

    plot_conf_mats(complete, 'complete', name=modality, chunk_size=[0], modalities=modalities)
    clear()

    plot_roc_curve(complete, name=modality, chunk_size=[0], modalities=modalities)
    clear()


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


def fzc_time_lag(clu, name):
    color_main = PYR_COLOR if name == 'pyr' else PV_COLOR
    color_second = LIGHT_PYR if name == 'pyr' else LIGHT_PV

    chunks = clu.calc_mean_waveform()
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
        ind = calc_pos(sig_m, chunks[i].argmin(), DELTA_MODE.F_ZCROSS)
        inds.append(ind)
    delta[np.arange(NUM_CHANNELS), inds] = chunks.min(axis=1)
    shift = chunks.shape[-1] // 2 - delta[main_c].argmin()
    cent_delta = np.roll(delta, shift, axis=1)

    dep_inds = delta.argmin(axis=1)
    cent_dep_inds = cent_delta.argmin(axis=1)
    print(f"{name} FZC time lag value is {((cent_dep_inds - 128) ** 2)[amps >= 0.25 * amps.max()].sum()}")

    fig, ax = plt.subplots()
    it = cent_delta[(amps >= 0.25 * amps.max())]
    it_nonc = delta[(amps >= 0.25 * amps.max())]
    channels = chunks[(amps >= 0.25 * amps.max())]
    heights = np.linspace(0, 1, np.count_nonzero(amps >= 0.25 * amps.max()) + 2)[1:-1]
    D = 256 * 4 / (it.argmin(axis=1).max() - it.argmin(axis=1).min())
    eps = 0.02
    for i, c in enumerate(it):
        color = color_main if i == main_c else color_second
        new_x = np.arange(256) / D + it_nonc[i].argmin() * (D - 1) / D - 128 + shift
        channel = (channels[i] + 1) / (2 * (len(it) + 2)) + heights[i]
        ax.plot(new_x, channel, color=color)
        rel_x = c.argmin() - 128
        if rel_x != 0:
            ax.axvline(x=rel_x, color='k', linestyle='--')
            ax.annotate(text='', xy=(rel_x, heights[i]+eps), xytext=(0, heights[i]+eps), arrowprops=dict(arrowstyle='<->'))
            ax.annotate(text=f'{abs(rel_x)}', xy=(rel_x / 2, heights[i] + 0.025))
        ax.scatter(rel_x, channel[it_nonc[i].argmin()], s=80, facecolors='none', edgecolors='k')
    ax.axis('off')
    ax.axvline(x=0, color='k')
    ax.get_yaxis().set_visible(False)

    plt.savefig(SAVE_PATH + f"{name}_fzc_time_lag_v1.pdf", transparent=True)
    clear()

    fig, ax = plt.subplots(NUM_CHANNELS, 1, sharex=True, sharey=True, figsize=(6, 9))
    window = 20
    for i, c in enumerate(chunks):
        not_valid = amps[i] < 0.25 * amps[main_c]
        color = 'k' if not_valid else (color_main if i == main_c else color_second)
        ax[i].plot(c[dep_inds[main_c] - window: dep_inds[main_c] + window], color=color, alpha=0.5)
        ax[i].axvline(x=window, ymin=-1, ymax=1, color='k', linestyle='--')
        ax[i].axvline(x=window - dep_inds[main_c] + dep_inds[i], ymin=-1, ymax=1, color=color)
        ax[i].axis('off')
        ax[i].set_xlim(0, 2 * window)
        ax[i].annotate(f"d={'X' if not_valid else abs(dep_inds[main_c] - dep_inds[i])}", xy=(window * 1.5, -0.5))

    plt.savefig(SAVE_PATH + f"{name}_fzc_time_lag_v2.pdf", transparent=True)
    clear()

    fig, ax = plt.subplots(NUM_CHANNELS, 1, sharex=True, sharey=True, figsize=(6, 9))
    for i, c in enumerate(chunks):
        not_valid = amps[i] < 0.25 * amps[main_c]
        color = 'k' if not_valid else (color_main if i == main_c else color_second)
        ax[i].plot(c, color=color, alpha=0.5)
        ax[i].axvline(x=dep_inds[main_c], ymin=-1, ymax=1, color='k', linestyle='--')
        ax[i].axvline(x=dep_inds[i], ymin=-1, ymax=1, color=color)
        ax[i].axis('off')
        ax[i].annotate(f"d={'X' if not_valid else abs(dep_inds[main_c] - dep_inds[i])}", xy=(window * 1.5, -0.5))

    plt.savefig(SAVE_PATH + f"{name}_fzc_time_lag_v3.pdf", transparent=True)
    clear()


def ach(bin_range, name, clu):
    N = 2 * bin_range + 2
    offset = 1 / 2
    bins = np.linspace(-bin_range - offset, bin_range + offset, N)
    c = PV_COLOR if name == 'pv' else PYR_COLOR

    try:
        hist = np.load(f'./ach_{name}_{bin_range}.npy')
    except FileNotFoundError:
        chunks = np.array([np.arange(len(clu.timings))])
        hist = calc_temporal_histogram(clu.timings, bins, chunks)[0]

    hist_up = signal.resample_poly(hist, UPSAMPLE ** 2, UPSAMPLE, padtype='line')
    hist_up = np.where(hist_up >= 0, hist_up, 0)

    bin_inds = np.linspace(bins[0] + offset, bins[-1] - offset, UPSAMPLE * (N - 1))

    fig, ax = plt.subplots()
    ax.bar(bin_inds, hist_up, color=c)
    ax.set_ylim(0, 60)
    plt.savefig(SAVE_PATH + f"{name}_ACH_{bin_range}.pdf", transparent=True)
    clear()

    np.save(f'./ach_{name}_{bin_range}.npy', hist)

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
    print(f"{name} unif_dist is {np.sum(dists)}")
    plt.savefig(SAVE_PATH + f"{name}_unif_dist.pdf", transparent=True)
    clear()


def dkl_mid(hist, name):
    color_main = PYR_COLOR if name == 'pyr' else PV_COLOR
    color_second = LIGHT_PYR if name == 'pyr' else LIGHT_PV
    resolution = 2
    zero_bin_ind = len(hist) // 2
    hist = (hist[:zero_bin_ind + 1:][::-1] + hist[zero_bin_ind:]) / 2
    hist_up = signal.resample_poly(hist, UPSAMPLE ** 2, UPSAMPLE, padtype='line')
    hist_up = np.where(hist_up >= 0, hist_up, 0)
    mid = hist_up[50 * resolution * UPSAMPLE:]
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


def plot_delta(clu, name):
    color_main = PYR_COLOR if name == 'pyr' else PV_COLOR
    color_second = LIGHT_PYR if name == 'pyr' else LIGHT_PV
    chunks = clu.calc_mean_waveform()
    main_c = get_main(chunks.data)
    if UPSAMPLE != 1:
        chunks = Spike(upsample_spike(chunks.data, UPSAMPLE, 20_000))
    chunks = chunks.get_data()
    chunks = chunks / (-chunks.min())
    delta = np.zeros(chunks.shape)
    inds = []
    med = np.median(chunks)
    for i in range(len(chunks)):
        sig_m = np.convolve(np.where(chunks[i] <= med, -1, 1), [-1, 1], 'same')
        sig_m[0] = sig_m[-1] = 1
        ind = calc_pos(sig_m, chunks[i].argmin(), DELTA_MODE.F_ZCROSS)
        inds.append(ind)
    delta[np.arange(NUM_CHANNELS), inds] = chunks.min(axis=1)

    fig, ax = plt.subplots(NUM_CHANNELS, 1, sharex=True, sharey=True, figsize=(6, 20))
    for i, c_ax in enumerate(ax[::-1]):
        c_ax.plot(chunks[i], c=color_second)
        c_ax.plot(delta[i], c=color_main)
        c_ax.axis('off')

    plt.savefig(SAVE_PATH + f"{name}_delta.pdf", transparent=True)
    clear()

    shift = chunks.shape[-1] // 2 - delta[main_c].argmin()
    ret = np.roll(delta, shift, axis=1)

    fig, ax = plt.subplots(NUM_CHANNELS, 1, sharex=True, sharey=True, figsize=(6, 20))
    for i, c_ax in enumerate(ax[::-1]):
        c_ax.plot(delta[i], c=color_second)
        c_ax.plot(ret[i], c=color_main)
        c_ax.annotate(text='', xy=(delta[i].argmin(), delta[i].min() / 2), xytext=(ret[i].argmin(), ret[i].min() / 2), arrowprops=dict(arrowstyle='<-'))

        c_ax.axis('off')

    plt.savefig(SAVE_PATH + f"{name}_delta_cent.pdf", transparent=True)
    clear()


def get_delta_results(modality, chunk_size=[0], res_name='prev results/results_rf_delta'):
    results = pd.read_csv(f'../ml/{res_name}.csv', index_col=0)
    complete = results[results.restriction == 'complete']
    complete = complete[complete.modality == modality]
    complete = complete.dropna(how='all', axis=1)

    grouped_complete = complete.groupby(by=['restriction', 'modality', 'chunk_size'])

    plot_results(grouped_complete.mean(), grouped_complete.sem(), 'complete', acc=True, chunk_size=chunk_size,
                 name='delta_morph')
    clear()
    plot_results(grouped_complete.mean(), grouped_complete.sem(), 'complete', acc=False, chunk_size=chunk_size,
                 name='delta_morph')
    clear()

    d = {'spatial': SPATIAL, 'morphological': MORPHOLOGICAL, 'temporal': TEMPORAL}
    modalities = [(modality, d[modality])]

    plot_conf_mats(complete, 'complete', name='delta', chunk_size=[0], modalities=modalities,
                   data_path='../prev datasets/data_sets_delta/complete_0/morphological/0_0.800.2/')
    clear()

    plot_roc_curve(complete, name='delta', chunk_size=[0], modalities=modalities)
    clear()


def chunk_fig(clu, name, cz):
    color = PYR_COLOR if name == 'pyr' else PV_COLOR
    spikes = clu.spikes
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


def chunk_results(res_name='results_rf'):
    results = pd.read_csv(f'../ml/{res_name}.csv', index_col=0)
    complete = results[results.restriction == 'complete']

    drop = [col for col in complete.columns.values if col not in ['restriction', 'modality', 'chunk_size', 'seed',
                                                                  'acc', 'auc']]

    complete = complete.drop(columns=drop)
    grouped_complete = complete.groupby(by=['restriction', 'modality', 'chunk_size'])

    plot_acc_vs_auc(grouped_complete.mean(), grouped_complete.sem(), 'complete', name='chunks_rf')
    clear()


def compare_densities(df, feature_names):
    # palette = {"CA1_PYR": PYR_COLOR, "CA1_PV": PV_COLOR, "NEO_PYR": LIGHT_PYR, "NEO_PV": LIGHT_PV}
    palette = {"PYR": PYR_COLOR, "IN": PV_COLOR}

    df = df[df.region <= 1]
    df.region = df.region * 2
    df['label X region'] = df.label + df.region

    df.label = df.label.map({1: 'PYR', 0: 'IN'})
    df['label X region'] = df['label X region'].map({0: 'NEO_PV', 1: 'NEO_PYR', 2: 'CA1_PV', 3: 'CA1_PYR'})

    for c in df.columns:
        if c not in feature_names:
            continue
        fig, ax = plt.subplots(2, sharex=True, sharey=True)
        _ = sns.kdeplot(data=df, x=c, hue="label X region", common_norm=False, fill=True, ax=ax[0])
        _ = sns.kdeplot(data=df, x=c, hue="label", common_norm=False, fill=True, palette=palette, ax=ax[1])
        plt.savefig(SAVE_PATH + f"ncx_vs_ca1_density_{c}.pdf", transparent=True)
        clear()


def get_comp_results(chunk_size=[0], res_name='results_rf_region'):
    results = pd.read_csv(f'../ml/{res_name}.csv', index_col=0)
    complete = results[results.restriction == 'complete']

    drop = [col for col in complete.columns.values if col not in ['restriction', 'modality', 'chunk_size', 'seed',
                                                                  'acc', 'pyr_acc', 'in_acc', 'dev_acc',
                                                                  'dev_pyr_acc', 'dev_in_acc', 'auc', 'fpr', 'tpr',
                                                                  'dev_auc', 'dev_fpr', 'dev_tpr']]

    complete = complete.drop(columns=drop)

    complete = complete[complete.chunk_size.isin(chunk_size)]

    grouped_complete = complete.groupby(by=['restriction', 'modality', 'chunk_size'])

    # accuracy and AUC
    plot_test_vs_dev(grouped_complete.mean(), grouped_complete.sem(), 'complete', acc=True, name='ncx_vs_ca1_acc',
                     chunk_size=chunk_size)
    clear()
    plot_test_vs_dev(grouped_complete.mean(), grouped_complete.sem(), 'complete', acc=False, name='ncx_vs_ca1_auc',
                     chunk_size=chunk_size)
    clear()

    # confusion matrices
    plot_conf_mats(complete, 'complete', name='NCX', chunk_size=chunk_size,
                   data_path='../data_sets_region/complete_0/spatial/0_0.800.2/', use_dev=False)
    clear()
    plot_conf_mats(complete, 'complete', name='CA1', chunk_size=chunk_size,
                   data_path='../data_sets_region/complete_0/spatial/0_0.800.2/', use_dev=True)
    clear()

    # ROC curves
    plot_roc_curve(complete, name='NCX', chunk_size=chunk_size, use_dev=False)
    clear()
    plot_roc_curve(complete, name='CA1', chunk_size=chunk_size, use_dev=True)
    clear()


if __name__ == '__main__':
    import warnings

    # warnings.simplefilter("error")

    # Load wanted units
    pyr_cluster = load_cluster(TEMP_PATH, pyr_name)
    pv_cluster = load_cluster(TEMP_PATH, pv_name)

    trans_units(pyr_cluster)
    trans_units(pv_cluster)

    # Waveforms
    pyr_cluster.plot_cluster(save=True, path=SAVE_PATH)
    clear()
    pv_cluster.plot_cluster(save=True, path=SAVE_PATH)
    clear()

    # Morphological features - figure 2

    clu = pyr_cluster
    color = PYR_COLOR

    t2p_fwhm(pyr_cluster, PYR_COLOR, LIGHT_PYR, 'pyr')
    t2p_fwhm(pv_cluster, PV_COLOR, LIGHT_PV, 'pv')

    max_speed(pyr_cluster, PYR_COLOR, LIGHT_PYR, 'pyr')
    max_speed(pv_cluster, PV_COLOR, LIGHT_PV, 'pv')

    features = ['break_measure', 'fwhm', 'get_acc', 'max_speed', 'peak2peak', 'trough2peak', 'rise_coef', 'smile_cry',
                'label']
    df = load_df(features)

    corr_mat(df, 'morph')

    d_features = ['fwhm', 'trough2peak', 'max_speed']
    density_plots(df, d_features, 'morph')

    get_results('morphological', chunk_size=[0])

    # Temporal features

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

    hist_pv = ach(50, 'pv', pv_cluster)
    hist_pyr = ach(50, 'pyr', pyr_cluster)
    hist_pv_long = ach(1000, 'pv', pv_cluster)
    hist_pyr_long = ach(1000, 'pyr', pyr_cluster)

    # unif_dist

    features = ['d_kl_start', 'd_kl_mid', 'jump', 'psd_center', 'der_psd_center', 'rise_time', 'unif_dist', 'label']
    df = load_df(features)

    corr_mat(df, 'temporal')
    d_features = ['unif_dist', 'd_kl_mid']

    density_plots(df, d_features, 'temporal')

    unif_dist(hist_pv, 'pv')
    unif_dist(hist_pyr, 'pyr')

    dkl_mid(hist_pv_long, 'pv')
    dkl_mid(hist_pyr_long, 'pyr')

    get_results('temporal', chunk_size=[0])

    # modified waveforms - fig 4

    plot_delta(pyr_cluster, 'pyr')
    plot_delta(pv_cluster, 'pv')

    get_delta_results('morphological')

    # Spatial features - figure 5

    spd(pyr_cluster, 'pyr')
    spd(pv_cluster, 'pv')

    features = ['spatial_dispersion_count', 'spatial_dispersion_sd', 'spatial_dispersion_area', 'geometrical_shift',
                'geometrical_shift_sd', 'graph_avg_speed', 'graph_slowest_path', 'graph_fastest_path', 'dep_red',
                'dep_sd', 'fzc_red', 'fzc_sd', 'szc_red', 'szc_sd', 'label']
    df = load_df(features)

    corr_mat(df, 'spatial')
    d_features = ['spatial_dispersion_count', 'spatial_dispersion_area', ' fzc_red']

    density_plots(df, d_features, 'spatial')

    fzc_time_lag(pyr_cluster, 'pyr')
    fzc_time_lag(pv_cluster, 'pv')

    get_results('spatial', chunk_size=[0])

    # chunks - fig 6
    
    chunk_fig(pyr_cluster, 'pyr', 200)

    chunk_results()

    # nCX vs CA1 - fig 7
    
    feature_names = ['fwhm', 'trough2peak', 'unif_dist', 'd_kl_mid', 'spatial_dispersion_sd', 'fzc_red']
    features_df = feature_names + ['region', 'label']
    df = load_df(features_df, trans_labels=False)

    compare_densities(df, feature_names)

    get_comp_results()
