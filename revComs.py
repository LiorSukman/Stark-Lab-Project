import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from utils.upsampling import upsample_spike
from preprocessing_pipeline import load_cluster
from utils.VIS_figures import load_df, trans_units, clear
from constants import PV_COLOR, PYR_COLOR

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

TEMP_PATH_NO_LIGHT = './temp_state_minus_light/'
pyr_name = 'es25nov11_13_3_3'  # pyr

SAVE_PATH = '../../data for figures/Review/'

def com5():
    clu = load_cluster(TEMP_PATH_NO_LIGHT, pyr_name)
    trans_units(clu)
    clu = clu.np_spikes
    print(f"Number of spikes is: {len(clu)}")

    mean_clu = clu.mean(axis=0)
    main_c = np.argmax(mean_clu.max(axis=1) - mean_clu.min(axis=1))

    """plt.plot(mean_clu[main_c])
    plt.ylim(-200, 85)
    plt.savefig(SAVE_PATH + f"R1Aa.pdf", transparent=True)
    clear()"""

    mean_up_clu = upsample_spike(mean_clu)[main_c]

    plt.plot(mean_up_clu)
    plt.ylim(-200, 85)
    plt.savefig(SAVE_PATH + f"C5A.pdf", transparent=True)
    clear()

    inds = np.arange(len(clu))
    np.random.shuffle(inds)
    clu = clu[inds]

    up_clu = np.array([upsample_spike(spike)[main_c] for spike in clu])

    """for i in range(2):
        plt.plot(clu[i][main_c])
        plt.ylim(-200, 85)
        plt.savefig(SAVE_PATH + f"R1Ba{i}.pdf", transparent=True)
        clear()

        plt.plot(up_clu[i])
        plt.ylim(-200, 85)
        plt.savefig(SAVE_PATH + f"R1Bb{i}.pdf", transparent=True)
        clear()"""

    up_mean_clu = up_clu.mean(axis=0)
    plt.plot(up_mean_clu, c='r')
    plt.ylim(-200, 85)
    plt.savefig(SAVE_PATH + f"C5B.pdf", transparent=True)
    clear()

    plt.plot(mean_up_clu)
    plt.plot(up_mean_clu, c='r')
    plt.ylim(-200, 85)
    plt.savefig(SAVE_PATH + f"C5C.pdf", transparent=True)
    clear()

def com8():
    print('Chunk size = 0')
    df = load_df(['spatial_dispersion_count', 'label'], path='clusterData_spd_020922/0')
    df_np = df.to_numpy()
    spd, label = df_np[:, 0], df_np[:, 1]
    removed = 8 - spd
    removed_pyr = removed[label == 'PYR']
    removed_pv = removed[label == 'PV']
    print('Removed for PYR [Q25, Q50, Q75]:', np.quantile(removed_pyr, q=[0.25, 0.5, 0.75]))
    print('Removed for PV [Q25, Q50, Q75]:', np.quantile(removed_pv, q=[0.25, 0.5, 0.75]))
    print('Mann-Whitney test:', stats.mannwhitneyu(removed_pyr, removed_pv).pvalue)

    print('\nChunk size = 25')
    df = load_df(['spatial_dispersion_count', 'label'], path='clusterData_spd_020922/25')
    df_np = df.to_numpy()
    spd, label = df_np[:, 0], df_np[:, 1]
    removed = 8 - spd
    removed_pyr = removed[label == 'PYR']
    removed_pv = removed[label == 'PV']
    print('Removed for PYR [Q25, Q50, Q75]:', np.quantile(removed_pyr, q=[0.25, 0.5, 0.75]))
    print('Removed for PV [Q25, Q50, Q75]:', np.quantile(removed_pv, q=[0.25, 0.5, 0.75]))
    print('Mann-Whitney test:', stats.mannwhitneyu(removed_pyr, removed_pv).pvalue)

def com19(d_fets, y_axis, name, conv_func):
    df = load_df(None, path='cluster_data/clusterData_no_light_29_03_22/0')

    conversion = conv_func(1.6 / 256)
    labels = df.label.to_numpy()
    fmc = df[d_fets['FMC']].to_numpy() * conversion
    smc = df[d_fets['SMC']].to_numpy() * conversion
    neg = df[d_fets['NEG']].to_numpy() * conversion
    fig, ax = plt.subplots()
    xs = np.asarray([1, 2, 3])
    ax.set_ylabel(y_axis)
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
        counter += 1

    plt.savefig(SAVE_PATH + f"{name}_graph.pdf", transparent=True)
    clear()


if __name__ == '__main__':
    com19({'FMC': 'fzc_red', 'NEG': 'dep_red', 'SMC': 'szc_red'}, 'ms^2', 'time_lag_red', lambda x: x ** 2)
    com19({'FMC': 'fzc_graph_slowest_path', 'NEG': 'dep_graph_slowest_path', 'SMC': 'szc_graph_slowest_path'},
          'mm/s', 'shortest_path', lambda x: 1 / x)

