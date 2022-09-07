import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from utils.upsampling import upsample_spike
from preprocessing_pipeline import load_cluster
from utils.VIS_figures import load_df, trans_units, clear

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

    plt.plot(mean_clu[main_c])
    plt.ylim(-200, 85)
    plt.savefig(SAVE_PATH + f"R1Aa.pdf", transparent=True)
    clear()

    mean_up_clu = upsample_spike(mean_clu)[main_c]

    plt.plot(mean_up_clu)
    plt.ylim(-200, 85)
    plt.savefig(SAVE_PATH + f"R1Ab.pdf", transparent=True)
    clear()

    inds = np.arange(len(clu))
    np.random.shuffle(inds)
    clu = clu[inds]

    up_clu = np.array([upsample_spike(spike)[main_c] for spike in clu])

    for i in range(2):
        plt.plot(clu[i][main_c])
        plt.ylim(-200, 85)
        plt.savefig(SAVE_PATH + f"R1Ba{i}.pdf", transparent=True)
        clear()

        plt.plot(up_clu[i])
        plt.ylim(-200, 85)
        plt.savefig(SAVE_PATH + f"R1Bb{i}.pdf", transparent=True)
        clear()

    up_mean_clu = up_clu.mean(axis=0)

    plt.plot(mean_up_clu)
    plt.plot(up_mean_clu + 25)
    plt.savefig(SAVE_PATH + f"R1C.pdf", transparent=True)
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


if __name__ == '__main__':
    com5()
