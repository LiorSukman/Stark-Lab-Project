import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import ML_util

from constants import SPATIAL, MORPHOLOGICAL, TEMPORAL, SPAT_TEMPO, STARK_SPAT, STARK_SPAT_TEMPO, STARK
from constants import WIDTH, T2P, WIDTH_SPAT, T2P_SPAT
from constants import TRANS_MORPH
from constants import feature_names as fet_names

chunks = [0, 500, 200]
restrictions = ['complete', 'no_small_sample']
modalities = ['spatial', 'morphological', 'temporal', 'spat_tempo']
NUM_FETS = 29
SAVE_PATH = '../../../data for figures/'


def change_length(lst, ref, length):
    new_thrs = np.linspace(0, 1, length)
    ref_ind = 0
    new_lst = []
    i = 0
    while i < length:
        new_thr = new_thrs[i]
        if ref[ref_ind] == new_thr:
            new_lst.append(lst[ref_ind])
            i += 1
        elif ref_ind < len(ref) and ref[ref_ind] < new_thr < ref[ref_ind + 1]:
            alpha = (ref[ref_ind + 1] - new_thr) / (ref[ref_ind + 1] - ref[ref_ind])
            new_val = alpha * lst[ref_ind] + (1 - alpha) * lst[ref_ind + 1]
            new_lst.append(new_val)
            i += 1
        else:
            ref_ind += 1
    return np.array(new_lst)


def str2lst(str):
    ret = []
    for val in list(str[1:-1].split(" ")):  # remove [] and split
        if len(val) == 0:
            continue
        if val[-1] == '.':  # for 0. case
            val = val[:-1]
        ret.append(float(val))
    return ret


def plot_roc_curve(df, name=None, chunk_size=[0], modalities=None):
    if modalities is None:
        modalities = [('spatial', SPATIAL), ('temporal', TEMPORAL), ('morphological', MORPHOLOGICAL)]

    for m_name, _ in modalities:
        df_m = df[df.modality == m_name]
        fig, ax = plt.subplots(len(chunk_size))
        if len(chunk_size) == 1:
            ax = [ax]
        for cz, cz_ax in zip(chunk_size, ax):
            df_cz = df_m[df_m.chunk_size == cz]
            fprs = [str2lst(lst) for lst in df_cz.fpr]
            tprs = [str2lst(lst) for lst in df_cz.tpr]

            tprs = np.array([change_length(lst, fpr, 50) for lst, fpr in zip(tprs, fprs)])

            mean_fprs = np.linspace(0, 1, 50)
            mean_tprs = tprs.mean(axis=0)
            std = tprs.std(axis=0)

            cz_ax.plot(mean_fprs, mean_tprs)
            cz_ax.fill_between(mean_fprs, mean_tprs - std, mean_tprs + std, alpha=0.2)

        if name is None:
            plt.show()
        else:
            plt.savefig(SAVE_PATH + f"{name}_{m_name}_conf_mat.pdf", transparent=True)


def plot_cf(tp, tn, fp, fn, tp_std, tn_std, fp_std, fn_std, cz_ax, cz, v_max):
    data = np.array([[tp, fn], [fp, tn]])
    data = data / data.sum(axis=1)
    pn_sign = u"\u00B1"
    labels = [[f"{tp:2g}{pn_sign}{tp_std:3f}", f"{fn:2g}{pn_sign}{fn_std:3f}"],
              [f"{fp:2g}{pn_sign}{fp_std:3f}", f"{tn:2g}{pn_sign}{tn_std:3f}"]]
    cmap = sns.light_palette("seagreen", as_cmap=True)
    ticklabels = ['PYR', 'IN']
    _ = sns.heatmap(data, annot=labels, fmt='', vmin=0, vmax=1, cmap=cmap, ax=cz_ax, xticklabels=ticklabels,
                    yticklabels=ticklabels)
    cz_ax.set_title(f"Chunk size={cz}")


def plot_conf_mats(df, restriction, name=None, chunk_size=[0], modalities=None):
    data_path = f'../data_sets_new/{restriction}_0/spatial/0_0.800.2/'
    _, _, test, _, _, _ = ML_util.get_dataset(data_path)
    test = np.squeeze(test)
    positive, negative = np.sum(test[:, -1] == 1), np.sum(test[:, -1] == 0)
    if modalities is None:
        modalities = [('spatial', SPATIAL), ('temporal', TEMPORAL), ('morphological', MORPHOLOGICAL)]

    tpp = df.pyr_acc * 0.01
    tnp = df.in_acc * 0.01

    df['tp'] = positive * tpp
    df['tn'] = negative * tnp
    df['fp'] = negative * (1 - tnp)
    df['fn'] = positive * (1 - tpp)

    grouped = complete.groupby(by=['restriction', 'modality', 'chunk_size'])
    mean = grouped.mean()
    std = grouped.std()

    for m_name, _ in modalities:
        mean_m = mean.xs(m_name, level="modality")
        std_m = std.xs(m_name, level="modality")
        fig, ax = plt.subplots(len(chunk_size))
        if len(chunk_size) == 1:
            ax = [ax]
        for cz, cz_ax in zip(chunk_size, ax):
            mean_cz = mean_m.xs(cz, level="chunk_size")
            std_cz = std_m.xs(cz, level="chunk_size")
            tp, tn, fp, fn = mean_cz.tp[0], mean_cz.tn[0], mean_cz.fp[0], mean_cz.fn[0]
            tp_std, tn_std, fp_std, fn_std = std_cz.tp[0], std_cz.tn[0], std_cz.fp[0], std_cz.fn[0]
            plot_cf(tp, tn, fp, fn, tp_std, tn_std, fp_std, fn_std, cz_ax, cz, max(positive, negative))
        if name is None:
            plt.show()
        else:
            plt.savefig(SAVE_PATH + f"{name}_{m_name}_conf_mat.pdf", transparent=True)


def get_title(restriction):
    seeds = np.arange(20)
    tot, tst, pyr, intn = [], [], [], []
    for seed in seeds:
        data_path = f"../data_sets_new/{restriction}_{seed}/spatial/0_0.800.2/"
        train, dev, test, _, _, _ = ML_util.get_dataset(data_path)
        tot.append(len(train) + len(dev) + len(test))
        tst.append(len(test))
        pyr.append(len([cell for cell in test if cell[0][-1] == 1]))
        intn.append(len([cell for cell in test if cell[0][-1] == 0]))

    return f"Scores by modality and chunk size; Total={np.mean(tot)}+-{np.std(tot)}, Test={np.mean(tst)}+-" \
           f"{np.std(tst)}, PYR={np.mean(pyr)}+-{np.std(pyr)}, IN={np.mean(intn)}+-{np.std(intn)} "


def autolabel(rects, ax, acc):
    """
    Attach a text label above each bar in *rects*, displaying its height.
    This function was taken from https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    """
    for rect in rects:
        height = rect.get_height()
        if height == 0:
            continue
        ax.annotate(f"{round(height, 2 if acc else 3)}{'%' if acc else ''}",
                    # ax.annotate('{}'.format(round(height, 3)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 11),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def get_labels(lst):
    ret = []
    for element in lst:
        if element not in ret:
            ret.append(element)
    return ret


def plot_fet_imp(df, sems, restriction, name=None, chunk_size=None, modalities=None):
    # title = get_title(restriction)
    if modalities is None:
        modalities = [('spatial', SPATIAL), ('temporal', TEMPORAL), ('spat_tempo', SPAT_TEMPO),
                      ('morphological', MORPHOLOGICAL)]
    fets_org = [f"feature {f + 1}" for f in range(NUM_FETS)]
    map = {f: name for (f, name) in zip(fets_org, fet_names)}
    df = df.rename(map, axis='columns')
    df = df.drop(columns=['seed', 'acc', 'auc', 'pyr_acc', 'in_acc'])
    sems = sems.rename(map, axis='columns')
    sems = sems.drop(columns=['seed', 'acc', 'auc', 'pyr_acc', 'in_acc'])
    for m_name, m_places in modalities:
        df_m = np.asarray(df.loc[:, m_name, :].dropna(axis=1))
        sems_m = np.asarray(sems.loc[:, m_name, :].dropna(axis=1))
        names_m = np.asarray(fet_names)[m_places[:-1]]

        order = np.argsort((-1 * df_m).mean(axis=0))
        df_m = df_m[:, order]
        sems_m = sems_m[:, order]
        names_m = names_m[order]

        x = np.arange(len(names_m))  # the label locations

        if chunk_size is None:
            v0, e0 = df_m[0], sems_m[0]
            v200, e200 = df_m[1], sems_m[1]
            v500, e500 = df_m[2], sems_m[2]
        else:
            d = {0: 0, 200: 1, 500: 2}
            v, e = df_m[d[chunk_size]], sems_m[d[chunk_size]]
            order = np.argsort((-1 * v))
            v = v[order]
            e = e[order]
            names_m = names_m[order]

        width = 0.3  # the width of the bars

        fig, ax = plt.subplots(figsize=(6, 12))
        if chunk_size is None:
            _ = ax.bar(x - width, v0, width - 0.02, label='chunk_size = 0', yerr=e0, color='#404040',
                       edgecolor='k')
            _ = ax.bar(x, v500, width - 0.02, label='chunk_size = 500', yerr=e500, color='#808080',
                       edgecolor='k')
            _ = ax.bar(x + width, v200, width - 0.02, label='chunk_size = 200', yerr=e200,
                       color='#ffffff', edgecolor='k')
        else:
            _ = ax.bar(x, v, width - 0.02, yerr=e, color='#404040', edgecolor='k')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Importance')
        # ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(names_m, rotation=-25)
        ax.legend()

        fig.tight_layout()

        plt.show()
        # plt.savefig(SAVE_PATH + f"{name}_fet_imp.pdf", transparent=True)


def plot_results(df, sems, restriction, acc=True, name=None, chunk_size=None):
    if name is None:
        title = get_title(restriction)

    labels = get_labels([ind[1] for ind in df.index])

    if acc:
        if chunk_size is not None:
            val = df.xs(chunk_size, level="chunk_size").acc
            sem = sems.xs(chunk_size, level="chunk_size").acc
        else:
            zero = df.xs(0, level="chunk_size").acc
            five_hundred = df.xs(500, level="chunk_size").acc
            two_hundred = df.xs(200, level="chunk_size").acc
            zero_sem = sems.xs(0, level="chunk_size").acc
            five_hundred_sem = sems.xs(500, level="chunk_size").acc
            two_hundred_sem = sems.xs(200, level="chunk_size").acc
    else:
        if chunk_size is not None:
            val = df.xs(chunk_size, level="chunk_size").auc
            sem = sems.xs(chunk_size, level="chunk_size").auc
        else:
            zero = df.xs(0, level="chunk_size").auc
            five_hundred = df.xs(500, level="chunk_size").auc
            two_hundred = df.xs(200, level="chunk_size").auc
            zero_sem = sems.xs(0, level="chunk_size").auc
            five_hundred_sem = sems.xs(500, level="chunk_size").auc
            two_hundred_sem = sems.xs(200, level="chunk_size").auc

    x = np.arange(len(labels))  # the label locations
    width = 0.3  # the width of the bars

    fig, ax = plt.subplots(figsize=(6, 12))
    if chunk_size is None:
        rects1 = ax.bar(x - width, zero, width - 0.02, label='chunk_size = 0', yerr=zero_sem, color='#404040',
                        edgecolor='k')
        rects2 = ax.bar(x, five_hundred, width - 0.02, label='chunk_size = 500', yerr=five_hundred_sem, color='#808080',
                        edgecolor='k')
        rects3 = ax.bar(x + width, two_hundred, width - 0.02, label='chunk_size = 200', yerr=two_hundred_sem,
                        color='#ffffff', edgecolor='k')
        autolabel(rects1, ax, acc)
        autolabel(rects2, ax, acc)
        autolabel(rects3, ax, acc)
    else:
        rects = ax.bar(x, val, width - 0.02, yerr=sem, color='#404040', edgecolor='k')
        autolabel(rects, ax, acc)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores (percentage)')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()
    plt.show()
    # plt.savefig(SAVE_PATH + f"{name}_{'acc' if acc else 'auc'}.pdf", transparent=True)


if __name__ == "__main__":
    """
    checks:
    1) Is the name of the model correct?
    2) Are we passing all the modalities we wish to plot?
    3) Have we the correct path for the confusion matrix?
    """
    model = 'rf'
    results = pd.read_csv(f'git_results_{model}.csv', index_col=0)
    complete = results[results.restriction == 'complete']
    # complete = complete[complete.modality == 'spatial']
    no_small_sample = results[results.restriction == 'no_small_sample']
    grouped_complete = complete.groupby(by=['restriction', 'modality', 'chunk_size'])
    grouped_no_small_sample = no_small_sample.groupby(by=['restriction', 'modality', 'chunk_size'])
    # plot_conf_mats(complete, 'complete')
    plot_roc_curve(complete)
    exit(0)

    plot_results(grouped_complete.mean(), grouped_complete.sem(), 'complete', acc=True)
    plot_results(grouped_complete.mean(), grouped_complete.sem(), 'complete', acc=False)
    plot_fet_imp(grouped_complete.mean(), grouped_complete.sem(), 'complete')
    exit(0)
    plot_results(grouped_no_small_sample.mean(), grouped_no_small_sample.sem(), 'no_small_sample', acc=True)
    plot_results(grouped_no_small_sample.mean(), grouped_no_small_sample.sem(), 'no_small_sample', acc=False)
