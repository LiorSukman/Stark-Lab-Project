import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import interpolate
from sklearn.metrics import confusion_matrix
import ml.ML_util as ML_util
from utils.hideen_prints import HiddenPrints

from constants import SPATIAL, MORPHOLOGICAL, TEMPORAL
from constants import feature_names_org as FET_NAMES

import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

restrictions = ['complete', 'no_small_sample']
modalities = ['spatial', 'morphological', 'temporal']
NUM_FETS = 34
SAVE_PATH = '../../../data for figures/Thesis/'


def change_length(x, y, length):
    valid = []
    for i in range(len(x)):
        if i == len(x) - 1:
            valid.append(i)
        elif x[i] != x[i + 1]:
            valid.append(i)
        else:
            continue

    x = x[valid]
    y = y[valid]

    xnew = np.linspace(0, 1, length)
    f = interpolate.interp1d(x, y, kind='linear')
    ynew = [0] + list(f(xnew))
    return np.array(ynew)


def str2lst(str):
    ret = []
    for val in list(str[1:-1].split(" ")):  # remove [] and split
        if len(val) == 0:
            continue
        if val[-1] == '.':  # for 0. case
            val = val[:-1]
        ret.append(float(val))
    return np.array(ret)


def plot_roc_curve(df, name=None, chunk_size=[200], modalities=None, use_dev=False):
    if modalities is None:
        modalities = [('spatial', SPATIAL), ('temporal', TEMPORAL), ('morphological', MORPHOLOGICAL)]

    for m_name, _ in modalities:
        df_m = df[df.modality == m_name]
        fig, ax = plt.subplots()
        for cz in chunk_size:
            df_cz = df_m[df_m.chunk_size == cz]

            fpr_col = 'fpr' if not use_dev else 'dev_fpr'
            tpr_col = 'tpr' if not use_dev else 'dev_tpr'

            fprs = [str2lst(lst) for lst in df_cz[fpr_col]]
            tprs = [str2lst(lst) for lst in df_cz[tpr_col]]

            x_length = 100

            tprs = np.array([change_length(fpr, tpr, x_length) for tpr, fpr in zip(tprs, fprs)])

            mean_fprs = [0] + list(np.linspace(0, 1, x_length))
            med_tprs = np.median(tprs, axis=0)
            iqr25 = np.quantile(tprs, 0.25, axis=0)
            iqr75 = np.quantile(tprs, 0.75, axis=0)

            auc_col = 'auc' if not use_dev else 'dev_auc'

            auc = np.median(df_cz[auc_col])

            label_add = f", chunk size={cz}" if len(chunk_size) > 1 else ""

            ax.plot(mean_fprs, med_tprs, label=f"AUC={auc:.3f}" + label_add)
            ax.fill_between(mean_fprs, iqr25, iqr75, alpha=0.2)

        ax.plot(mean_fprs, mean_fprs, color='k', linestyle='--', label='chance level')
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.legend()
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")

        if name is None:
            plt.title(f"{m_name} chunk sizes: {chunk_size}")
            plt.show()
        else:
            plt.savefig(SAVE_PATH + f"{name}_{m_name}_roc_curve.pdf", transparent=True)


def plot_cf_thr(preds, thr, labels, c_ax):
    preds_thr = preds >= thr
    confs_thr = [confusion_matrix(labels, pred) for pred in preds_thr]
    conf_thr = np.roll(np.roll(np.median(confs_thr, axis=0), 1, axis=0), 1, axis=1)

    data = conf_thr / np.expand_dims(conf_thr.sum(axis=1), axis=1)
    labels = [[f"{conf_thr[0, 0]:2g}", f"{conf_thr[0, 1]:2g}"],
              [f"{conf_thr[1, 0]:2g}", f"{conf_thr[1, 1]:2g}"]]
    cmap = sns.light_palette("seagreen", as_cmap=True)
    ticklabels = ['PYR', 'PV']
    _ = sns.heatmap(data, annot=labels, fmt='', vmin=0, vmax=1, cmap=cmap, ax=c_ax, xticklabels=ticklabels,
                    yticklabels=ticklabels)
    cbar = c_ax.collections[0].colorbar
    cbar.set_ticks([0, .5, 1])
    cbar.set_ticklabels(['0%', '50%', '100%'])
    c_ax.set_ylabel("True Label")
    c_ax.set_xlabel('Predicted Label')

    return c_ax


def plot_cf(tp, tn, fp, fn, tp_q25, tn_q25, fp_q25, fn_q25, tp_q75, tn_q75, fp_q75, fn_q75, cz_ax, cz):
    data = np.array([[tp, fn], [fp, tn]])
    data = data / np.expand_dims(data.sum(axis=1), axis=1)

    labels = [[f"{tp:2g} [{tp_q25:.2f} {tp_q75:.2f}]", f"{fn:2g} [{fn_q25:.2f} {fn_q75:.2f}]"],
              [f"{fp:2g} [{fp_q25:.2f} {fp_q75:.2f}]", f"{tn:2g} [{tn_q25:.2f} {tn_q75:.2f}]"]]
    cmap = sns.light_palette("seagreen", as_cmap=True)
    ticklabels = ['PYR', 'PV']
    _ = sns.heatmap(data, annot=labels, fmt='', vmin=0, vmax=1, cmap=cmap, ax=cz_ax, xticklabels=ticklabels,
                    yticklabels=ticklabels)
    cbar = cz_ax.collections[0].colorbar
    cbar.set_ticks([0, .25, .5, .75, 1])
    cbar.set_ticklabels(['0%', '25%', '50%', '75%', '100%'])
    cz_ax.set_ylabel("True Label")
    cz_ax.set_xlabel('Predicted Label')


def plot_conf_mats(df, restriction, name=None, chunk_size=[0], modalities=None, use_dev=False, data_path=None):
    df_temp = df.copy()
    if data_path is None:
        raise Warning('Using default dataset path in plot_conf_mats')
        data_path = f'../data_sets_290322/{restriction}_0/spatial/0_0.800.2/'
    with HiddenPrints():
        _, dev, test, _, _, _ = ML_util.get_dataset(data_path)
    test = np.squeeze(dev) if use_dev else np.squeeze(test)
    positive, negative = np.sum(test[:, -1] == 1), np.sum(test[:, -1] == 0)
    if modalities is None:
        modalities = [('spatial', SPATIAL), ('temporal', TEMPORAL), ('morphological', MORPHOLOGICAL)]

    if not use_dev:
        tpp = df_temp.pyr_acc * 0.01
        tnp = df_temp.in_acc * 0.01
    else:
        tpp = df_temp.dev_pyr_acc * 0.01
        tnp = df_temp.dev_in_acc * 0.01

    df_temp['tp'] = positive * tpp
    df_temp['tn'] = negative * tnp
    df_temp['fp'] = negative * (1 - tnp)
    df_temp['fn'] = positive * (1 - tpp)

    grouped = df_temp.groupby(by=['restriction', 'modality', 'chunk_size'])
    med = grouped.median()
    q25 = grouped.quantile(0.25)
    q75 = grouped.quantile(0.75)

    for m_name, _ in modalities:
        med_m = med.xs(m_name, level="modality")
        q25_m = q25.xs(m_name, level="modality")
        q75_m = q75.xs(m_name, level="modality")
        fig, ax = plt.subplots(len(chunk_size), sharex=True, sharey=True)
        if len(chunk_size) == 1:
            ax = [ax]
        for cz, cz_ax in zip(chunk_size, ax):
            med_cz = med_m.xs(cz, level="chunk_size")
            q25_cz = q25_m.xs(cz, level="chunk_size")
            q75_cz = q75_m.xs(cz, level="chunk_size")
            tp, tn, fp, fn = med_cz.tp[0], med_cz.tn[0], med_cz.fp[0], med_cz.fn[0]
            tp_q25, tn_q25, fp_q25, fn_q25 = q25_cz.tp[0], q25_cz.tn[0], q25_cz.fp[0], q25_cz.fn[0]
            tp_q75, tn_q75, fp_q75, fn_q75 = q75_cz.tp[0], q75_cz.tn[0], q75_cz.fp[0], q75_cz.fn[0]
            plot_cf(tp, tn, fp, fn, tp_q25, tn_q25, fp_q25, fn_q25, tp_q75, tn_q75, fp_q75, fn_q75, cz_ax, cz)
            if len(chunk_size) > 1:
                cz_ax.set_title(f"Chunk size={cz}")
        if name is None:
            plt.show()
        else:
            plt.savefig(SAVE_PATH + f"{name}_{m_name}_conf_mat.pdf", transparent=True)


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
                    ha='center', va='bottom', size=10)


def get_labels(lst):
    ret = []
    for element in lst:
        if element not in ret:
            ret.append(element)
    return ret


def plot_fet_imp(df, sems, restriction, base, name=None, chunk_size=None, modalities=None, semsn=None, fet_names_map=None):
    # TODO make sure order is determined only by rows of used chunk_size
    if name is None:
        title = ""
    if modalities is None:
        modalities = [('spatial', SPATIAL), ('temporal', TEMPORAL), ('morphological', MORPHOLOGICAL)]

    if fet_names_map is None:
        fet_names_map = FET_NAMES

    if chunk_size is None:
        chunk_size = get_labels([ind[2] for ind in df.index])

    fets_org = [f"test feature {f + 1}" for f in range(NUM_FETS)]
    rem = [f"dev feature {f + 1}" for f in range(NUM_FETS) if f"dev feature {f + 1}" in df.columns]
    df = df.drop(columns=rem)
    sems = sems.drop(columns=rem)
    base = base.drop(columns=rem)
    if semsn is not None:
        semsn = semsn.drop(columns=rem)

    map = {f: name for (f, name) in zip(fets_org, fet_names_map)}
    df = df.rename(map, axis='columns')
    df = df.drop(columns=['seed', 'acc', 'auc', 'pyr_acc', 'in_acc', 'f1', 'mcc'], errors='ignore')
    df = df.drop(columns=['dev_acc', 'dev_auc', 'dev_pyr_acc', 'dev_in_acc', 'dev_f1', 'dev_mcc'], errors='ignore')
    sems = sems.rename(map, axis='columns')
    sems = sems.drop(columns=['seed', 'acc', 'auc', 'pyr_acc', 'in_acc', 'f1', 'mcc'], errors='ignore')
    sems = sems.drop(columns=['dev_acc', 'dev_auc', 'dev_pyr_acc', 'dev_in_acc', 'dev_f1', 'dev_mcc'], errors='ignore')
    base = base.rename(map, axis='columns')
    base = base.drop(columns=['seed', 'acc', 'auc', 'pyr_acc', 'in_acc', 'f1', 'mcc'], errors='ignore')
    base = base.drop(columns=['dev_acc', 'dev_auc', 'dev_pyr_acc', 'dev_in_acc', 'dev_f1', 'dev_mcc'], errors='ignore')

    if semsn is not None:
        semsn = semsn.rename(map, axis='columns')
        semsn = semsn.drop(columns=['seed', 'acc', 'auc', 'pyr_acc', 'in_acc', 'f1', 'mcc'], errors='ignore')
        semsn = semsn.drop(columns=['dev_acc', 'dev_auc', 'dev_pyr_acc', 'dev_in_acc', 'dev_f1', 'dev_mcc'], errors='ignore')

    for m_name, m_places in modalities:
        df_m = df.xs(m_name, level="modality").dropna(axis=1)
        sems_m = sems.xs(m_name, level="modality").dropna(axis=1)
        base_m = base.xs(m_name, level="modality").dropna(axis=1)
        semsn_m = None
        if semsn is not None:
            semsn_m = semsn.xs(m_name, level="modality").dropna(axis=1)

        names_m = np.asarray(fet_names_map)[m_places[:-1]]

        df_order = np.asarray(df_m)
        order = np.argsort((-1 * df_order).mean(axis=0))

        names_m = names_m[order]

        x = np.arange(len(names_m))  # the label locations

        width = 0.75  # 1 / (len(chunk_size) + 1)  # the width of the bars

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = ['#FFFFFF', '#E0E0E0', '#C0C0C0', '#A0A0A0', '#808080', '#696969'][::6 // (len(chunk_size) * 2)][::-1]

        for i, cz in enumerate(chunk_size):
            df_cz = np.asarray(df_m.xs(cz, level="chunk_size"))[:, order].flatten()
            sems_cz = np.asarray(sems_m.xs(cz, level="chunk_size"))[:, order].flatten()
            base_cz = np.asarray(base_m.xs(cz, level="chunk_size"))[:, order].flatten()
            if semsn_m is not None:
                semsn_cz = 2 * df_cz - np.asarray(semsn_m.xs(cz, level="chunk_size"))[:, order].flatten()
                sems_cz = np.concatenate((np.expand_dims(semsn_cz, axis=0), np.expand_dims(sems_cz, axis=0))) - df_cz
            else:
                sems_cz = np.concatenate((np.expand_dims(sems_cz, axis=0), np.expand_dims(sems_cz, axis=0)))

            ax.bar(x - (i - len(chunk_size) // 2) * width, df_cz, width, label=f'chunk_size = {cz}', yerr=sems_cz,
                   color=colors[i])
            ax.hlines(base_cz, x - (i - len(chunk_size) // 2) * width - (width / 2),
                      x - (i - len(chunk_size) // 2) * width + (width / 2), linestyles='dashed', color='k')
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Importance (SHAP value)')
        # ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(names_m, rotation=-90)

        if len(chunk_size) > 1:
            ax.legend()

        fig.tight_layout()

        if name is None:
            plt.suptitle(title)
            plt.show()
        else:
            plt.savefig(SAVE_PATH + f"{name}_fet_imp.pdf", transparent=True)


def plot_auc_chunks(df, sems, restriction, name=None, semsn=None):
    if name is None:
        title = ''

    if semsn is None:
        semsn = sems

    labels = get_labels([ind[1] for ind in df.index])

    chunk_size = get_labels([ind[2] for ind in df.index])

    x = np.roll(np.array(chunk_size), -1)
    x[-1] = x[-2] * 2
    for m_name in labels:
        fig, ax = plt.subplots(figsize=(9, 6))

        val = np.roll(np.asarray(df.xs(m_name, level="modality")['auc']), -1)
        sem = np.expand_dims(np.roll(np.asarray(sems.xs(m_name, level="modality")['auc']), -1), axis=0)
        semn = np.expand_dims(np.roll(np.asarray(semsn.xs(m_name, level="modality")['auc']), -1), axis=0)
        yerr = np.concatenate((val - semn, sem - val), axis=0)
        ax.bar(np.arange(len(x) - 1), val[:-1], yerr=yerr[:, :-1], color='#A0A0A0', width=0.75)
        ax.axhline(val[-1], xmin=-0.75 / 2, xmax=4 + 0.75 / 2, color='k', linestyle='--')
        x_labels = [str(int(xt)) for xt in x][:-1]
        ax.set_xticks(np.arange(len(x) - 1))
        ax.set_xticklabels(labels=x_labels)
        ax.set_ylabel('AUC')
        ax.set_xlabel('Chunk Size')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.tight_layout()
        ax.set_ylim(ymin=val.min() * .9, ymax=1)

        if name is None:
            plt.suptitle(title)
            plt.show()
        else:
            plt.savefig(SAVE_PATH + f"{name}_{m_name}_auc.pdf", transparent=True)
            plt.clf()
            plt.cla()
            plt.close('all')


def plot_auc_chunks_bp(df, name=None, plot=True, ax_inp=None, edge_color='k', shift=0):
    # chunk_sizes = df.chunk_size.unique()[1::][::-1] # remove no chunking
    chunk_sizes = np.roll(df.chunk_size.unique(), -1)[::-1]
    mods = df.modality.unique()

    for m_name in mods:
        df_m = df[df.modality == m_name]
        chunk_sizes_m = df_m.chunk_size.to_numpy()
        chunk_aucs = [df_m.auc[chunk_sizes_m == cs].to_numpy() for cs in chunk_sizes]

        if ax_inp is None:
            fig, ax = plt.subplots(figsize=(9, 6))
        else:
            ax = ax_inp

        ax.axhline(y=np.median(df_m.auc[chunk_sizes_m == 0].to_numpy()), color='k', linestyle='--')
        # Since highlighting is a bit to much
        # ax.axhspan(ymin=np.quantile(df_m.auc[chunk_sizes_m == 0].to_numpy(), 0.25),
        #            ymax=np.quantile(df_m.auc[chunk_sizes_m == 0].to_numpy(), 0.75), color='k', alpha=0.2)
        bp = ax.boxplot(chunk_aucs, labels=chunk_sizes.astype(np.int32), positions=2.5 * np.arange(len(chunk_sizes)) + shift,
                   flierprops=dict(markeredgecolor=edge_color, marker='+'), notch=True, bootstrap=1_000)

        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color=edge_color)

        ax.set_xticks(2.5 * np.arange(len(chunk_sizes)))

        ax.set_ylabel('AUC')
        ax.set_xlabel('Chunk Size')
        ax.set_ylim(top=1.01)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if not plot:
            return ax

        if name is None:
            plt.show()
        else:
            plt.savefig(SAVE_PATH + f"{name}_{m_name}_auc.pdf", transparent=True)
            plt.clf()
            plt.cla()
            plt.close('all')


def plot_test_vs_dev_bp(df, chunk_sizes=(0, 0, 0), name=None, diff=False):
    labels = ['morphological', 'temporal', 'spatial']
    fig, ax = plt.subplots(figsize=(6, 7.5))  # 8.5 10 for FENS

    x = np.arange(len(labels)) + 1  # the label locations

    for i, (mod, cz) in enumerate(zip(labels, chunk_sizes)):
        filter1 = df.chunk_size == cz
        filter2 = df.modality == mod

        val = df[filter1 & filter2].auc.to_numpy()
        dev_val = df[filter1 & filter2].dev_auc.to_numpy()

        if diff:
            diff_vals = 100 * (dev_val - val) / dev_val
            ax.boxplot(diff_vals, positions=[x[i]],
                       flierprops=dict(markerfacecolor='#808080', marker='+'), medianprops={"color": "k"},
                       patch_artist=True, widths=0.4, notch=True, bootstrap=1_000)
        else:
            ax.boxplot(val, positions=[x[i]+0.25],
                       flierprops=dict(markerfacecolor='#808080', marker='+'), medianprops={"color": "k"},
                       patch_artist=True, widths=0.4, notch=True, bootstrap=1_000)
            ax.boxplot(dev_val, positions=[x[i]-0.25],
                       flierprops=dict(markerfacecolor='#808080', marker='+'), medianprops={"color": "k"},
                       patch_artist=True, widths=0.4, notch=True, bootstrap=1_000)

    if diff:
        ticks = x
    else:
        ticks = []
        for t in x:
            ticks += [t - 0.25, t + 0.25]
    ax.set_xticks(ticks)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if name is None:
        plt.show()
    else:
        diff_str = "diff" if diff else ""
        plt.savefig(SAVE_PATH + f"{name}{diff_str}.pdf", transparent=True)


def plot_test_vs_dev(df, sems, restriction, acc=True, name=None, chunk_size=None, mode='bar', semsn=None):
    if name is None:
        title = 'dev-test comparison'

    labels = get_labels([ind[1] for ind in df.index])

    if semsn is None:
        semsn = sems

    if chunk_size is None:
        chunk_size = (0, 0, 0)  # get_labels([ind[2] for ind in df.index])

    fig, ax = plt.subplots()

    x = np.arange(len(labels))  # the label locations
    width = 1 / 3  # the width of the bars

    if len(chunk_size) * 2 > 6 and mode == 'bar':
        raise AssertionError('not enough colors in plot_test_vs_dev function to plot the required graph')
    colors = ['#A0A0A0', '#E0E0E0']

    if mode == 'bar':
        for i, (mod, cz) in enumerate(zip(('morphological', 'spatial', 'temporal'), chunk_size)):
            col = 'acc' if acc else 'auc'

            val = np.asarray(df.xs(cz, level="chunk_size").xs(mod, level="modality")[col])
            sem = np.expand_dims(np.asarray(sems.xs(cz, level="chunk_size").xs(mod, level="modality")[col]), axis=0)
            semn = np.expand_dims(np.asarray(semsn.xs(cz, level="chunk_size").xs(mod, level="modality")[col]), axis=0)
            yerr = np.concatenate((val - semn, sem - val), axis=0)

            dev_col = 'dev_acc' if acc else 'dev_auc'
            dev_val = np.asarray(df.xs(cz, level="chunk_size").xs(mod, level="modality")[dev_col])
            dev_sem = np.expand_dims(np.asarray(sems.xs(cz, level="chunk_size").xs(mod, level="modality")[dev_col]),
                                     axis=0)
            dev_semn = np.expand_dims(np.asarray(semsn.xs(cz, level="chunk_size").xs(mod, level="modality")[dev_col]),
                                      axis=0)
            dev_yerr = np.concatenate((dev_val - dev_semn, dev_sem - dev_val), axis=0)

            _ = ax.bar(x[i] + 0.5 * width, val, width, label='nCX', yerr=yerr, color=colors[0])
            # autolabel(rects1, ax, acc)
            _ = ax.bar(x[i] - 0.5 * width, dev_val, width, yerr=dev_yerr, label='CA1', color=colors[1])
            # autolabel(rects2, ax, acc)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        ax.set_yticks([0, 0.5, 1])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('AUC')

    ax.legend()

    fig.tight_layout()

    if name is None:
        plt.suptitle(title)
        plt.show()
    else:
        plt.savefig(SAVE_PATH + f"{name}_{'acc' if acc else 'auc'}.pdf", transparent=True)


def plot_results(df, sems, restriction, acc=True, name=None, chunk_size=None, mode='bar', dev=False, semsn=None):
    if name is None:
        title = ''

    labels = get_labels([ind[1] for ind in df.index])

    if chunk_size is None:
        chunk_size = get_labels([ind[2] for ind in df.index])

    if semsn is None:
        semsn = sems

    if mode == 'bar':
        fig, ax = plt.subplots(figsize=(3 * len(chunk_size) * len(labels), 9))
    else:
        fig, ax = plt.subplots(figsize=(9, 6))

    x = np.arange(len(labels))  # the label locations
    width = 1 / (len(chunk_size) + 1)  # the width of the bars

    col = 'acc' if acc else 'auc'
    if dev:
        col = 'dev_' + col

    colors = ['#FFFFFF', '#E0E0E0', '#C0C0C0', '#A0A0A0', '#808080', '#696969'][::6 // len(chunk_size)]
    colors = ['#A0A0A0', '#E0E0E0']

    if mode == 'bar':
        for i, cz in enumerate(chunk_size):
            val = df.xs(cz, level="chunk_size")[col].to_numpy()
            sem = sems.xs(cz, level="chunk_size")[col].to_numpy() - val
            semn = val - semsn.xs(cz, level="chunk_size")[col].to_numpy()
            yerr = np.concatenate((semn, sem), axis=0)

            rects = ax.bar(x - (i - len(chunk_size) / 2) * width, val, width, label=f'chunk size={cz}',
                           yerr=np.expand_dims(yerr, axis=1),
                           color=colors[i])
            autolabel(rects, ax, acc)

        chance_level = 0.5 if not acc else 79.8  # TODO make this generic
        ax.axhline(y=chance_level, xmin=x[0], xmax=x[-1] + 2 * width * len(chunk_size), color='k', linestyle='--')

        if len(labels) > 1:
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_ylim(0, 110 if acc else 1.1)

    elif mode == 'plot':
        x = np.roll(np.array(chunk_size), -1)
        x[-1] = x[-2] * 2
        linestyles = ['solid', 'dashed', 'dashdot']
        for i, m_name in enumerate(labels):
            val = np.roll(np.asarray(df.xs(m_name, level="modality")[col]), -1)
            sem = np.roll(np.asarray(sems.xs(m_name, level="modality")[col]), -1) - val
            semn = val - np.roll(np.asarray(semsn.xs(m_name, level="modality")[col]), -1)
            yerr = np.concatenate((semn, sem), axis=0)

            ax.errorbar(x, val, yerr=yerr, c='k', linestyle=linestyles[i], label=m_name)

        x_labels = [str(int(xt)) for xt in x]
        x_labels[-1] = '0'
        ax.set_xticks(x)
        ax.set_xticklabels(labels=x_labels, rotation=-25)
    else:
        raise KeyError('mode can be only bar or plot')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    if acc:
        ax.set_ylabel('Scores (percentage)')
    else:
        ax.set_ylabel('AUC value')

    if len(chunk_size) > 1:
        ax.legend()

    fig.tight_layout()

    if name is None:
        plt.suptitle(title)
        plt.show()
    else:
        plt.savefig(SAVE_PATH + f"{name}_{'acc' if acc else 'auc'}.pdf", transparent=True)


if __name__ == "__main__":
    """
    checks:
    1) Is the name of the model correct?
    2) Are we passing all the modalities we wish to plot?
    3) Have we the correct path for the confusion matrix?
    """
    import warnings

    # warnings.simplefilter("error")
    cs = 0
    m = 'spatial'
    results = pd.read_csv(f'results_rf_090522_region.csv', index_col=0)

    """complete = results[results.restriction == 'complete']
    complete = complete[complete.modality == m]

    complete = complete[complete.chunk_size.isin([cs])]
    complete = complete.dropna(how='all', axis=1)

    grouped_complete = complete.groupby(by=['restriction', 'modality', 'chunk_size'])

    base = pd.read_csv(f'../ml/results_rf_100522_chance_balanced.csv', index_col=0)
    base = base[base.restriction == 'complete']
    base = base[base.modality == m]

    base = base[base.chunk_size.isin([cs])]
    base = base.dropna(how='all', axis=1)

    grouped_base = base.groupby(by=['restriction', 'modality', 'chunk_size'])

    complete_0 = complete[complete.chunk_size == cs]
    complete_0 = complete_0.dropna(how='all', axis=1)

    grouped_complete_0 = complete_0.groupby(by=['restriction', 'modality', 'chunk_size'])

    plot_fet_imp(grouped_complete_0.median(), grouped_complete_0.quantile(0.75), 'complete', grouped_base.median(),
                 chunk_size=[cs], name=None, modalities=[(m, SPATIAL)], semsn=grouped_complete_0.quantile(0.25))"""

    plot_test_vs_dev_bp(results, chunk_sizes=(800, 25, 800))

    """train, dev, test, _, _, _ = ML_util.get_dataset('../data_sets_090522_region/complete_2/spatial/0_0.800.2/')
    train_labels = train[:, :,  -1].flatten()
    ca1_labels = dev[:, :,  -1].flatten()
    ncx_labels = test[:, :,  -1].flatten()
    print(f"training PYR: {train_labels.sum()}, PV: {train_labels.size - train_labels.sum()}")
    print(f"CA1 PYR: {ca1_labels.sum()}, PV: {ca1_labels.size - ca1_labels.sum()}")
    print(f"nCX PYR: {ncx_labels.sum()}, PV: {ncx_labels.size - ncx_labels.sum()}")

    print()

    train, dev, test, _, _, _ = ML_util.get_dataset('../data_sets_290322/complete_2/spatial/0_0.800.2/')
    train_labels = train[:, :, -1].flatten()
    dev_labels = dev[:, :, -1].flatten()
    test_labels = test[:, :, -1].flatten()
    print(f"training PYR: {train_labels.sum()}, PV: {train_labels.size - train_labels.sum()}")
    print(f"dev PYR: {ca1_labels.sum()}, PV: {ca1_labels.size - ca1_labels.sum()}")
    print(f"test PYR: {ncx_labels.sum()}, PV: {ncx_labels.size - ncx_labels.sum()}")"""

    # plot_acc_vs_auc(grouped_complete.median(), grouped_complete.quantile(0.75), 'complete', name=None,
    #                semsn=grouped_complete.quantile(0.25))
    # plot_acc_vs_auc_new(grouped_complete.median(), grouped_complete.quantile(0.75), 'complete', name=None,
    #                     semsn=grouped_complete.quantile(0.25))

    """_, _, _, train_names_25, dev_names_25, test_names_25 = ML_util.get_dataset('../data_sets_290322/complete_2/spatial/25_0.800.2/')

    set_train_25 = set(train_names_25)
    set_test_25 = set(test_names_25)

    print(set_train_25.intersection(set_test_25))

    _, _, _, train_names_0, dev_names_0, test_names_0 = ML_util.get_dataset('../data_sets_290322/complete_2/spatial/0_0.800.2/')

    set_train_0 = set(train_names_0)
    set_test_0 = set(test_names_0)

    print(set_train_0.intersection(set_test_0))

    print(set_train_0.intersection(set_test_25))"""
