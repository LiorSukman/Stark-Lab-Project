import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import interpolate
from scipy.stats import sem as calc_sem
import ml.ML_util as ML_util
from utils.hideen_prints import HiddenPrints

from constants import SPATIAL, MORPHOLOGICAL, TEMPORAL, SPAT_TEMPO
from constants import TRANS_MORPH
from constants import feature_names as fet_names

chunks = [0, 500, 200]
restrictions = ['complete', 'no_small_sample']
modalities = ['spatial', 'morphological', 'temporal', 'spat_tempo']
NUM_FETS = 30
SAVE_PATH = '../../../data for figures/New/'


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


def my_change_length(x, y, length):
    xnew = np.linspace(0, 1, length)
    ynew = []
    loc = 0

    assert x[0] == 0 and x[-1] == 1 and y[0] == 0 and y[-1] == 1

    for xn in xnew:
        while True:
            if loc == len(x) - 1 or loc == 0:
                ynew.append(y[loc])
                if loc == 0:
                    loc += 1
                break
            elif x[loc - 1] <= xn < x[loc]:
                if y[loc - 1] == y[loc]:
                    ynew.append(y[loc - 1])
                else:
                    m = (y[loc - 1] - y[loc]) / (x[loc - 1] - x[loc])
                    ynew.append(y[loc - 1] + m * (xn - x[loc - 1]))
                break
            elif xn > x[loc]:
                loc += 1
            elif xn == x[loc]:
                ynew.append(y[loc])
                loc += 1
                break
            else:
                raise AssertionError

    ynew[-1] = 1

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
            mean_tprs = tprs.mean(axis=0)
            sem = calc_sem(tprs, axis=0)

            auc_col = 'auc' if not use_dev else 'dev_auc'

            auc = df_cz[auc_col].mean()

            label_add = f", chunk size = {cz}" if len(chunk_size) > 1 else ""

            ax.plot(mean_fprs, mean_tprs, label=f"AUC={auc:.3f}"+label_add)
            ax.fill_between(mean_fprs, mean_tprs - sem, mean_tprs + sem, alpha=0.2)

        ax.plot(mean_fprs, mean_fprs, color='k', linestyle='--', label='chance level')
        ax.legend()
        ax.set_xlabel("False Positive rate")
        ax.set_ylabel("True Positive rate")

        if name is None:
            plt.title(f"{m_name} chunk sizes: {chunk_size}")
            plt.show()
        else:
            plt.savefig(SAVE_PATH + f"{name}_{m_name}_roc_curve.pdf", transparent=True)


def plot_cf(tp, tn, fp, fn, tp_std, tn_std, fp_std, fn_std, cz_ax, cz):
    data = np.array([[tp, fn], [fp, tn]])
    data = data / data.sum(axis=1)
    pn_sign = u"\u00B1"
    labels = [[f"{tp:2g}{pn_sign}{tp_std:.2f}", f"{fn:2g}{pn_sign}{fn_std:.2f}"],
              [f"{fp:2g}{pn_sign}{fp_std:.2f}", f"{tn:2g}{pn_sign}{tn_std:.2f}"]]
    cmap = sns.light_palette("seagreen", as_cmap=True)
    ticklabels = ['PYR', 'IN']
    _ = sns.heatmap(data, annot=labels, fmt='', vmin=0, vmax=1, cmap=cmap, ax=cz_ax, xticklabels=ticklabels,
                    yticklabels=ticklabels)
    cz_ax.set_ylabel("True Label")
    cz_ax.set_xlabel('Predicted Label')


def plot_conf_mats(df, restriction, name=None, chunk_size=[0], modalities=None, use_dev=False, data_path=None):
    df_temp = df.copy()
    if data_path is None:
        data_path = f'../data_sets/{restriction}_0/spatial/0_0.800.2/'
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
    mean = grouped.mean()
    std = grouped.std()

    for m_name, _ in modalities:
        mean_m = mean.xs(m_name, level="modality")
        std_m = std.xs(m_name, level="modality")
        fig, ax = plt.subplots(len(chunk_size), sharex=True, sharey=True)
        if len(chunk_size) == 1:
            ax = [ax]
        for cz, cz_ax in zip(chunk_size, ax):
            mean_cz = mean_m.xs(cz, level="chunk_size")
            std_cz = std_m.xs(cz, level="chunk_size")
            tp, tn, fp, fn = mean_cz.tp[0], mean_cz.tn[0], mean_cz.fp[0], mean_cz.fn[0]
            tp_std, tn_std, fp_std, fn_std = std_cz.tp[0], std_cz.tn[0], std_cz.fp[0], std_cz.fn[0]
            plot_cf(tp, tn, fp, fn, tp_std, tn_std, fp_std, fn_std, cz_ax, cz)
            if len(chunk_size) > 1:
                cz_ax.set_title(f"Chunk size={cz}")
        if name is None:
            plt.show()
        else:
            plt.savefig(SAVE_PATH + f"{name}_{m_name}_conf_mat.pdf", transparent=True)


def get_title(restriction):
    seeds = np.arange(20)
    tot, tst, pyr, intn = [], [], [], []
    for seed in seeds:
        data_path = f"../data_sets/{restriction}_{seed}/spatial/0_0.800.2/"
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
                    ha='center', va='bottom', size=10)


def get_labels(lst):
    ret = []
    for element in lst:
        if element not in ret:
            ret.append(element)
    return ret


def plot_fet_imp(df, sems, restriction, name=None, chunk_size=None, modalities=None):
    # TODO make sure order is determined only by rows of used chunk_size
    if name is None:
        title = get_title(restriction)
    if modalities is None:
        modalities = [('spatial', SPATIAL), ('temporal', TEMPORAL), ('morphological', MORPHOLOGICAL)]

    if chunk_size is None:
        chunk_size = get_labels([ind[2] for ind in df.index])

    fets_org = [f"test feature {f + 1}" for f in range(NUM_FETS)]
    rem = [f"dev feature {f + 1}" for f in range(NUM_FETS) if f"dev feature {f + 1}" in df.columns]
    df = df.drop(columns=rem)
    sems = sems.drop(columns=rem)

    map = {f: name for (f, name) in zip(fets_org, fet_names)}
    df = df.rename(map, axis='columns')
    df = df.drop(columns=['seed', 'acc', 'auc', 'pyr_acc', 'in_acc'])
    df = df.drop(columns=['dev_acc', 'dev_auc', 'dev_pyr_acc', 'dev_in_acc'])
    sems = sems.rename(map, axis='columns')
    sems = sems.drop(columns=['seed', 'acc', 'auc', 'pyr_acc', 'in_acc'])
    sems = sems.drop(columns=['dev_acc', 'dev_auc', 'dev_pyr_acc', 'dev_in_acc'])
    for m_name, m_places in modalities:
        df_m = df.xs(m_name, level="modality").dropna(axis=1)
        sems_m = sems.xs(m_name, level="modality").dropna(axis=1)
        names_m = np.asarray(fet_names)[m_places[:-1]]

        df_order = np.asarray(df_m)
        order = np.argsort((-1 * df_order).mean(axis=0))

        names_m = names_m[order]

        x = np.arange(len(names_m))  # the label locations

        width = 1 / (len(chunk_size) + 1)  # the width of the bars

        fig, ax = plt.subplots(figsize=(6, 12))

        colors = ['#696969', '#808080', '#A9A9A9', '#C0C0C0', '#D3D3D3', '#DCDCDC'][::6 // len(chunk_size)]

        for i, cz in enumerate(chunk_size):
            df_cz = np.asarray(df_m.xs(cz, level="chunk_size"))[:, order].flatten()
            sems_cz = np.asarray(sems_m.xs(cz, level="chunk_size"))[:, order].flatten()

            ax.bar(x - (i - len(chunk_size) // 2) * width, df_cz, width, label=f'chunk_size = {cz}', yerr=sems_cz,
                   edgecolor='k', color=colors[i])
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Importance')
        # ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(names_m, rotation=-25)

        if len(chunk_size) > 1:
            ax.legend()

        fig.tight_layout()

        if name is None:
            plt.suptitle(title)
            plt.show()
        else:
            plt.savefig(SAVE_PATH + f"{name}_fet_imp.pdf", transparent=True)


def plot_acc_vs_auc(df, sems, restriction, name=None):
    if name is None:
        title = get_title(restriction)

    labels = get_labels([ind[1] for ind in df.index])

    chunk_size = get_labels([ind[2] for ind in df.index])

    x = np.roll(np.array(chunk_size), -1)
    x[-1] = x[-2] * 2
    for m_name in labels:
        fig, ax1 = plt.subplots(figsize=(9, 6))
        ax2 = ax1.twinx()

        val_acc = np.roll(np.asarray(df.xs(m_name, level="modality").acc), -1)
        sem_acc = np.roll(np.asarray(sems.xs(m_name, level="modality").acc), -1)
        val_auc = np.roll(np.asarray(df.xs(m_name, level="modality").auc), -1)
        sem_auc = np.roll(np.asarray(sems.xs(m_name, level="modality").auc), -1)

        ax1.set_ylim(ymin=val_acc.min() * 0.95, ymax=val_acc.max()*1.1)
        ax2.set_ylim(ymin=val_auc.min() * 0.9, ymax=val_auc.max()*1.05)

        # ax.errorbar(x, val, yerr=sem, c='k', linestyle=ls, label=col)
        line1 = ax1.errorbar(x, val_acc, yerr=sem_acc, color='k', linestyle='solid', label='accuracy', capsize=2)
        line2 = ax2.errorbar(x, val_auc, yerr=sem_auc, color='k', linestyle='dashed', label='AUC', capsize=2)

        lines = [line1[0]] + [line2[0]]
        labs = ['accuracy', 'AUC']
        ax1.legend(lines, labs, loc=0)

        x_labels = [str(int(xt)) for xt in x]
        x_labels[-1] = '0'
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels=x_labels, rotation=-25)

        ax1.set_ylabel('Scores (percentage)')
        ax2.set_ylabel('AUC value')

        fig.tight_layout()

        if name is None:
            plt.suptitle(title)
            plt.show()
        else:
            plt.savefig(SAVE_PATH + f"{name}_{m_name}.pdf", transparent=True)
            plt.clf()
            plt.cla()
            plt.close('all')


def plot_acc_vs_auc_bar(df, name=None, chunk_size=[0], modalities=None):
    if modalities is None:
        modalities = [('spatial', SPATIAL), ('temporal', TEMPORAL), ('morphological', MORPHOLOGICAL)]

    for m_name, _ in modalities:
        df_m = df[df.modality == m_name]
        fig, ax1 = plt.subplots()
        x = np.arange(len(chunk_size))  # the label locations
        width = 0.4  # the width of the bars
        for cz in chunk_size:
            df_cz = df_m[df_m.chunk_size == cz]
            ax2 = ax1.twinx()
            ymax = 110
            scale = 0.01
            ax1.set_ylim(ymin=0, ymax=ymax)
            ax2.set_ylim(ymin=0, ymax=ymax * scale)

            grouped = df_cz.groupby(by=['restriction', 'modality', 'chunk_size'])
            mean = grouped.mean()
            sem = grouped.sem()
            accs = df_cz.acc.to_numpy()
            aucs = df_cz.auc.to_numpy()

            auc_val = mean.auc
            acc_val = mean.acc
            auc_sem = sem.auc
            acc_sem = sem.acc

            rects1 = ax1.bar(x - width, acc_val, width, yerr=acc_sem, color='#404040', edgecolor='k')
            rects2 = ax2.bar(x + width, auc_val, width, yerr=auc_sem, color='#404040', edgecolor='k')
            autolabel(rects1, ax1, True)
            autolabel(rects2, ax2, False)

            for acc, auc in zip(accs, aucs):
                ax2.plot([x - width, x + width], [acc * scale, auc], color='k', marker='o')

        if name is None:
            plt.show()
        else:
            plt.savefig(SAVE_PATH + f"{name}_{m_name}_acc_vs_auc.pdf", transparent=True)


def plot_test_vs_dev(df, sems, restriction, acc=True, name=None, chunk_size=None, mode='bar'):
    if name is None:
        title = 'dev-test comparison' + get_title(restriction)

    labels = get_labels([ind[1] for ind in df.index])

    if chunk_size is None:
        chunk_size = get_labels([ind[2] for ind in df.index])

    fig, ax = plt.subplots()

    x = np.arange(len(labels))  # the label locations
    width = 1 / (2 * len(chunk_size) + 1)  # the width of the bars

    colors = ['#696969', '#808080', '#A9A9A9', '#C0C0C0', '#D3D3D3', '#DCDCDC'][::6 // len(chunk_size)]

    if mode == 'bar':
        for i, cz in enumerate(chunk_size):
            col = 'acc' if acc else 'auc'
            val = df.xs(cz, level="chunk_size")[col]
            sem = sems.xs(cz, level="chunk_size")[col]

            dev_col = 'dev_acc' if acc else 'dev_auc'
            dev_val = df.xs(cz, level="chunk_size")[dev_col]
            dev_sem = sems.xs(cz, level="chunk_size")[dev_col]

            rects1 = ax.bar(x - (2 * i - len(chunk_size) // 2) * width, val, width, label=f'nCX; chunk size={cz}',
                            yerr=sem, color=colors[i])
            autolabel(rects1, ax, acc)

            rects2 = ax.bar(x - (1 + 2 * i - len(chunk_size) // 2) * width, dev_val, width, hatch="\\", yerr=dev_sem,
                            label=f'CA1; chunk size={cz}', color=colors[i])
            autolabel(rects2, ax, acc)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
    elif mode == 'plot':
        x = chunk_size
        linestyles = ['solid', 'dashed', 'dashdot']
        for i, m_name in enumerate(labels):
            col = 'acc' if acc else 'auc'
            val = np.asarray(df.xs(m_name, level="modality")[col])  # [::-1]
            sem = np.asarray(sems.xs(m_name, level="modality")[col])  # [::-1]

            ax.errorbar(x, val, yerr=sem, c='k', linestyle=linestyles[i], label=m_name)

        ax.set_xticks(x)
    else:
        raise KeyError('mode can be only bar or plot')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores (percentage)')

    ax.legend()

    fig.tight_layout()

    if name is None:
        plt.suptitle(title)
        plt.show()
    else:
        plt.savefig(SAVE_PATH + f"{name}_{'acc' if acc else 'auc'}.pdf", transparent=True)


def plot_results(df, sems, restriction, acc=True, name=None, chunk_size=None, mode='bar', dev=False):
    if name is None:
        title = get_title(restriction)

    labels = get_labels([ind[1] for ind in df.index])

    if chunk_size is None:
        chunk_size = get_labels([ind[2] for ind in df.index])

    if mode == 'bar':
        fig, ax = plt.subplots(figsize=(3 * len(chunk_size) * len(labels), 9))
    else:
        fig, ax = plt.subplots(figsize=(9, 6))

    x = np.arange(len(labels))  # the label locations
    width = 1 / (len(chunk_size) + 1)  # the width of the bars

    col = 'acc' if acc else 'auc'
    if dev:
        col = 'dev_' + col

    colors = ['#696969', '#808080', '#A9A9A9', '#C0C0C0', '#D3D3D3', '#DCDCDC'][::6 // len(chunk_size)]

    if mode == 'bar':
        for i, cz in enumerate(chunk_size):
            val = df.xs(cz, level="chunk_size")[col]
            sem = sems.xs(cz, level="chunk_size")[col]

            rects = ax.bar(x - (i - len(chunk_size) / 2) * width, val, width, label=f'chunk size={cz}', yerr=sem,
                           edgecolor='k', color=colors[i])
            autolabel(rects, ax, acc)

        chance_level = 0.5 if not acc else 79.8  # TODO make this generic
        ax.axhline(y=chance_level, xmin=x[0], xmax=x[-1]+2*width*len(chunk_size), color='k', linestyle='--')

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
            sem = np.roll(np.asarray(sems.xs(m_name, level="modality")[col]), -1)

            ax.errorbar(x, val, yerr=sem, c='k', linestyle=linestyles[i], label=m_name)

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
    model = 'rf'
    results = pd.read_csv(f'results_{model}.csv', index_col=0)
    complete = results[results.restriction == 'complete']
    # complete = complete[complete.chunk_size == 0]
    complete = complete.dropna(how='all', axis=1)
    no_small_sample = results[results.restriction == 'no_small_sample']
    grouped_complete = complete.groupby(by=['restriction', 'modality', 'chunk_size'])
    grouped_no_small_sample = no_small_sample.groupby(by=['restriction', 'modality', 'chunk_size'])
    # plot_acc_vs_auc_bar(complete)
    #plot_conf_mats(complete, 'complete')
    #plot_roc_curve(complete)

    #plot_acc_vs_auc(grouped_complete.mean(), grouped_complete.sem(), 'complete')

    #plot_results(grouped_complete.mean(), grouped_complete.sem(), 'complete', acc=True, mode='bar', dev=False)
    #plot_results(grouped_complete.mean(), grouped_complete.sem(), 'complete', acc=False, mode='bar', dev=False)
    #plot_fet_imp(grouped_complete.mean(), grouped_complete.sem(), 'complete')
    #plot_test_vs_dev(grouped_complete.mean(), grouped_complete.sem(), 'complete', acc=True, mode='bar')
    #plot_test_vs_dev(grouped_complete.mean(), grouped_complete.sem(), 'complete', acc=False, mode='bar')

    _, dev, test, _, _, _ = ML_util.get_dataset('../data_sets_region/complete_1/spatial/0_0.800.2/')
    ca1_labels = dev[:, :,  -1].flatten()
    ncx_labels = test[:, :,  -1].flatten()
    print(ncx_labels.mean())
    print(ca1_labels.mean())
