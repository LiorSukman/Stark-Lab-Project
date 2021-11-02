import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ML_util

from constants import SPATIAL, MORPHOLOGICAL, TEMPORAL, SPAT_TEMPO, STARK_SPAT, STARK_SPAT_TEMPO, STARK
from constants import WIDTH, T2P, WIDTH_SPAT, T2P_SPAT
from constants import TRANS_MORPH
from constants import feature_names as fet_names

chunks = [0, 500, 200]
restrictions = ['complete', 'no_small_sample']
modalities = ['spatial', 'morphological', 'temporal', 'spat_tempo']
NUM_FETS = 28


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


def autolabel(rects, ax):
    """
    Attach a text label above each bar in *rects*, displaying its height.
    This function was taken from https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    """
    for rect in rects:
        height = rect.get_height()
        if height == 0:
            continue
        ax.annotate(f'{round(height, 3)}%',
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

def plot_fet_imp(df, sems, restriction):
    #title = get_title(restriction)
    modalities = [('spatial', SPATIAL), ('temporal', TEMPORAL), ('spat_tempo', SPAT_TEMPO),
                  ('morphological', MORPHOLOGICAL)]
    fets_org = [f"feature {f+1}" for f in range(NUM_FETS)]
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

        v0, e0 = df_m[0], sems_m[0]
        v200, e200 = df_m[1], sems_m[1]
        v500, e500 = df_m[2], sems_m[2]

        width = 0.3  # the width of the bars

        fig, ax = plt.subplots(figsize=(6, 12))
        rects1 = ax.bar(x - width, v0, width - 0.02, label='chunk_size = 0', yerr=e0, color='#404040',
                        edgecolor='k')
        rects2 = ax.bar(x, v500, width - 0.02, label='chunk_size = 500', yerr=e500, color='#808080',
                        edgecolor='k')
        rects3 = ax.bar(x + width, v200, width - 0.02, label='chunk_size = 200', yerr=e200,
                        color='#ffffff', edgecolor='k')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Importance')
        # ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(names_m, rotation=-25)
        ax.legend()

        fig.tight_layout()

        plt.show()


def plot_results(df, sems, restriction, acc=True):
    title = get_title(restriction)
    print(title)

    labels = get_labels([ind[1] for ind in df.index])

    if acc:
        zero = df.xs(0, level="chunk_size").acc
        five_hundred = df.xs(500, level="chunk_size").acc
        two_hundred = df.xs(200, level="chunk_size").acc
        zero_sem = sems.xs(0, level="chunk_size").acc
        five_hundred_sem = sems.xs(500, level="chunk_size").acc
        two_hundred_sem = sems.xs(200, level="chunk_size").acc
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
    rects1 = ax.bar(x - width, zero, width-0.02, label='chunk_size = 0', yerr=zero_sem, color='#404040', edgecolor='k')
    rects2 = ax.bar(x, five_hundred, width-0.02, label='chunk_size = 500', yerr=five_hundred_sem, color='#808080', edgecolor='k')
    rects3 = ax.bar(x + width, two_hundred, width-0.02, label='chunk_size = 200', yerr=two_hundred_sem, color='#ffffff', edgecolor='k')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores (percentage)')
    # ax.set_ylim(ymin=60)
    # ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)
    ax.axis('off')

    fig.tight_layout()

    plt.show()

if __name__ == "__main__":
    model = 'rf'
    results = pd.read_csv(f'results_{model}_w_perm_imp.csv', index_col=0)
    complete = results[results.restriction == 'complete']
    #complete = complete[complete.modality == 'morphological']
    no_small_sample = results[results.restriction == 'no_small_sample']
    grouped_complete = complete.groupby(by=['restriction', 'modality', 'chunk_size'])
    grouped_no_small_sample = no_small_sample.groupby(by=['restriction', 'modality', 'chunk_size'])

    #plot_results(grouped_complete.mean(), grouped_complete.sem(), 'complete', acc=True)
    #plot_results(grouped_complete.mean(), grouped_complete.sem(), 'complete', acc=False)
    plot_fet_imp(grouped_complete.mean(), grouped_complete.sem(), 'complete')
    plot_results(grouped_no_small_sample.mean(), grouped_no_small_sample.sem(), 'no_small_sample', acc=True)
    plot_results(grouped_no_small_sample.mean(), grouped_no_small_sample.sem(), 'no_small_sample', acc=False)
