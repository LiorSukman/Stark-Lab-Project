import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ML_util

chunks = [0, 500, 200]
restrictions = ['complete', 'no_small_sample']
modalities = ['spatial', 'morphological', 'temporal', 'spat_tempo']

def get_title(restriction):
    seeds = np.arange(10)
    tot, tst, pyr, intn = [], [], [], []
    for seed in seeds:
        data_path = f"../data_sets/{restriction}_{seed}/spatial/0_0.60.20.2/"
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
        ax.annotate('{}'.format(round(height, 2)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def get_labels(lst):
    ret = []
    for element in lst:
        if element not in ret:
            ret.append(element)
    return ret

def plot_results(df, sems, restriction, acc=True):
    title = get_title(restriction)

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

    fig, ax = plt.subplots(figsize=(12, 12))
    rects1 = ax.bar(x - width, zero, width, label='chunk_size = 0', yerr=zero_sem)
    rects2 = ax.bar(x, five_hundred, width, label='chunk_size = 500', yerr=five_hundred_sem)
    rects3 = ax.bar(x + width, two_hundred, width, label='chunk_size = 200', yerr=two_hundred_sem)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Scores (percentage)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    autolabel(rects1, ax)
    autolabel(rects2, ax)
    autolabel(rects3, ax)

    fig.tight_layout()

    plt.show()


results = pd.read_csv('results_svm.csv', index_col=0)
complete = results[results.restriction == 'complete']
no_small_sample = results[results.restriction == 'no_small_sample']
grouped_complete = complete.groupby(by=['restriction', 'modality', 'chunk_size'])
grouped_no_small_sample = no_small_sample.groupby(by=['restriction', 'modality', 'chunk_size'])

plot_results(grouped_complete.mean(), grouped_complete.sem(), 'complete', acc=True)
plot_results(grouped_complete.mean(), grouped_complete.sem(), 'complete', acc=False)
plot_results(grouped_no_small_sample.mean(), grouped_no_small_sample.sem(), 'no_small_sample', acc=True)
plot_results(grouped_no_small_sample.mean(), grouped_no_small_sample.sem(), 'no_small_sample', acc=False)
