import seaborn as sns
import os
import matplotlib.pyplot as plt
import numpy as np


def create_heatmap(col_labels, row_labels, x_label, y_label, title, data, path=None):
    fig, ax = plt.subplots(1, 1, figsize=(len(col_labels) / 2, len(row_labels) / 2))
    ax = sns.heatmap(data, annot=True, ax=ax)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax.set_xticks(np.arange(0, len(col_labels)) + 0.5)
    ax.set_yticks(np.arange(0, len(row_labels)) + 0.5)
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    if path is None:
        plt.show()
    else:
        if not os.path.isdir(path):
            os.mkdir(path)
        plt.savefig(path + title, bbox_inches='tight')
