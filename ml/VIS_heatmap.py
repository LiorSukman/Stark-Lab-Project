import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def create_heatmap(col_labels, row_labels, x_label, y_label, title, data, path = None):
    ax = sns.heatmap(data, annot = True)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    plt.setp(ax.get_xticklabels(), rotation = 30, ha = "right", rotation_mode = "anchor")
    plt.setp(ax.get_yticklabels(), rotation = 30, ha = "right", rotation_mode = "anchor")

    if path == None:
        plt.show()
    else:
        plt.savefig(self.path + title, bbox_inches = 'tight')
