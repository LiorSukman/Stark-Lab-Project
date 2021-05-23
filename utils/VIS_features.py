import pandas as pd
import os
import seaborn as sn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import scipy.stats
import numpy as np
import argparse

NUM_FETS = 16
DIR = "../clustersData/0"


def get_df(data_path):
    """
    This function reads all files in DIR and creates a pandas data frame from all
    their information, then returns it.
    """
    df = pd.DataFrame()

    print("Iterating over files...")
    for file in os.listdir(data_path):
        if file.endswith(".csv") and 'all_clusters' not in file:
            path = os.path.join(data_path, file)
            df = df.append(pd.read_csv(path), ignore_index=True)

    return df


def corr_matrix(path):
    """
    The function creates a correlation matrix based on all features in the data
    see help for parameter explanation
    """
    df = get_df(path)

    # drop the label
    df = df.drop(labels='label', axis=1)

    # create and plot the correlation matrix
    corrMatrix = df.corr()
    sn.heatmap(corrMatrix, annot=True)
    plt.show()


def feature_comparison(path, num_fets):
    """
    The function creates a bar graph comapring all features with separation by cell type,
    see help for parameter explanation
    """
    ind = np.arange(num_fets)
    df = get_df(path)

    data = df.to_numpy()
    features, data_labels = data[::-1], data[:-1]

    labels = ['dep_red', 'dep_sd', 'hyp_red', 'hyp_sd', 'spatial_dispersion_count', 'spatial_dispersion_sd', 'da',
              'da_sd', 'graph_avg_speed', 'graph_slowest_path', 'graph_fastest_path', 'Channels contrast',
              'geometrical_avg_shift', 'geometrical_shift_sd', 'geometrical_max_change']
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    pyr_inds = data_labels == 1
    in_inds = data_labels == 0
    pyr_fets = features[pyr_inds]
    pyr_means = np.mean(pyr_fets, axis=0)
    pyr_sem = scipy.stats.sem(pyr_fets, axis=0)
    in_fets = features[in_inds]
    in_means = np.mean(in_fets, axis=0)
    in_sem = scipy.stats.sem(in_fets, axis=0)
    width = 0.35

    p1 = plt.bar(ind - width / 2, pyr_means, width, yerr=pyr_sem)
    p2 = plt.bar(ind + width / 2, in_means, width, yerr=in_sem)
    plt.xticks(ind, labels, rotation=30, ha="right", rotation_mode="anchor")
    plt.legend((p1[0], p2[0]), ('Pyramidal', 'Interneuron'))
    plt.ylabel('Standardized scores')
    plt.show()


def feature_histogram(path, index, bins_start, bins_end, num_bins, title, x_label, y_label):
    """
    creates a density histogram for a feature,
    see help for parameter explanation
    """

    df = get_df(path)

    data = df.to_numpy()
    features, data_labels = data[:, :-1], data[:, -1]

    pyr_inds = data_labels == 1
    in_inds = data_labels == 0
    pyr_fet = features[pyr_inds][:, index]
    in_fet = features[in_inds][:, index]

    bins = np.linspace(bins_start, bins_end, num_bins)
    print(bins)

    plt.hist(in_fet, bins=bins, alpha=0.5, label='Interneuron', density=True)
    plt.hist(pyr_fet, bins=bins, alpha=0.5, label='Pyramidal', density=True)

    plt.legend(loc='upper right')
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VIS_features\n")

    parser.add_argument('--graph_type', type=str, help='visualization type (can be bar, hist or mat)', default='hist')
    parser.add_argument('--data_path', type=str, help='path to data', default=DIR)
    parser.add_argument('--num_fets', type=int, help='number of features in the data (relevant for the bar graph)',
                        default=NUM_FETS)
    parser.add_argument('--index', type=int, help='feature index (relevant for the hist graph)', default=0)
    parser.add_argument('--bins_start', type=int, help='start of bins (relevant for the hist graph)', default=0)
    parser.add_argument('--bins_end', type=int, help='end of bins (relevant for the hist graph)', default=100)
    parser.add_argument('--bins_num', type=int, help='number of bins (relevant for the hist graph)', default=100)
    parser.add_argument('--title', type=str, help='graph title (relevant for the hist graph)', default='Density Plot')
    parser.add_argument('--x_label', type=str, help='x axis label (relevant for the hist graph)', default='measurement')
    parser.add_argument('--y_label', type=str, help='y axis label (relevant for the hist graph)', default='density')

    args = parser.parse_args()

    graph_type = args.graph_type
    data_path = args.data_path
    num_fets = args.num_fets
    index = args.index
    bins_start = args.bins_start
    bins_end = args.bins_end
    bins_num = args.bins_num
    title = args.title
    x_label = args.x_label
    y_label = args.y_label

    if graph_type == 'mat':
        corr_matrix(data_path)
    elif graph_type == 'bar':
        feature_comparison(data_path, num_fets)
    elif graph_type == 'hist':
        feature_histogram(data_path, index, bins_start, bins_end, bins_num, title, x_label, y_label)
