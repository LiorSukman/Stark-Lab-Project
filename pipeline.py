import pandas as pd
import numpy as np
import time
import argparse

from read_data import read_all_directories
from clusters import Spike, Cluster

# import the different features
from features.FET_time_lag import Time_Lag_Feature
from features.FET_spd import SPD
from features.FET_da import DA
from features.FET_depolarization_graph import DepolarizationGraph
from features.FET_channel_contrast_feature import ChannelContrast
from features.FET_geometrical_estimation import GeometricalEstimation

features = [Time_Lag_Feature(), SPD(), DA(), DepolarizationGraph(), ChannelContrast(), GeometricalEstimation()]

TEMP_PATH = 'temp_state\\'


def get_list_of_relevant_waveforms_from_cluster(cluster, spikes_in_waveform=[200]):
    """
    inputs:
    cluster: an object of type Cluster; holding all the information for a specific unit
    spikes_in_waveform: list of non-negative integers (default = [200]); representing the different chunk sizes
    value of zero means taking the unit-based approach (i.e. a single chunk for each cluster)

    returns:
    a list of size |spikes_in_waveform|, each element is a list of chunks 
    """
    ret = []

    # for each chunk size create the data
    for chunk_size in spikes_in_waveform:
        if chunk_size == 0:  # unit based approach
            mean = cluster.calc_mean_waveform()
            ret.append([mean])
        elif chunk_size == 1:  # chunk based approach with raw spikes
            ret.append(cluster.spikes)
        else:  # chunk based approach
            if cluster.np_spikes is None:  # this is done for faster processing
                cluster.finalize_spikes()
            spikes = cluster.np_spikes
            np.random.shuffle(spikes)
            k = spikes.shape[0] // chunk_size  # number of chunks
            if k == 0:  # cluster size is larger than the number of spikes in this cluster, same as chunk size of 0
                ret.append([cluster.calc_mean_waveform()])
                continue
            chunks = np.array_split(spikes, k)  # split the data into k chunks of minimal size of
            res = []
            for chunk in chunks:
                res.append(Spike(data=chunk.mean(axis=0)))  # take the average spike
            ret.append(res)

    return ret


def run(path, chunk_sizes, csv_folder, mat_file, load_path):
    """
    main pipeline function
    input:
    path: string; path to a text file containing the paths to the data directories
    mat_file: string; path the mat file containing the spv data
    chunk_sizes: list of non-negative integers; representing the different chunk sizes
    csv_folder: string; the folder in which the processed data should be saved, assumed to be created
    load_path: string; path from which to load clusters, if given path is ignored, ignored if None

    The function creates csv files with the features for the different units 
    """
    clusters_generator = read_all_directories(path, mat_file)

    # define headers for saving later 
    headers = []
    for feature in features:
        headers += feature.get_headers()
    headers += ['label']

    for clusters in clusters_generator:
        for cluster in clusters:  # for each unit
            cluster.save_cluster(TEMP_PATH)
            # print('Fixing punits...')
            cluster.fix_punits()
            # print('Dividing data to chunks...')
            relevant_data = get_list_of_relevant_waveforms_from_cluster(cluster, spikes_in_waveform=chunk_sizes)
            for chunk_size, relData in zip(chunk_sizes, relevant_data):
                feature_mat_for_cluster = None
                is_first_feature = True
                for feature in features:
                    # print('processing feature ' + feature.name + '...')
                    # start_time = time.time()
                    mat_result = feature.calculateFeature(relData)  # calculates the features, returns a matrix
                    # end_time = time.time()
                    # print('processing took %.4f seconds' % (end_time - start_time))
                    if is_first_feature:
                        feature_mat_for_cluster = mat_result
                    else:
                        feature_mat_for_cluster = np.concatenate((feature_mat_for_cluster, mat_result), axis=1)

                    is_first_feature = False

                # Append the label for the cluster
                labels = np.ones((len(relData), 1)) * cluster.label
                feature_mat_for_cluster = np.concatenate((feature_mat_for_cluster, labels), axis=1)

                # Save the data to a separate file (one for each cluster)
                path = csv_folder + str(chunk_size) + '\\' + cluster.get_unique_name() + ".csv"
                df = pd.DataFrame(data=feature_mat_for_cluster)
                df.to_csv(path_or_buf=path, index=False, header=headers)  # save to csv
            print('saved clusters to csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pipeline\n")

    parser.add_argument('--dirs_file', type=str, help='path to data directories file', default='dirs.txt')
    parser.add_argument('--chunk_sizes', type=int, help='chunk sizes to create data for, can be a list',
                        default=[0, 200, 500])
    parser.add_argument('--save_path', type=str, default='clustersData\\',
                        help='path to save csv files to, make sure the directory exists')
    parser.add_argument('--load_path', type=str, default=None,
                        help='path to load clusters from, make sure directory exists')
    parser.add_argument('--spv_mat', type=str, default='Data\\CelltypeClassification.mat', help='path to SPv matrix')

    args = parser.parse_args()

    dirs_file = args.dirs_file
    chunk_sizes = args.chunk_sizes
    save_path = args.save_path
    load_path = args.load_path
    spv_mat = args.spv_mat

    run(dirs_file, chunk_sizes, save_path, spv_mat, load_path)
