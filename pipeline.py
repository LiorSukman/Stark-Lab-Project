import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy.signal as signal
import argparse
import os
from os import listdir
from os.path import isfile, join

from read_data import read_all_directories
from clusters import Spike, Cluster
from constants import UPSAMPLE

# import the different features
from features.spatial_features_calc import calc_spatial_features, get_spatial_features_names
from features.morphological_features_calc import calc_morphological_features, get_morphological_features_names
from features.temporal_features_calc import calc_temporal_features, get_temporal_features_names

TEMP_PATH = 'temp_state\\'


def load_clusters(load_path):
    files_list = [TEMP_PATH + f for f in listdir(load_path) if isfile(join(load_path, f))]
    clusters = set()
    for file in tqdm(files_list):
        path_elements = file.split('\\')[-1].split('__')
        if 'timing' in path_elements[-1]:
            continue
        unique_name = path_elements[0]
        if unique_name not in clusters:
            clusters.add(unique_name)
        else:
            raise Exception('Duplicate file in load path')
        cluster = Cluster()
        cluster.load_cluster(file)
        timing_file = file.replace('spikes', 'timings')
        cluster.load_cluster(timing_file)

        assert cluster.assert_legal()
        yield [cluster]


def create_chunks(cluster, spikes_in_waveform=(200,)):
    """
    inputs:
    cluster: an object of type Cluster; holding all the information for a specific unit
    spikes_in_waveform: tuple of non-negative integers (default = [200]); representing the different chunk sizes
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
                cluster.finalize_cluster()
            spikes = cluster.np_spikes
            np.random.shuffle(spikes)
            k = spikes.shape[0] // chunk_size  # number of chunks
            if k == 0:  # cluster size is larger than the number of spikes in this cluster, same as chunk size of 0
                ret.append([cluster.calc_mean_waveform()])
                continue
            chunks = np.array_split(spikes, k)  # split the data into k chunks of minimal size of chunk_size
            res = []
            for chunk in chunks:
                res.append(Spike(data=chunk.mean(axis=0)))  # take the average spike
            ret.append(res)

    return ret


def only_save(path, mat_file):
    clusters_generator = read_all_directories(path, mat_file)
    for clusters in clusters_generator:
        for cluster in clusters:  # for each unit
            cluster.fix_punits()
            cluster.save_cluster(TEMP_PATH)


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
    if load_path is None:
        clusters_generator = read_all_directories(path, mat_file)
    else:
        clusters_generator = load_clusters(load_path)

    # define headers for saving later 
    # headers = get_spatial_features_names()
    headers = get_morphological_features_names()
    headers += get_temporal_features_names()
    headers += ['label']

    for clusters in clusters_generator:
        for cluster in clusters:  # for each unit
            print('Processing cluster:' + cluster.get_unique_name())
            # print('Fixing punits...')
            if load_path is None:
                cluster.fix_punits()
                cluster.save_cluster(TEMP_PATH)
            # print('Dividing data to chunks...')
            relevant_data = create_chunks(cluster, spikes_in_waveform=chunk_sizes)
            temporal_features_mat = calc_temporal_features(cluster.timings)
            for chunk_size, rel_data in zip(chunk_sizes, relevant_data):
                # upsample
                rel_data = [Spike(data=signal.resample(spike.data, UPSAMPLE * spike.data.shape[1], axis=1))
                            for spike in rel_data]
                # spatial_features_mat = calc_spatial_features(rel_data)
                morphological_features_mat = calc_morphological_features(rel_data)
                # feature_mat_for_cluster = np.concatenate((spatial_features_mat, morphological_features_mat,
                #                                          temporal_features_mat), axis=1)
                feature_mat_for_cluster = np.concatenate(
                    (morphological_features_mat,
                     np.repeat(temporal_features_mat, len(morphological_features_mat), axis=0)), axis=1)

                # Append the label for the cluster
                labels = np.ones((len(rel_data), 1)) * cluster.label
                feature_mat_for_cluster = np.concatenate((feature_mat_for_cluster, labels), axis=1)

                # Save the data to a separate file (one for each cluster)
                path = csv_folder + str(chunk_size)
                if not os.path.isdir(path):
                    os.mkdir(path)
                path += '\\' + cluster.get_unique_name() + ".csv"
                df = pd.DataFrame(data=feature_mat_for_cluster)
                df.to_csv(path_or_buf=path, index=False, header=headers)  # save to csv
            """path = csv_folder + cluster.get_unique_name() + ".csv"
            df = pd.DataFrame(data=temporal_features_mat)
            df.to_csv(path_or_buf=path, index=False, header=headers)"""
            print('saved clusters to csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pipeline\n")

    parser.add_argument('--dirs_file', type=str, help='path to data directories file', default='dirs.txt')
    parser.add_argument('--chunk_sizes', type=int, help='chunk sizes to create data for, can be a list',
                        default=[0, 200, 500])
    parser.add_argument('--save_path', type=str, default='clustersData\\',
                        help='path to save csv files to, make sure the directory exists')
    parser.add_argument('--load_path', type=str, default='temp_state\\',
                        help='path to load clusters from, make sure directory exists')
    parser.add_argument('--calc_features', type=bool, default=True,
                        help='path to load clusters from, make sure directory exists')
    parser.add_argument('--spv_mat', type=str, default='Data\\CelltypeClassification.mat', help='path to SPv matrix')

    args = parser.parse_args()

    dirs_file = args.dirs_file
    arg_chunk_sizes = args.chunk_sizes
    save_path = args.save_path
    arg_load_path = args.load_path
    spv_mat = args.spv_mat

    if args.calc_features:
        run(dirs_file, tuple(arg_chunk_sizes), save_path, spv_mat, arg_load_path)
    else:
        only_save(dirs_file, spv_mat)
