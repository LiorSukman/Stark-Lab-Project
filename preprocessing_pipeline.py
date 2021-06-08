from turtledemo.chaos import plot

import pandas as pd
import numpy as np
from tqdm import tqdm
import scipy.signal as signal
import argparse
import os
from os import listdir
from os.path import isfile, join
import time
import random
import sys
import matplotlib.pyplot as plt

from read_data import read_all_directories
from clusters import Spike, Cluster
from xml_reader import read_xml
from constants import UPSAMPLE, VERBOS, SEED

# import the different features
from features.spatial_features_calc import calc_spatial_features, get_spatial_features_names
from features.morphological_features_calc import calc_morphological_features, get_morphological_features_names
from features.temporal_features_calc import calc_temporal_features, get_temporal_features_names

TEMP_PATH = 'temp_state\\'

def show_cluster(load_path, name):
    files = [TEMP_PATH + f for f in listdir(load_path) if isfile(join(load_path, f)) and name+'_' in f]
    assert len(files) == 2
    spikes_f, timimg_f = files
    if 'timing' not in timimg_f:
        spikes_f, timimg_f = timimg_f, spikes_f
    cluster = Cluster()
    cluster.load_cluster(spikes_f)
    cluster.load_cluster(timimg_f)
    assert cluster.assert_legal()
    cluster.plot_cluster()

def create_fig(load_path, rows, cols):
    clusters = set()
    files_list = [TEMP_PATH + f for f in listdir(load_path) if isfile(join(load_path, f))]
    random.shuffle(files_list)
    fig, ax = plt.subplots(rows, cols, sharex=True, figsize=(4 * cols, 3 * rows))
    plt.tight_layout(pad=3)
    counter = 0
    for i, file in enumerate(files_list):
        if counter == rows * cols:
            break
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
        c_ax = ax[counter // cols, counter % cols]
        cluster.plot_cluster(ax=c_ax)
        label = 'Pyramidal' if cluster.label == 1 else 'PV' if cluster.label == 0 else 'untitled'
        title = f"{label} cluster {unique_name}"
        c_ax.set_title(title, size=10)
        counter += 1
        print(f"Processed cluster {unique_name} for display ({counter}/{cols * rows})")
    plt.show()


def load_clusters(load_path):
    files_list = [TEMP_PATH + f for f in listdir(load_path) if isfile(join(load_path, f))]
    clusters = set()
    for file in tqdm(files_list):
        start_time = time.time()
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

        end_time = time.time()
        if VERBOS:
            print(f"cluster loading took {end_time - start_time:.3f} seconds")
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
            np.random.seed(SEED)
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


def only_save(path, mat_file, xml):
    if xml:
        groups = read_xml('Data/')
    else:
        groups = None
    clusters_generator = read_all_directories(path, mat_file, groups)
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
    headers = get_spatial_features_names()
    headers += get_morphological_features_names()
    headers += get_temporal_features_names()
    headers += ['max_abs', 'name', 'label']

    for clusters in clusters_generator:
        for cluster in clusters:  # for each unit
            print('Processing cluster:' + cluster.get_unique_name())
            max_abs = np.absolute(cluster.calc_mean_waveform().get_data()).max()
            # print('Fixing punits...')
            if load_path is None:
                cluster.fix_punits()
                cluster.save_cluster(TEMP_PATH)
            # print('Dividing data to chunks...')
            start_time = time.time()
            relevant_data = create_chunks(cluster, spikes_in_waveform=chunk_sizes)
            end_time = time.time()
            if VERBOS:
                print(f"chunk creation took {end_time - start_time:.3f} seconds")

            path = csv_folder + 'mean_spikes'
            if not os.path.isdir(path):
                os.mkdir(path)
            path += '\\' + cluster.get_unique_name()
            np.save(path, cluster.calc_mean_waveform().data)

            temporal_features_mat = calc_temporal_features(cluster.timings)
            for chunk_size, rel_data in zip(chunk_sizes, relevant_data):
                # upsample
                rel_data = [Spike(data=signal.resample(spike.data, UPSAMPLE * spike.data.shape[1], axis=1))
                            for spike in rel_data]
                spatial_features_mat = calc_spatial_features(rel_data)
                morphological_features_mat = calc_morphological_features(rel_data)
                feature_mat_for_cluster = np.concatenate((spatial_features_mat, morphological_features_mat,
                                                          np.repeat(temporal_features_mat, len(spatial_features_mat),
                                                                    axis=0)), axis=1)
                # Append metadata for the cluster
                max_abss = np.ones((len(rel_data), 1)) * max_abs
                feature_mat_for_cluster = np.concatenate((feature_mat_for_cluster, max_abss), axis=1)

                names = np.ones((len(rel_data), 1), dtype=object) * cluster.get_unique_name()
                feature_mat_for_cluster = np.concatenate((feature_mat_for_cluster, names), axis=1)

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
    parser = argparse.ArgumentParser(description="preprocessing pipeline\n")

    parser.add_argument('--dirs_file', type=str, help='path to data directories file', default='dirs.txt')
    parser.add_argument('--chunk_sizes', type=int, help='chunk sizes to create data for, can be a list',
                        default=[0, 200, 500])
    parser.add_argument('--save_path', type=str, default='clustersData\\',
                        help='path to save csv files to, make sure the directory exists')
    parser.add_argument('--load_path', type=str, default='temp_state\\',
                        help='path to load clusters from, make sure directory exists')
    parser.add_argument('--calc_features', type=bool, default=False,
                        help='path to load clusters from, make sure directory exists')
    parser.add_argument('--display', type=bool, default=True,
                        help='display a set of random clusters')
    parser.add_argument('--plot_cluster', type=str, default=None,
                        help='display a specific cluster')
    parser.add_argument('--spv_mat', type=str, default='Data\\CelltypeClassification.mat', help='path to SPv matrix')
    parser.add_argument('--xml', type=bool, default=True, help='whether to assert using information in xml files when '
                                                               'reading the raw data')

    args = parser.parse_args()

    dirs_file = args.dirs_file
    arg_chunk_sizes = args.chunk_sizes
    save_path = args.save_path
    arg_load_path = args.load_path
    spv_mat = args.spv_mat
    plot_cluster = args.plot_cluster
    xml = args.xml

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    if plot_cluster is not None:
        show_cluster(arg_load_path, plot_cluster)
        sys.exit(0)

    if args.display:
        create_fig(arg_load_path, 10, 8)
        sys.exit(0)

    if args.calc_features:
        run(dirs_file, tuple(arg_chunk_sizes), save_path, spv_mat, arg_load_path)
    else:
        only_save(dirs_file, spv_mat, xml)
