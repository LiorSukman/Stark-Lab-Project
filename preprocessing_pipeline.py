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
from light_removal import remove_light
from constants import UPSAMPLE, VERBOS, SEED, SESSION_TO_ANIMAL

# import the different features
from features.spatial_features_calc import calc_spatial_features, get_spatial_features_names
from features.morphological_features_calc import calc_morphological_features, get_morphological_features_names
from features.temporal_features_calc import calc_temporal_features, get_temporal_features_names

TEMP_PATH = 'temp_state\\'

punits = {'es04feb12_1_3_2',
          'es04feb12_1_4_17',
          'es04feb12_1_4_15',
          'es27mar12_2_2_2',
          'es21may12_1_1_5',
          'm361r2_13_1_6',
          'm258r1_42_1_12',
          'm649r1_21_2_18',
          'm649r1_21_2_7',
          'm649r1_21_2_22',
          'm649r1_22_2_3',
          'm649r1_22_3_2'}

def show_cluster(load_path, name):
    files = [TEMP_PATH + f for f in listdir(load_path) if isfile(join(load_path, f)) and name + '_' in f]
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
    # random.shuffle(files_list)
    fig, ax = plt.subplots(rows, cols, sharex=True, figsize=(4 * cols, 3 * rows))
    plt.tight_layout(pad=3)
    counter = 0
    for i, file in enumerate(files_list):
        if counter == rows * cols:
            break
        path_elements = file.split('\\')[-1].split('__')
        if path_elements[0].split('/')[-1] not in punits:
            continue
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


def load_clusters(load_path, groups=None):
    files_list = [TEMP_PATH + f for f in listdir(load_path) if isfile(join(load_path, f))]
    clusters = set()
    for file in tqdm(files_list):
        name = '_'.join(file.split('\\')[-1].split('__')[0].split('_')[:-1])
        if groups is not None and groups[name] != 8:
            continue
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
    ret_spikes = []
    ret_inds = []
    # for each chunk size create the data
    for chunk_size in spikes_in_waveform:
        if chunk_size == 0:  # unit based approach
            mean = cluster.calc_mean_waveform()
            ret_spikes.append([mean])
            ret_inds.append(np.expand_dims(np.arange(len(cluster.spikes)), axis=0))
        elif chunk_size == 1:  # chunk based approach with raw spikes
            ret_spikes.append(cluster.spikes)
            ret_inds.append(np.expand_dims(np.arange(len(cluster.spikes)), 1))
        else:  # chunk based approach
            if cluster.np_spikes is None:  # this is done for faster processing
                cluster.finalize_cluster()
            spikes = cluster.np_spikes
            inds = np.arange(spikes.shape[0])
            np.random.seed(SEED)
            np.random.shuffle(inds)
            spikes = spikes[inds]
            k = spikes.shape[0] // chunk_size  # number of chunks
            if k == 0:  # cluster size is larger than the number of spikes in this cluster, same as chunk size of 0
                ret_spikes.append([cluster.calc_mean_waveform()])
                ret_inds.append(np.array([np.arange(len(cluster.spikes))]))
                continue
            chunks = np.array_split(spikes, k)  # split the data into k chunks of minimal size of chunk_size

            ret_inds.append(np.array(np.array_split(inds, k)))
            res = []
            for chunk in chunks:
                res.append(Spike(data=chunk.mean(axis=0)))  # take the average spike
            ret_spikes.append(res)

    return ret_spikes, ret_inds


def only_save(path, mat_file, xml, consider_lights, remove_lights):
    if xml:
        groups = read_xml('Data/')
    else:
        groups = None
    clusters_generator = read_all_directories(path, mat_file, groups)
    punits_counter = 0
    stats = pd.DataFrame(
        {'recording': [], 'shank': [], 'id': [], 'label': [], 'spike_prop': [], 'time_prop': []})
    for clusters in clusters_generator:
        pairs = None
        for cluster in clusters:  # for each unit
            recording_name = '_'.join(cluster.get_unique_name().split('_')[:-2])
            if SESSION_TO_ANIMAL[recording_name] == 401:
                print('Skipped cluster from animal 401')
                continue
            is_punit = cluster.fix_punits()
            if is_punit:
                punits_counter += 1

            if consider_lights:
                inds, spike_prop, time_prop, pairs = remove_light(cluster, remove_lights, pairs=pairs)
                cluster.timings = cluster.timings[inds]
                cluster.finalize_cluster()
                cluster.np_spikes = cluster.np_spikes[inds]
                print(f"for cluster {cluster.get_unique_name()}, spike proportion is {spike_prop} while time proportion"
                      f" is {time_prop}")

                stats = stats.append({'recording': cluster.filename, 'shank': cluster.shank,
                                      'id': cluster.num_within_file, 'label': cluster.label, 'spike_prop': spike_prop,
                                      'time_prop': time_prop}, ignore_index=True)

            cluster.save_cluster(TEMP_PATH)
    if consider_lights:
        stats.to_csv('light_stats.csv')
    print(f"number of punits is {punits_counter}")


def run(path, chunk_sizes, csv_folder, mat_file, load_path, xml=None):
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
    if xml:
        groups = read_xml('Data/')
    else:
        groups = None

    if load_path is None:
        clusters_generator = read_all_directories(path, mat_file, groups)
    else:
        clusters_generator = load_clusters(load_path, groups)

    # define headers for saving later 
    headers = get_spatial_features_names()
    headers += get_morphological_features_names()
    headers += get_temporal_features_names()
    headers += ['max_abs', 'name', 'label']

    for clusters in clusters_generator:
        for cluster in clusters:  # for each unit
            print('Processing cluster:' + cluster.get_unique_name())
            recording_name = '_'.join(cluster.get_unique_name().split('_')[:-2])
            if SESSION_TO_ANIMAL[recording_name] == 401:
                print('Skipped cluster from animal 401')
                continue
            max_abs = np.absolute(cluster.calc_mean_waveform().get_data()).max()
            # print('Fixing punits...')
            if load_path is None:
                cluster.fix_punits()
                cluster.save_cluster(TEMP_PATH)
            # print('Dividing data to chunks...')
            start_time = time.time()
            spike_chunks, ind_chunks = create_chunks(cluster, spikes_in_waveform=chunk_sizes)
            end_time = time.time()
            if VERBOS:
                print(f"chunk creation took {end_time - start_time:.3f} seconds")

            path = csv_folder + 'mean_spikes'
            if not os.path.isdir(path):
                os.mkdir(path)
            path += '\\' + cluster.get_unique_name()
            np.save(path, cluster.calc_mean_waveform().data)

            for chunk_size, rel_data, inds in zip(chunk_sizes, spike_chunks, ind_chunks):
                # upsample
                rel_data = [Spike(data=signal.resample(spike.data, UPSAMPLE * spike.data.shape[1], axis=1))
                            for spike in rel_data]
                temporal_features_mat = calc_temporal_features(cluster.timings, inds)
                spatial_features_mat = calc_spatial_features(rel_data)
                morphological_features_mat = calc_morphological_features(rel_data)
                feature_mat_for_cluster = np.concatenate((spatial_features_mat, morphological_features_mat,
                                                          temporal_features_mat), axis=1)
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
    parser.add_argument('--calc_features', type=bool, default=True,
                        help='path to load clusters from, make sure directory exists')
    parser.add_argument('--display', type=bool, default=False,
                        help='display a set of random clusters')
    parser.add_argument('--consider_light', type=bool, default=False,
                        help='Whether to take into account light stimulus while reading the data, will be processed'
                             ' based on remove_light (only taken into account if calc_features is False)')
    parser.add_argument('--remove_light', type=bool, default=True,
                        help='if True remove light induced spikes, otherwise keeps only light induced spikes')
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
    consider_lights = args.consider_light
    remove_lights = args.remove_light
    plot_cluster = args.plot_cluster
    xml = args.xml

    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    if plot_cluster is not None:
        show_cluster(arg_load_path, plot_cluster)
        sys.exit(0)

    if args.display:
        create_fig(arg_load_path, 3, 4)
        sys.exit(0)

    if args.calc_features:
        run(dirs_file, tuple(arg_chunk_sizes), save_path, spv_mat, arg_load_path, xml)
    else:
        only_save(dirs_file, spv_mat, xml, consider_lights, remove_lights)
