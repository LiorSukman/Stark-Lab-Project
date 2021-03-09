import os
import math
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable

def create_batches(data, batch_size, should_shuffle = True):
   """
   This function should recieve the data as np array of waveforms (features and label), and return a list of batches, each batch is a tuple
   of torch's Variable of tensors:
   1) The input features
   2) The corresponding labels
   If should_shuffle (optional, bool, default = True) argument is True we shuffle the data (in place) before creating the batches
   """
   if should_shuffle:
      np.random.shuffle(data)
   batches = []
   cur_batch = 0
   number_of_batches = math.ceil(len(data) / batch_size)
   while cur_batch < number_of_batches:
      batch = data[cur_batch * batch_size: (cur_batch + 1) * batch_size]
      batch = (torch.from_numpy(batch[:,:-1]), torch.from_numpy(batch[:,-1]).long())
      batch = (Variable(batch[0], requires_grad = True), Variable(batch[1]))
      cur_batch += 1
      batches.append(batch)
   return batches

def split_features(data):
   """
   The function simply separates the features and the labels of the clusters
   """
   return data[:,:-1], data[:,-1]

def parse_test(data):
   """
   The function recieves test data of a cluster and transforms it so the NN can handle it 
   """
   features, label = split_features(data)
   features = Variable(torch.from_numpy(features))
   return features, label[0]
      
def is_legal(cluster):
   """
   This function determines whether or not a cluster's label is legal. It is assumed that all waveforms
   of the cluster have the same label.
   To learn more about the different labels please refer to the pdf file or the read_data.py file
   """
   row = cluster[0]
   return row[-1] >= 0

def read_data(path, should_filter = True, keep = []):
   """
   The function reads the data from all files in the path.
   It is assumed that each file represeents a single cluster, and have some number of waveforms.
   The should_filter (optional, bool, default = True) argument indicated whether we should filter out
   clusters with problematic label (i.e. < 0)
   """
   files = os.listdir(path)
   clusters = []
   for file in sorted(files):
      df = pd.read_csv(path + '/' + file)
      nd = df.to_numpy(dtype = 'float32')

      if keep != []:
         nd = nd[:, keep]
      
      if should_filter:
         if is_legal(nd):
            clusters.append(nd)
         else:
            continue
      else:
         clusters.append(nd)
   return np.asarray(clusters)

def break_data(data):
   """
   The function recieves unordered data and returns a list with three numpy arrays: 1) with all the pyramidal clusters,
   2) with all the interneuron clusters and 3) with all the unlabeled clusters
   """
   pyr_inds  = get_inds(data, 1)
   in_inds  = get_inds(data, 0)
   ut_inds  = get_inds(data, -1)
   ret = [data[pyr_inds], data[in_inds], data[ut_inds]]
   return ret

def was_created(paths, per_train, per_dev, per_test):
   """
   The function checks if all datasets were already creted and return True iff so
   """
   for path in paths:
      path = path + str(per_train) + str(per_dev) + str(per_test)
      if not os.path.isdir(path):
         return False
   return True
   

def create_datasets(per_train = 0.6, per_dev = 0.2, per_test = 0.2, datasets = 'datas.txt', should_filter = True, save_path = '../data_sets', verbos = False, keep = []):
   """
   The function creates all datasets from the data referenced by the datasets file and saves them
   """
   paths = []
   with open(datasets, 'r') as fid:
      while True:
         path = fid.readline()
         if path == '':
            break
         else:
            paths.append(path.rstrip())
   names = [path.split('/')[-1] + '_' for path in paths]

   should_load = was_created([save_path + '/' + name for name in names], per_train, per_dev, per_test)
   
   inds = []
   inds_initialized = False
   for name, path in zip(names, paths):
      if not should_load:
         print('Reading data from %s...' % path)
         data = read_data(path, should_filter, keep = keep)
         data = break_data(data)
         if not inds_initialized:
            for c in data:
               inds_temp = np.arange(c.shape[0])
               np.random.shuffle(inds_temp)
               inds.append(inds_temp)
            inds_initialized = True
         data = [c[inds[i]] for i, c in enumerate(data)]
      else:
         data = None # only because we need to send something to split_data()
      print('Splitting %s set...' % name)
      split_data(data, per_train = per_train, per_dev = per_dev, per_test = per_test, path = save_path, data_name = name, should_shuffle = False, should_load = should_load, verbos = verbos)

def get_dataset(path):
   """
   This function simply loads a dataset from the path and returns it. It is assumed that create_datasets() was already executed
   """
   print('Loading data set from %s...' % (path))
   train = np.load(path + 'train.npy')
   dev = np.load(path + 'dev.npy')
   test = np.load(path + 'test.npy')

   data = np.concatenate((train, dev, test))
   num_clusters = data.shape[0]
   num_wfs = count_waveforms(data)
   print_data_stats(train, 'train', num_clusters, num_wfs)
   print_data_stats(dev, 'dev', num_clusters, num_wfs)
   print_data_stats(test, 'test', num_clusters, num_wfs)

   return train, dev, test

def take_partial_data(data, start, end):
   """
   The function recieves data which is a list with three numpy arrays: 1) clusters with pyramidal label, 2) clusters with
   interneuron label and 3) unlabeled clusters. It returns a numpy array of clusters consisting of all parts of the data
   made from the start to end percentiles of the original elements of the data.
   """
   len0 = len(data[0])
   len1 = len(data[1])
   len2 = len(data[2])
   ret = np.concatenate((data[0][math.floor(start * len0) : math.floor(end * len0)],
      data[1][math.floor(start * len1) : math.floor(end * len1)],
      data[2][math.floor(start * len2) : math.floor(end * len2)]))
   return ret
   
def split_data(data, per_train = 0.6, per_dev = 0.2, per_test = 0.2 , path = '../data_sets', should_load = True, data_name = '', should_shuffle = True, verbos = False):
   """
   This function recieves the data as an ndarray. The first level is the different clusters, i.e each file,
   the second level is the different waveforms whithin each clusters and the third is the actual features (with the label)
   The function splits the entire data randomly to train, dev and test sets according to the given precentage.
   It is worth mentioning that although the number of clusters in each set should be according to the function's arguments
   the number of waveforms in each set is actually distributed independently.
   """
   assert per_train + per_dev + per_test == 1
   name = data_name + str(per_train) + str(per_dev) + str(per_test) + '/'
   full_path = path + '/' + name if path != None else None
   if path != None and os.path.exists(full_path) and should_load:
      print('Loading data set from %s...' % (full_path))
      train = np.load(full_path + 'train.npy')
      dev = np.load(full_path + 'dev.npy')
      test = np.load(full_path + 'test.npy')
   else:
      per_dev += per_train
      
      if should_shuffle:
         data = break_data(data)
         [np.random.shuffle(d) for d in data]
         
      train = take_partial_data(data, 0, per_train)
      dev = take_partial_data(data, per_train, per_dev)
      test = take_partial_data(data, per_dev, 1)

      if path != None:
         try:
            if not os.path.exists(full_path):
               os.mkdir(full_path)
         except OSError:
            print ("Creation of the directory %s failed, not saving set" % full_path)
         else:
            print ("Successfully created the directory %s now saving data set" % full_path)
            np.save(full_path + 'train', train)
            np.save(full_path + 'dev', dev)
            np.save(full_path + 'test', test)

   if verbos:
      data = np.concatenate((train, dev, test))
      num_clusters = data.shape[0]
      num_wfs = count_waveforms(data)
      print_data_stats(train, 'train', num_clusters, num_wfs)
      print_data_stats(dev, 'dev', num_clusters, num_wfs)
      print_data_stats(test, 'test', num_clusters, num_wfs)
   
   return train, dev, test

def print_data_stats(data, name, total_clusters, total_waveforms):
   """
   This function prints various statistics about the given set
   pyr == pyramidal ; in == interneuron; ut == untagged ; wfs == waveforms ; clstr == cluster
   """
   if len(data) == 0:
      print('No examples in %s set' % name)
      return
   num_clstr = data.shape[0]
   num_wfs = count_waveforms(data)
   clstr_ratio = num_clstr / total_clusters
   wfs_ratio = num_wfs / total_waveforms
   print('Total number of clusters in %s data is %d (%.3f%%) consisting of %d waveforms (%.3f%%)'
         % (name, num_clstr, 100 * clstr_ratio, num_wfs, 100 * wfs_ratio))

   pyr_clstrs = data[get_inds(data, 1)]
   num_pyr_clstr = pyr_clstrs.shape[0]
   ratio_pyr_clstr = num_pyr_clstr / num_clstr
   num_pyr_wfs = count_waveforms(pyr_clstrs)
   pyr_wfs_ratio = num_pyr_wfs / num_wfs
   print('Total number of pyramidal clusters in %s data is %d (%.3f%%) consisting of %d waveforms (%.3f%%)'
      % (name, num_pyr_clstr, 100 * ratio_pyr_clstr, num_pyr_wfs, 100 * pyr_wfs_ratio))
   
   in_clstrs = data[get_inds(data, 0)]
   num_in_clstr = in_clstrs.shape[0]
   ratio_in_clstr = num_in_clstr / num_clstr
   num_in_wfs = count_waveforms(in_clstrs)
   in_wfs_ratio = num_in_wfs / num_wfs
   print('Total number of interneurons clusters in %s data is %d (%.3f%%) consisting of %d waveforms (%.3f%%)'
      % (name, num_in_clstr, 100 * ratio_in_clstr, num_in_wfs, 100 * in_wfs_ratio))
   
   ut_clstrs = data[get_inds(data, -1)]
   num_ut_clstr = ut_clstrs.shape[0]
   ratio_ut_clstr = num_ut_clstr / num_clstr
   num_ut_wfs = count_waveforms(ut_clstrs)
   ut_wfs_ratio = num_ut_wfs / num_wfs
   print('Total number of untagged clusters in %s data is %d (%.3f%%) consisting of %d waveforms (%.3f%%)'
      % (name, num_ut_clstr, 100 * ratio_ut_clstr, num_ut_wfs, 100 * ut_wfs_ratio))        

def get_inds(data, label):
   """
   The function recieved a numpy array of clusters (numpy arrays of varying sizes) and returns
   the indeces with the given label. If the label is -1 all clusters with negative (i.e. untagged)
   labels indices are returned.
   This function is needed as numpy has hard time working with varying size arrays.
   """
   inds = []
   for ind, cluster in enumerate(data):
      if label >= 0:
         if cluster[0, -1] == label:
            inds.append(ind)
      else:
         if cluster[0, -1] < 0:
            inds.append(ind)
   return inds

def count_waveforms(data):
   """
   This function counts the number of waveforms in all clusters of the data.
   The main usage of this function is statistical data gathering.
   """
   counter = 0
   for cluster in data:
      counter += cluster.shape[0]
   return counter

def squeeze_clusters(data):
   """
   This function receives an nd array with elements with varying sizes.
   It removes the first dimension.
   As numpy doesn't nicely support varying sizes we implement what otherwise could have been achieved using reshape or squeeze 
   """
   res = []
   for cluster in data:
      for waveform in cluster:
         res.append(waveform)
   return np.asarray(res)

