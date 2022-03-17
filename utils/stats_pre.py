import pandas as pd
import numpy as np
import os

PATH = '../ml/results_rf_region_corrected.csv'
EXT_FETS_PATH = '../clustersData_no_light_FINAL_new/0'
DEST = '../statistics/results_rf_region_proc_only0.csv'

def read_data(path):
    """
   The function reads the data from all files in the path.
   It is assumed that each file represents a single cluster, and have some number of waveforms.
   The should_filter (optional, bool, default = True) argument indicated whether we should filter out
   clusters with problematic label (i.e. < 0)
   """
    files = os.listdir(path)
    clusters = []
    for i, file in enumerate(sorted(files)):
        df = pd.read_csv(path + '/' + file)
        region = df.region[0]
        df = df.drop(columns=['name', 'region'])
        nd = df.to_numpy(dtype='float64')[0]

        if nd[-1] < 0 or region in [0, 1]:
            clusters.append(nd)
        else:
            continue
    cols = df.columns
    return np.asarray(clusters), cols


if __name__ == '__main__':
    df = pd.read_csv(PATH, index_col=0)

    drop = [col for col in df.columns.values if col not in ['modality', 'chunk_size', 'seed', 'acc', 'auc']]
    drop_dev = [col for col in df.columns.values if col not in ['modality', 'chunk_size', 'seed', 'dev_acc', 'dev_auc']]

    ncx = df.drop(columns=drop)
    ca1 = df.drop(columns=drop_dev).rename(columns={'dev_acc': 'acc', 'dev_auc': 'auc'})

    ncx['region'] = 'ncx'
    ca1['region'] = 'ca1'

    ret = ncx.append(ca1)
    ret = ret[ret.chunk_size == 0]
    ret = ret.drop(columns=['chunk_size'])

    ret.to_csv(DEST, index=False)

    """data, cols = read_data(EXT_FETS_PATH)
    df = pd.DataFrame(data, columns=cols)
    df.to_csv(DEST, index=False)"""



