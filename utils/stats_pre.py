import pandas as pd

PATH = '../ml/results_rf_region.csv'
DEST = '../statistics/results_rf_region_proc.csv'

if __name__ == '__main__':
    df = pd.read_csv(PATH, index_col=0)

    drop = [col for col in df.columns.values if col not in ['modality', 'chunk_size', 'seed', 'acc', 'auc']]
    drop_dev = [col for col in df.columns.values if col not in ['modality', 'chunk_size', 'seed', 'dev_acc', 'dev_auc']]

    ncx = df.drop(columns=drop)
    ca1 = df.drop(columns=drop_dev).rename(columns={'dev_acc': 'acc', 'dev_auc': 'auc'})

    ncx['region'] = 'ncx'
    ca1['region'] = 'ca1'

    ret = ncx.append(ca1)

    ret.to_csv(DEST, index=False)


