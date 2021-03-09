import numpy as np
import scipy.io
from ml.ML_util import is_legal, read_data

def get_labels(clusters):
    labels = []
    for cluster in clusters:
        labels.append(cluster[0, -1])
    return np.asarray(labels)

def print_statistics_clusters(labels):
    total = len(labels)
    pyrs = len(labels[labels == 1])
    ins = len(labels[labels == 0])
    uts = len(labels[labels == -1])
    doubles = len(labels[labels == -3])

    print('There is a total of %d clusters' % total)
    print('There are %d pyramidal cells in the data which is %.3f%%' % (pyrs, 100 * pyrs / total))
    print('There are %d interneurons cells in the data which is %.3f%%' % (ins, 100 * ins / total))
    print('There are %d double tagged cells in the data which is %.3f%%' % (doubles, 100 * doubles / total))
    print('There are %d untagged cells in the data which is %.3f%%' % (uts, 100 * uts / total))

def counter_one(array):
    count = 0
    for elem in array:
        if elem[0] == 1:
            count += 1
    return count

def counter_two(array1, array2):
    count = 0
    for elem1, elem2 in zip(array1, array2):
        if elem1[0] == 1 and elem2[0] == 1:
            count += 1
    return count

def print_statistics_spv():
    cellClassMat = scipy.io.loadmat("Data\\CelltypeClassification.mat")['sPV']
    options = ['act', 'exc', 'inh']
    total = len(cellClassMat[options[0]][0][0])
    for field in options:
        count = counter_one(cellClassMat[field][0][0])
        print('There are %d cells with %s tag which is %.3f%%' % (count, field, 100 * count / total))

    for i, field1 in enumerate(options):
        for field2 in options[i + 1:]
            count = counter_two(cellClassMat[field1][0][0], cellClassMat[field2][0][0])
            print('There are %d cells with %s and %s tags which is %.3f%%' % (count, field1, field2, 100 * count / total)) 
        

def run():
    clusters = read_data('clustersData/', should_filter = False)
    labels = get_labels(clusters)
    print_statistics_clusters(labels)
    print_statistics_spv()


if __name__ == "__main__":
    run()
