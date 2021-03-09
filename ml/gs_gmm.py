from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import numpy as np
from warnings import warn
import argparse

import ML_util
from feature_dropping_svm import remove_features

N = 100 # number of time to repeat each configuration in the grid search

def metric(model, features, labels, verbos = False):
    total = len(features)
    total_pyr = total_in = 0
    model_clusters = {}
    errors = 0
    pyr_clusters = in_clusters = 0
    total_pyr += len(labels[labels == 1])
    total_in += len(labels[labels == 0])
    preds = model.predict(features)
    for i, pred in enumerate(preds):
        if pred not in model_clusters:
            model_clusters[pred] = [labels[i]]
        else:
            model_clusters[pred].append(labels[i])
    for ind, cluster in enumerate(model_clusters):
        labels = model_clusters[cluster]
        labels = np.asarray(labels)
        total = labels.shape[0]
        counter_pyr = len(labels[labels == 1])
        counter_in = len(labels[labels == 0])
        counter_ut = len(labels[labels < 0])
        if verbos:
            print('In cluster %d there are %d examples, of which %d are pyramidal (%.2f), %d are interneurons (%.2f) and %d are untagged (%.2f)' %
                  (ind + 1, total, counter_pyr, 100 * counter_pyr / total,  counter_in, 100 * counter_in / total, counter_ut,
                   100 * counter_ut / total))
        label = 0 if counter_in >= counter_pyr else 1 if counter_pyr != 0 else -1
        pyr_clusters += label
        in_clusters += 1 - label
        errors += counter_pyr if label == 0 else counter_in if label == 1 else 0

    if verbos:
        print()
        print('Total of %d pyramidal cluster(s) and %d interneuron cluster(s)' % (pyr_clusters, in_clusters))
        print('Total number of errors is %d, the average error is %.3f' % (errors, errors / len(model_clusters)))
        
    return errors, errors / len(model_clusters)
        
def find_elbow(curve):
    """
    This function finds the elbow of the curve
    code adapted from https://stackoverflow.com/questions/2018178/finding-the-best-trade-off-point-on-a-curve
    """
    nPoints = len(curve)
    allCoord = np.vstack((np.arange(nPoints), curve)).T
    np.array([range(nPoints), curve])
    firstPoint = allCoord[0]
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec ** 2))
    vecFromFirst = allCoord - firstPoint
    scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis = 1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis = 1))
    idxOfBestPoint = np.argmax(distToLine)
    return idxOfBestPoint
                   

def run(use_tsne, tsne_n_components, use_scale, use_pca, pca_n_components, use_ica, ica_n_components,
        dataset_path, verbos, saving_path, min_search_components, max_search_components, n):
    """
    GMM grid search function, see help for explanations
    """

    data, _, _ = ML_util.get_dataset(dataset_path)

    pca = None
    scaler = None

    # squeeze the data and separate features and labels
    data_squeezed = ML_util.squeeze_clusters(data)
    data_features, data_labels = ML_util.split_features(data_squeezed)

    if use_pca and use_ica:
        raise Exception('Using both PCA and ICA is not allowed')

    # scale data if needed
    if use_scale or use_pca or use_ica:
        print('Scaling data...')
        scaler = StandardScaler()
        scaler.fit(data_features)
        data_features = scaler.transform(data_features)

    # if we need to use PCA dimension reduction, we will fit PCA and transform the data
    if use_pca: 
        if pca_n_components == None:
            raise Exception('If use_pca is True, pca_n_components must be specified')
        if pca_n_components > data_features.shape[1]:
            raise Exception('Number of required components is larger than the number of features')
        if not use_scale:
            warn('When Using PCA data will be scaled even if use_scale is set to False')
        pca = PCA(n_components = pca_n_components, whiten = True)
        print('Fitting PCA and transformming data...')
        data_features = pca.fit_transform(data_features)
        print('explained variance by PCA components is: ' + str(pca.explained_variance_ratio_))

    # if we need to use ICA dimension reduction, we will fit ICA and transform the data
    if use_ica: 
        if ica_n_components == None:
            raise Exception('If use_ica is True, ica_n_components must be specified')
        if ica_n_components > data_features.shape[1]:
            raise Exception('Number of required components is larger than the number of features')
        if not use_scale:
            warn('When Using ICA data will be scaled even if use_scale is set to False')
        ica = FastICA(n_components = ica_n_components, whiten = True)
        print('Fitting ICA and transformming the data...')
        data_features = ica.fit_transform(data_features)

    # if we need to use TSNE dimension reduction, we will fit TSNE and transform the data
    if use_tsne: 
        if tsne_n_components == None:
            raise Exception('If use_tsne is True but no loading path is given, tsne_n_components must be specified')
        if tsne_n_components > data_features.shape[1]:
            raise Exception('Number of required components is larger than the number of features')
        print('Fitting and transforming data using tSNE...')
        data_vis = TSNE(n_components = tsne_n_components, perplexity = 30, verbose = 1, n_iter = 2000).fit_transform(data_features)
        if verbos:
            plt.scatter(data_vis[data_labels < 0, vis_tsne_feature1], data_vis[data_labels < 0, vis_tsne_feature2], label = 'unlabeled')
            plt.scatter(data_vis[data_labels == 1, vis_tsne_feature1], data_vis[data_labels == 1, vis_tsne_feature2], label = 'pyramidal')
            plt.scatter(data_vis[data_labels == 0, vis_tsne_feature1], data_vis[data_labels == 0, vis_tsne_feature2], label = 'interneuron')
            plt.legend()
            plt.ylabel('tSNE feature 1')
            plt.xlabel('tSNE feature 2')
            plt.title('tSNE Representation')
            plt.show()

    # define the parameters we will check
    n_components_options = np.arange(min_search_components, max_search_components + 1)
    covariance_type_options = ['full', 'tied', 'diag', 'spherical']

    # define variables to hold data from the grid search
    bics = []

    # actual grid search
    print('Starting grid search...')
    for covariance_type in covariance_type_options:
        bics_temp = []
        for n_components in n_components_options:
            bic = 0
            for i in range(n):
                clst = GaussianMixture(n_components = n_components, covariance_type = covariance_type, reg_covar = 0.01, max_iter = 200)
                clst.fit(data_features)
                bic += clst.bic(data_features) / n # get bic score, averaged for n repetiotions
            bics_temp.append(bic)
        bics.append(bics_temp)

        # evaluate best n_components for the specific covariance_type
        best_n_components = find_elbow(bics_temp) + 2
        print('Best BIC with %s covariance type was %.3f, achieved with %d components' % (covariance_type, bics_temp[best_n_components - 2], best_n_components))
        best_bic = float('inf')
        avg_errors = 0
        best_model = None
        for i in range(2 * n):
            clst = GaussianMixture(n_components = best_n_components, covariance_type = covariance_type, reg_covar = 0.01)
            clst.fit(data_features)
            errors, _ = metric(clst, data_features, data_labels, verbos = False)
            avg_errors += errors / (2 * n)
            clst_bic = clst.bic(data_features)
            if  clst_bic < best_bic:
                best_model = clst
                best_bic = clst_bic
        print('Average number of errors for the model is: %.2f' % avg_errors)
        print('Best model BIC score is: %.2f' % best_bic)
        metric(best_model, data_features, data_labels, verbos = verbos)
        print()

        # create and save a plot showing the change of BIC by n_components
        plt.clf()
        plt.plot(n_components_options, bics_temp)
        plt.xticks(n_components_options)
        title = 'BIC score by n_components with ' + covariance_type + ' covariance type'
        plt.title(title)
        plt.xlabel('n_components')
        plt.ylabel('BIC')
        plt.xticks(n_components_options)
        plt.savefig(saving_path + title + '_' + str(use_pca) + '_' + str(use_ica))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="GMM grid search\n")
    
    parser.add_argument('--use_tsne', type=bool, help='use tSNE for visualization', default = False)
    parser.add_argument('--tsne_n_components', type=int, help='number of tSNE components', default = 2)
    parser.add_argument('--use_scale', type=bool, help='scale the data', default = False)
    parser.add_argument('--use_pca', type=bool, help='transform the data using PCA', default = False)
    parser.add_argument('--pca_n_components', type=int, help='number of PCA components', default = 2)
    parser.add_argument('--use_ica', type=bool, help='transform the data using ICA', default = False)
    parser.add_argument('--ica_n_components', type=int, help='number of ICA components', default = 2)
    parser.add_argument('--dataset_path', type=str, help='path to the dataset, assume it was created', default = '../data_sets/0_1.00.00.0/')
    parser.add_argument('--verbos', type=bool, help='verbosity level (bool)', default = True)
    parser.add_argument('--saving_path', type=str, help='path to save graphs, assumed to be created', default = '../graphs/')
    parser.add_argument('--min_search_components', type=int, help='minimal number of gmm components to check', default = 2)
    parser.add_argument('--max_search_components', type=int, help='maximal number of gmm components to check', default = 20)
    parser.add_argument('--n', type=int, help='number of repetitions per configuration', default = N)


    args = parser.parse_args()
    
    use_tsne = args.use_tsne
    tsne_n_components = args.tsne_n_components
    use_scale = args.use_scale
    use_pca = args.use_pca
    pca_n_components = args.pca_n_components
    use_ica = args.use_ica
    ica_n_components = args.ica_n_components
    dataset_path = args.dataset_path
    verbos = args.verbos
    saving_path = args.saving_path
    min_search_components = args.min_search_components
    max_search_components = args.max_search_components
    n = args.n
    
    run(use_tsne, tsne_n_components, use_scale, use_pca, pca_n_components, use_ica, ica_n_components, dataset_path,
        verbos, saving_path, min_search_components, max_search_components, n)
