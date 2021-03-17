import sklearn.ensemble
import sklearn.manifold
from scipy.stats import describe
from joblib import Parallel, delayed
from sklearn.model_selection import *
from sklearn.ensemble import RandomForestClassifier
import scipy.io
from runGMM import *


# Global variables
leafs = np.nan
realLeafs = np.nan


def return_synthetic_data(X):
    """
    The function returns a matrix with the same dimensions as X but with synthetic data
    based on the marginal distributions of its features
    """
    features = len(X[0])
    X_syn = np.zeros(X.shape)

    for i in range(features):
        obs_vec = X[:,i]
        syn_vec = np.random.choice(obs_vec, len(obs_vec)) # here we chose the synthetic data to match the marginal distributions of the real data
        X_syn[:,i] += syn_vec

    return X_syn


def merge_work_and_synthetic_samples(X_real, X_syn):

    """
    The function merges the data into one sample, giving the label "1" to the real data and label "2" to the synthetic data
    """

    # build the labels vector
    Y = np.ones(len(X_real))
    Y_syn = np.ones(len(X_syn)) * 2

    Y_total = np.concatenate((Y, Y_syn)) # Classes vector
    X_total = np.concatenate((X_real, X_syn)) # Merged array
    return X_total, Y_total


def calcDisMat(rf, X_real):

    """
    :param rf: Random forest classifier object
    :param X_real: (Number of obs. ,Number of feat.) Matrix
    :return: (Number of obs., Number of obs.) Distance matrix (1 - Similarity matrix)
    *** Parallel Version ***
    The function builds the similarity matrix for the real observations X_syn based on the random forest we've trained.
    The matrix is normalized so that the highest similarity is 1 and the lowest is 0
    This function counts only leafs in which the object is classified as a "real" object,
    meaning that more than half of the observations at a leaf are "real".
    """

    if 0 == 0:

        # Number of Observations
        n = len(X_real)

        # leafs: A row represents a unit, a column represents a tree (estimator).
        #        A cell's value is the index of the leaf where a unit ended up at a specific tree.
        print ('Applying forest to results... ')
        global leafs
        global realLeafs
        leafs = np.array(rf.apply(X_real), dtype=np.int16)

        # Make sure we are not out of the int16 limits
        print('Max leaf index: ',)
        print(np.max(leafs))

        # realLeafs: A row represents a galaxy, a column represents a tree (estimator).
        #            A cell's value is True if most of the galaxies are real, and False otherwise.
        print('Calculating real leafs... ')
        # Same structure as leafs: 1 if real, 0 else
        estList = rf.estimators_
        n_est = len(estList)

        realLeafs = Parallel(n_jobs=-1, verbose=1)(
            delayed(realLeaf_i)(estList[i].tree_.value, leafs[:, i]) for i in range(n_est))
        realLeafs = np.array(realLeafs, dtype=bool).T

        # We calculate the similarity matrix based on leafs & realLeafs
        print('Calculating similarity matrix... ')
        sim_mat = np.zeros((n, n), dtype=np.float16)

        # It is suggested to run the same parallelization using the multiprocessing package and test which is faster
        # pool = mp.Pool()
        # f = partial(dist_i_mp)
        # res = pool.map(dist_i_mp, range(n))
        res = Parallel(n_jobs=-1, verbose=1)(delayed(dist_i)(i, leafs[i:, :], leafs[i], realLeafs[i:, :], realLeafs[i]) for i in range(n))

        del leafs, realLeafs
        for tup in res:
            i = tup[1]
            sim_mat[i:, i] = tup[0]

        # Symmetrisize the similarity matrix
        sim_mat  = np.array(sim_mat, dtype=np.float16)
        sim_mat += sim_mat.T - np.diag(np.diagonal(sim_mat))

        # dissimilarity=1-similarity (Dij=1-Sij)
        return 1 - sim_mat


# @jit
def realLeaf_i(val, cleafs):
    cRealLeafs = np.zeros(cleafs.shape)
    for i in range(len(cleafs)):
        cleaf = val[cleafs[i]][0]
        cRealLeafs[i] = cleaf[0] > cleaf[1]
    return np.array(cRealLeafs, np.int8)


def dist_i(i, a, b, c, d):
    """
    :param i: int: The index of the galaxy
    :param a: leafs[i:, :]
    :param b: leafs[i]
    :param c: realLeafs[i:, :]
    :param d: realLeafs[i]
    :return: Returns the last (n-i) rows of the ith column in the similarity matrix
    """

    # Count the number of leafs where galaxies are at the same leafs (a==b) and classified as real (c)
    mutLeafs = np.logical_and((a == b), c).sum(1)

    # ount the number of leafs where galaxies are classified as real
    mutRealLeafs = np.logical_and(d, c).sum(1, dtype=np.float64)

    return np.divide(mutLeafs, mutRealLeafs, dtype=np.float16), i


def dist_i_mp(i):
    """
    It is compatible with the multiprocessing commented out in the calcDisMat function.
    :param i: int: The index of the galaxy
    :return: Returns the last (n-i) rows of the ith column in the similarity matrix.
    """
    mutLeafs = np.logical_and((leafs[i:, :] == leafs[i]), realLeafs[i:, :], dtype=np.int16).sum(1)
    mutRealLeafs = np.logical_and(realLeafs[i], realLeafs[i:, :], dtype=np.int16).sum(1, dtype=np.float64)

    return np.divide(mutLeafs, mutRealLeafs, dtype=np.float16), i


def get_leaf_sizes(rf, X):

    """
    :param rf: Random forest classifier object
    :param X: The data we would like to classify and return its leafs' sizes
    :return: leafs_size: A list with all of the leafs sizes for each leaf an observation at X ended up at
    """

    apply_mat = rf.apply(X)
    leafs_sizes = []
    for i, est in enumerate(rf.estimators_):
        real_leafs = np.unique(apply_mat[:, i])
        for j in range(len(real_leafs)):
            leafs_sizes.append(est.tree_.value[real_leafs[j]][0].sum())
    return leafs_sizes


def make_clickable(val):
    return '<a href="{}">{}</a>'.format(val, val)


def seriation(Z, N, cur_index):
    '''
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z

        seriation computes the order implied by a hierarchical tree (dendrogram)
    '''
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index - N, 0])
        right = int(Z[cur_index - N, 1])
        return (seriation(Z, N, left) + seriation(Z, N, right))


def find_best_prams(X, y):
    clf = RandomForestClassifier(n_estimators=100, n_jobs=1)
    cv = ShuffleSplit(n_splits=10, test_size=0.4)
    param_grid = {'max_depth': [4, 6, 8, 10, 12, 14, 16], 'min_samples_split': [2, 5, 8, 11, 14, 17, 20]}

    gs = GridSearchCV(clf, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=5)
    gs.fit(X, y) # Use all data, not just trains - it uses cross-validation anyway

    print(gs.best_score_)
    print(gs.best_params_) # Then use these parameters to train the real classifier

    return gs

def plot_sne(dist_mat, labels, title , i,isTag=False):
    sne = sklearn.manifold.TSNE(n_components=2, perplexity=70, metric='precomputed', verbose=1, n_iter=2000).fit_transform(dist_mat)
    # print("kl_divergence_: ", kl_divergence_)
    sne_f1 = sne[:, 0]
    sne_f2 = sne[:, 1]

    scipy.io.savemat("tsne_%d.mat" % i, mdict={'out': sne}, oned_as='row')

    plt.figure(figsize=(7, 7))
    plt.scatter(sne_f1, sne_f2, s=3)
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.title('t-SNE Scatter Plot')
    plt.show()

    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(111)
    sne_f1_0 = sne_f1[labels == 0]
    sne_f1_1 = sne_f1[labels == 1]
    sne_f1_2 = sne_f1[labels == 2]
    sne_f2_0 = sne_f2[labels == 0]
    sne_f2_1 = sne_f2[labels == 1]
    sne_f2_2 = sne_f2[labels == 2]

    if isTag:
        ax1.scatter(sne_f1_0, sne_f2_0, c='tab:blue', label="act or inh", marker='.', s=3)
        ax1.scatter(sne_f1_1, sne_f2_1, c='tab:red', label="exc", marker='.', s=3)
    else:
        ax1.scatter(sne_f1_0, sne_f2_0, c='tab:blue',  label="Un tag", marker='.', s=3)
        ax1.scatter(sne_f1_1, sne_f2_1, c='tab:red', label="exc", marker='.', s=3)
        ax1.scatter(sne_f1_2, sne_f2_2, c='tab:orange', label="act or inh", marker='.', s=3)

    ax1.set_xlabel('t-SNE Feature 1')
    ax1.set_ylabel('t-SNE Feature 2')
    ax1.set_title(title)

    ax1.legend()
    plt.show()



def sort_list(list1, list2):
    '''
    this function return list1 sorted by the order of list2
    '''
    zipped_pairs = zip(list2, list1)
    z = [x for _, x in sorted(zipped_pairs)]

    return z


def calc_random_forest(X, max_depth, min_samples_split):
    X_no_nan = np.nan_to_num(X)
    # Create synthetic data
    X_syn = return_synthetic_data(X_no_nan)

    # Merge real and synthetic data and label them
    X, y = merge_work_and_synthetic_samples(X_no_nan, X_syn)

    # find_best_prams(X, y)

    # Find the best parms for future running
    # gs = find_best_prams(X, y)

    # Set RF hyperparameters
    n_est = 100

    # rf = sklearn.ensemble.RandomForestClassifier(n_estimators=100, n_jobs=1,max_depth= 10, min_samples_split = 2) ## 5,12
    
    rf = sklearn.ensemble.RandomForestClassifier(n_estimators=n_est,
                                                 min_samples_split=min_samples_split,
                                                 n_jobs=-1,
                                                 verbose=1,
                                                 max_depth=max_depth)

    # Fit RF classifier
    rf.fit(X, y)

    real_prob = rf.predict_proba(X_no_nan)[:, 0]
    print('Mean probability of a real object to be classified as real: %.3f' % np.mean(real_prob))
    syn_prob = rf.predict_proba(X_syn)[:, 1]
    print('Mean probability of a synthetic object to be classified as synthetic : %.3f' % np.mean(syn_prob))
    # Calculate trees' depths
    trees_depths = [est.tree_.max_depth for est in rf.estimators_]
    # Print Statistics
    print(describe(trees_depths))
    # Plot histogram
    plt.figure(figsize=(7, 7))
    # plt.hist(trees_depths, bins=n_est / 10)
    plt.hist(trees_depths, bins=10)
    plt.title('Trees Depths Histogram')
    plt.xlabel('Tree Depth')
    plt.ylabel('Count')
    plt.show(block=False)
    # Calculate leafs' sizes
    leafs_sizes = get_leaf_sizes(rf, X_no_nan)
    # Print Statistizs
    print(describe(leafs_sizes))
    # Plot histogram
    plt.figure(figsize=(7, 7))
    plt.hist(leafs_sizes)
    # plt.hist(leafs_sizes, n_est * (2 * len(gs)) / 3000)
    plt.title('Leaf Sizes Histogram')
    plt.xlabel('Leaf Size')
    plt.ylabel('Count')
    plt.show(block=False)

    return rf, X_no_nan



# def plot_all_methods(dist_mat, s):
#     fig = plt.figure()
#
#     methods = ["ward", "single", "average", "complete"]
#     for i, method in enumerate(methods):
#         print("Method:\t", method)
#         fig.add_subplot(2, 2, 1+i)
#         ordered_dist_mat, res_order, res_linkage = compute_serial_matrix(dist_mat, method)
#         labels_by_re_order = sort_list(s, res_order)
#         plt.pcolormesh(ordered_dist_mat)
#         plt.title(method)
#         plt.colorbar()
#         plt.xlim([0, N])
#         plt.ylim([0, N])
#         t = range(dist_mat.shape[0])
#         plt.scatter(t, t, c=labels_by_re_order, cmap='Set1')
#         # true_matrix = (ordered_dist_mat < 0.4)*1
#         # z = np.copy(true_matrix)
#         # num_of_island = numIslands(z)
#         # print(num_of_island)
#         # max_of_s_array, max_i_array, max_j_array,max_i, max_j = printMaxSubSquare(true_matrix)
#         # print((max_i_array))
#     plt.show()


def run_supervise():
    features, X, X_tag, X_no_tag, labels, features_names, tag_index, all_labels = load_data()

    # rf, X_process = calc_random_forest(X_tag)
    # feature1= 6
    # feature2 = 7
    # # rf, X_process = calc_random_forest(X_tag[:, [feature1, feature2]])
    # dist_mat = calcDisMat(rf, X_process)
    # scipy.io.savemat("dist_mat_supervise.mat", mdict={'out': dist_mat}, oned_as='row')

    matdata = scipy.io.loadmat("dist_mat_supervise.mat")
    dist_mat = matdata['out']

    # plot_all_methods(dist_mat, s)

    # plot_sne(dist_mat, np.array(labels),"t-SNE plot for tag supervise data. features: %s, %s" % (features_names[feature1],features_names[feature2]), True,)
    # plot_mult_sne(dist_mat, True,)


def run_unSupervise():
    features, X, X_tag, X_no_tag, labels, features_names, tag_index, all_labels = load_data()

    max_depth = [16]
    min_samples_split = [8]
    i = 0
    for depth in max_depth:
        for smpale in min_samples_split:
            rf, X_process = calc_random_forest(X, depth, smpale)
            dist_mat = calcDisMat(rf, X_process)
            plot_sne(dist_mat, all_labels, "t-SNE plot for tag Unsupervise data", i)
            i += 1

    # tsne_table = []
    # (fig, subplots) = plt.subplots(4, 4)
    # for k in range(16):
    #     i = int(k / 4)
    #     j = k % 4
    #     ax = subplots[i][j]
    #     matdata = scipy.io.loadmat("tsne_%d.mat" % k)
    #     sne = matdata['out']
    #     tsne_table.append(sne)
    #
    #     sne_f1 = sne[:, 0]
    #     sne_f2 = sne[:, 1]
    #
    #     sne_f1_0 = sne_f1[all_labels == 0]
    #     sne_f1_1 = sne_f1[all_labels == 1]
    #     sne_f1_2 = sne_f1[all_labels == 2]
    #     sne_f2_0 = sne_f2[all_labels == 0]
    #     sne_f2_1 = sne_f2[all_labels == 1]
    #     sne_f2_2 = sne_f2[all_labels == 2]
    #
    #     ax.scatter(sne_f1_0, sne_f2_0, c='black', label="un-tag", marker='.', s=3)
    #     ax.scatter(sne_f1_1, sne_f2_1, c='red', label="exc", marker='.', s=3)
    #     ax.scatter(sne_f1_2, sne_f2_2, c='blue', label="act or inh", marker='.', s=3)
    #     ax.set_title("max_depth: %d, min_samples_split: %d" %(max_depth[i], min_samples_split[j]))
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #
    #     # ax.legend()
    #
    # fig.tight_layout()
    # plt.show()


    tsne_table = []
    # n_clust = [4, 6, 6, 7, 5, 5, 4, 6, 6, 5, 6, 7, 7, 4, 5, 5]
    #
    # (fig, subplots) = plt.subplots(4, 4)
    # for k in range(16):
    #     i = int(k / 4)
    #     j = k % 4
    #     ax = subplots[i][j]
    #     matdata = scipy.io.loadmat("tsne_%d.mat" % k)
    #     sne = matdata['out']
    #     # tsne_table.append(sne)
    #
    #     sne_f1 = sne[:, 0]
    #     sne_f2 = sne[:, 1]
    #     # good_GMM(sne, ax)
    #     ax.set_title("N. of clustering %d" % n_clust[k])
    #     ax.set_xticks([])
    #     final_GMM_2d(sne, n_clust[k], 'full', 10, all_labels, ax)
    #
    # # plt.title("Gradient of BIC Scores for type full", fontsize=20)
    # plt.show()


    rf, X_process = calc_random_forest(X, 4, 15)
    dist_mat = calcDisMat(rf, X_process)
    scipy.io.savemat("dist_mat_final_@.mat", mdict={'out': dist_mat}, oned_as='row')

    plt.clf()
    N = len(dist_mat)
    plt.pcolormesh(dist_mat)
    plt.colorbar()
    plt.xlim([0, N])
    plt.ylim([0, N])
    plt.title('Dissimilarity matrix')
    plt.show()

    # matdata = scipy.io.loadmat("dist_mat_fianl.mat")
    # dist_mat = matdata['out']
    # plot_all_methods(dist_mat, s)

    plot_sne(dist_mat, all_labels, "t-SNE plot for tag Unsupervise data", 1000)

    # plot_mult_sne(dist_mat, all_labels, features_names)


def plot_mult_sne(dist_mat, labels, features_names , isTag=False):
    labels = np.array(labels)
    (fig, subplots) = plt.subplots(3, 5, figsize=(15, 8))

    perplexities = [20, 35, 50, 70, 100]
    n_iter = [1000, 2500, 5000]
    sne_vec = []
    param_vec = []
    for i, iter in enumerate(n_iter):
        for j, perp in enumerate(perplexities):
            print("running num: ", str((i) * (5) + j + 1))
            ax = subplots[i][j]
            sne = sklearn.manifold.TSNE(n_components=2, perplexity=perp, metric='precomputed', verbose=1, n_iter=iter,
                                            ).fit_transform(dist_mat)
            # plot_sne2(dist_mat, np.array(labels), "iter: %d, perplexity: %d" % (iter, perp), sne, ax, True)
            sne_vec.append(sne)
            param_vec.append((iter, perp))

            sne_f1 = sne[:, 0]
            sne_f2 = sne[:, 1]
            sne_f1_0 = sne_f1[labels == 0]
            sne_f1_1 = sne_f1[labels == 1]
            sne_f1_2 = sne_f1[labels == 2]
            sne_f2_0 = sne_f2[labels == 0]
            sne_f2_1 = sne_f2[labels == 1]
            sne_f2_2 = sne_f2[labels == 2]

            if isTag:
                ax.scatter(sne_f1_0, sne_f2_0, c='tab:blue', label="act or inh", marker='.', s=3)
                ax.scatter(sne_f1_1, sne_f2_1, c='tab:red', label="exc", marker='.', s=3)
            else:
                ax.scatter(sne_f1_0, sne_f2_0, c='tab:blue', label="un-tag", marker='.', s=3)
                ax.scatter(sne_f1_1, sne_f2_1, c='tab:red', label="exc", marker='.', s=3)
                ax.scatter(sne_f1_2, sne_f2_2, c='tab:orange', label="act or inh", marker='.', s=3)

            # ax1.set_xlabel('t-SNE Feature 1')
            # ax1.set_ylabel('t-SNE Feature 2')
            ax.set_title("iter: %d, perplexity: %d" % (iter, perp,))
            # ax.legend()
            # ax.xaxis.set_major_formatter(NullFormatter())
            # ax.yaxis.set_major_formatter(NullFormatter())
            ax.axis('tight')

    scipy.io.savemat("sne_vec.mat", mdict={'out': sne_vec}, oned_as='row')
    scipy.io.savemat("param_vec.mat", mdict={'out': param_vec}, oned_as='row')


    fig.tight_layout()
    plt.show()

    save_data = []
    save_data.append(sne_vec)
    save_data.append(param_vec)
    scipy.io.savemat("save_data.mat", mdict={'out': save_data}, oned_as='row')


def run_GMM_on_tSNE():
    features, X, X_tag, X_no_tag, labels, features_names, tag_index, all_labels = load_data()

    matdata = scipy.io.loadmat("tsne_1000.mat")
    tsne = matdata['out']
    # good_GMM(tsne)
    (fig, subplots) = plt.subplots(1, 1)

    plot_GMM_2d(tsne, 5, 'full', 10, all_labels, subplots)
    plt.show()




if __name__ == '__main__':
    features, X, X_tag, X_no_tag, labels, features_names, tag_index, all_labels = load_data()
    # calc_random_forest(X)
    # find_best_prams(X_tag, labels)
    run_unSupervise()
    # run_GMM_on_tSNE()
    # GMM_BIC(X, all_labels,all_labels)
    # run_GMM_on_tSNE()
    # hier_clust(X)
