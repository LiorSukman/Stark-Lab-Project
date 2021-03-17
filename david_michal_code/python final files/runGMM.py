import pandas as pd
from sklearn import datasets
from sklearn.mixture import GaussianMixture
from runSVM import *
import itertools
from scipy import linalg
import matplotlib as mpl

from sklearn import mixture
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture as GMM
from matplotlib import rcParams
from sklearn.externals import joblib


# Output a pickle file for the model

def clustring_GMM(X,num_components):
    # select first two columns
    print("starting clustring")
    # turn it into a dataframe
    d = pd.DataFrame(X[:, [0, 10, 11]])
    # d = X
    # plot the data
    fig = plt.figure()
    ax = Axes3D(fig)

    # ax.scatter(d[2], d[7], d[9])
    # plt.scatter3(d[2], d[7], d[9])
    # plt.show()
    gmm = GaussianMixture(n_components=num_components)
    # print(d[7])
    # Fit the GMM model for the dataset
    # which expresses the dataset as a
    # mixture of 3 Gaussian Distribution
    gmm.fit(d)

    # Assign a label to each sample
    labels = gmm.predict(d)
    d['labels'] = labels

    # for i in range(num_components):
    #     d_labels = d[d['labels'] == i]
    #     plt.scatter(d_labels[0], d_labels[1], c='r')

    d0 = d[d['labels'] == 0]
    d1 = d[d['labels'] == 1]
    d2 = d[d['labels'] == 2]

    # plot three clusters in same plot
    plt.scatter(d0[0], d0[1], c='r')
    plt.scatter(d1[0], d1[1], c='yellow')
    plt.scatter(d2[0], d2[1], c='g')
    plt.show()
    # print the converged log-likelihood value
    print(gmm.lower_bound_)

    # print the number of iterations needed
    # for the log-likelihood value to converge
    print(gmm.n_iter_)


def learning_GMM_BIC(X):
    # Number of samples per component
    n_samples = 500

    # Generate random sample, two components
    # np.random.seed(0)
    C = np.array([[0., -0.1], [1.7, .4]])
    X1 = np.r_[np.dot(np.random.randn(n_samples, 2), C),
              .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]

    print("X1.shape: ",X1.shape)
    print("1.shape: ",X.shape)

    lowest_bic = np.infty
    bic = []
    n_components_range = range(1, 7)
    cv_types = ['spherical', 'tied', 'diag', 'full']
    for cv_type in cv_types:
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          covariance_type=cv_type)
            gmm.fit(X)
            bic.append(gmm.bic(X))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

    bic = np.array(bic)
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                                  'darkorange'])
    clf = best_gmm
    bars = []

    # Plot the BIC scores
    plt.figure(figsize=(8, 6))
    spl = plt.subplot(2, 1, 1)
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                      (i + 1) * len(n_components_range)],
                            width=.2, color=color))
    plt.xticks(n_components_range)
    plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    plt.title('BIC score per model')
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 + \
           .2 * np.floor(bic.argmin() / len(n_components_range))
    plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    spl.set_xlabel('Number of components')
    spl.legend([b[0] for b in bars], cv_types)

    # Plot the winner
    splot = plt.subplot(2, 1, 2)
    Y_ = clf.predict(X)
    for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
                                               color_iter)):
        v, w = linalg.eigh(cov)
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan2(w[0][1], w[0][0])
        angle = 180. * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(.5)
        splot.add_artist(ell)

    plt.xticks(())
    plt.yticks(())
    plt.title('Selected GMM: full model, 2 components')
    plt.subplots_adjust(hspace=.35, bottom=.02)
    plt.show()


# def GMM_2(X):
#     color_iter = itertools.cycle(['navy', 'c', 'cornflowerblue', 'gold',
#                                   'darkorange'])
#
#     def plot_results(X, Y_, means, covariances, index, title):
#         splot = plt.subplot(2, 1, 1 + index)
#         for i, (mean, covar, color) in enumerate(zip(
#                 means, covariances, color_iter)):
#             v, w = linalg.eigh(covar)
#             v = 2. * np.sqrt(2.) * np.sqrt(v)
#             u = w[0] / linalg.norm(w[0])
#             # as the DP will not use every component it has access to
#             # unless it needs it, we shouldn't plot the redundant
#             # components.
#             if not np.any(Y_ == i):
#                 continue
#             plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)
#
#             # Plot an ellipse to show the Gaussian component
#             angle = np.arctan(u[1] / u[0])
#             angle = 180. * angle / np.pi  # convert to degrees
#             ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
#             ell.set_clip_box(splot.bbox)
#             ell.set_alpha(0.5)
#             splot.add_artist(ell)
#
#         plt.xlim(-9., 5.)
#         plt.ylim(-3., 6.)
#         plt.xticks(())
#         plt.yticks(())
#         plt.title(title)
#
#     # Number of samples per component
#     # n_samples = 500
#     #
#     # # Generate random sample, two components
#     # np.random.seed(0)
#     # C = np.array([[0., -0.1], [1.7, .4]])
#     # X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
#     #           .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]
#
#     # Fit a Gaussian mixture with EM using five components
#     gmm = mixture.GaussianMixture(n_components=2, covariance_type='full').fit(X)
#     plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, 0,
#                  'Gaussian Mixture')
#
#     # Fit a Dirichlet process Gaussian mixture using five components
#     dpgmm = mixture.BayesianGaussianMixture(n_components=2,
#                                             covariance_type='full').fit(X)
#     plot_results(X, dpgmm.predict(X), dpgmm.means_, dpgmm.covariances_, 1,
#                  'Bayesian Gaussian Mixture with a Dirichlet process prior')
#
#     plt.show()


def run_BIC_GMM(X):

    rcParams['figure.figsize'] = 16, 8

    def draw_ellipse(position, covariance, ax=None, **kwargs):
        """Draw an ellipse with a given position and covariance"""
        ax = ax or plt.gca()
        # Convert covariance to principal axes
        if covariance.shape == (2, 2):
            U, s, Vt = np.linalg.svd(covariance)
            angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
            width, height = 2 * np.sqrt(s)
        else:
            angle = 0
            width, height = 2 * np.sqrt(covariance)

        # Draw the Ellipse
        for nsig in range(1, 4):
            ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                                 angle, **kwargs))

    def plot_gmm(gmm, X, label=True, ax=None):
        ax = ax or plt.gca()
        labels = gmm.fit(X).predict(X)
        if label:
            ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
        else:
            ax.scatter(X[:, 0], X[:, 1], s=40, zorder=2)

        w_factor = 0.2 / gmm.weights_.max()
        for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
            draw_ellipse(pos, covar, alpha=w * w_factor)
        plt.title("GMM with %d components" % len(gmm.means_), fontsize=(20))
        plt.xlabel("U.A.")
        plt.ylabel("U.A.")

    def SelBest(arr: list, X: int) -> list:
        '''
        returns the set of X configurations with shorter distance
        '''
        dx = np.argsort(arr)[:X]
        return arr[dx]

    # load out dataset
    embeddings = X

    # # BIC
    # n_clusters = np.arange(2, 20)
    # bics = []
    # bics_err = []
    # iterations = 20
    # for n in n_clusters:
    #     print(n)
    #     tmp_bic = []
    #     for _ in range(iterations):
    #         gmm = GMM(n, n_init=2).fit(embeddings)
    #
    #         tmp_bic.append(gmm.bic(embeddings))
    #     val = np.mean(SelBest(np.array(tmp_bic), int(iterations / 5)))
    #     err = np.std(tmp_bic)
    #     bics.append(val)
    #     bics_err.append(err)
    #
    #
    # plt.errorbar(n_clusters, bics, yerr=bics_err, label='BIC')
    # plt.title("BIC Scores", fontsize=20)
    # plt.xticks(n_clusters)
    # plt.xlabel("N. of clusters")
    # plt.ylabel("Score")
    # plt.legend()
    # plt.show()

    # BIC_better
    n_clusters = np.arange(2, 20)
    iterations = 20
    cv_types = ['full']
    # cv_types = ['spherical', 'tied', 'diag', 'full']
    by_type = []
    for cv_type in cv_types:
        bics = []
        bics_err = []
        for n in n_clusters:
            print(n)
            tmp_bic = []
            for _ in range(iterations):
                gmm = GMM(n, n_init=2, covariance_type=cv_type).fit(embeddings)

                tmp_bic.append(gmm.bic(embeddings))
            val = np.mean(SelBest(np.array(tmp_bic), int(iterations / 5)))
            err = np.std(tmp_bic)
            bics.append(val)
            bics_err.append(err)
        by_type.append((bics, bics_err))


    (fig, subplots) = plt.subplots(1, 1)

    for i, type in enumerate(cv_types):
        bics = by_type[i][0]
        bics_err = by_type[i][1]

        # ax = subplots[0][i]
        #
        # ax.errorbar(n_clusters, bics, yerr=bics_err, label='BIC')
        # ax.set_title("Gradient of BIC Scores for type %s" % (type), fontsize=10)
        # ax.set_xticks(n_clusters)
        # ax.set_xlabel("N. of clusters")
        # ax.set_ylabel("BIC")
        # ax.legend()

        # ax = subplots[0][i]
        ax = subplots
        ax.errorbar(n_clusters, np.gradient(bics), yerr=bics_err, label='BIC')
        ax.set_title("Gradient of BIC Scores for type %s" % (type), fontsize=10)
        ax.set_xticks(n_clusters)
        ax.set_xlabel("N. of clusters")
        ax.set_ylabel("grad(BIC)")
        ax.legend()

    plt.show()


def calc_error(true_labels, model_labels):
    num_of_groups = max(model_labels)
    error = 0
    for i in range(num_of_groups+1):
        group_label = true_labels[model_labels == i]
        group_hist = np.histogram(group_label, bins=np.arange(4))[0]
        if(group_hist[1] ==0 or group_hist[2] == 0):
            continue
        if(group_hist[1] < group_hist[2]):
            error += group_hist[1]
        else:
            error += group_hist[2]
    return error

def get_GMM_model(X, n_comp, best_type, n_init, true_labels, join_labels):

    clf = GMM(n_components=n_comp, covariance_type=best_type, n_init=n_init)
    clf.fit(X)
    label = clf.predict(X)
    error = calc_error(true_labels, label)
    join_labels.append(label)

    return error, clf, join_labels

def plot_GMM_2d(X, n_comp, best_type, n_init, true_labels):

    clf = GMM(n_components=n_comp, covariance_type=best_type, n_init=n_init)
    clf.fit(X)
    label = clf.predict(X)


    plt.scatter(X[:, 0], X[:, 1], c=label, s=16, lw=0, cmap='Paired')
    sne_f1 = X[:, 0]
    sne_f2 = X[:, 1]
    sne_f1_0 = sne_f1[true_labels == 0]
    sne_f1_1 = sne_f1[true_labels == 1]
    sne_f1_2 = sne_f1[true_labels == 2]
    sne_f2_0 = sne_f2[true_labels == 0]
    sne_f2_1 = sne_f2[true_labels == 1]
    sne_f2_2 = sne_f2[true_labels == 2]

    plt.scatter(sne_f1_1, sne_f2_1, c='red', label="exc", marker='.', s=3)
    plt.scatter(sne_f1_2, sne_f2_2, c='blue', label="act or inh", marker='.', s=3)
    error = calc_error(true_labels, label)
    plt.set_title("Chosen GMM" %(n_comp, error))
    plt.xlabel('x')
    plt.ylabel('y')
    num_of_groups = max(label)

    plt.show()
    # calc_error_dest(all_labels, label)
    return error, clf

def load_best_GMM_model(X):
    clf_load = joblib.load('GMM_model_full_7_test.pkl')
    labels = clf_load.predict(X)
    scipy.io.savemat("best_labels_full_7.mat", mdict={'out': labels}, oned_as='row')


if __name__ == '__main__':
    features, X, X_tag, X_no_tag, labels, features_names, tag_index, all_labels = load_data()
    learning_GMM_BIC(X)
    run_BIC_GMM(X)
    plot_GMM_2d(X, 7, 'full', 5, labels)

# code for calculate average error and saving the model with minimum error.
    errors = []
    clf_vec = []
    join_labels = []
    matdata = scipy.io.loadmat("tsne_1000.mat")
    tsne = matdata['out']
    join_labels.append(all_labels)
    for i in range(100):
        print("iter : %d" % (i))
        error, clf, join_labels = get_GMM_model(X, 7, 'full', 20, all_labels, join_labels)
        errors.append(error)
        clf_vec.append(clf)
    ind = np.argmin(errors)
    joblib.dump(clf_vec[ind], 'GMM_tSNE_model_full_4.pkl')

    join_labels = np.array(join_labels).T
    scipy.io.savemat("join_labels_7_full_new.mat", mdict={'out': join_labels}, oned_as='row')
    errors = np.array(errors)
    mean = errors.mean()
    print(mean)
    print(min(errors))

