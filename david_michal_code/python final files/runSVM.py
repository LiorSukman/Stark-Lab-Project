'''
==================
Run RBF SVM
==================



Intuitively, the ``gamma`` parameter defines how far the influence of a single
training example reaches, with low values meaning 'far' and high values meaning
'close'. The ``gamma`` parameters can be seen as the inverse of the radius of
influence of samples selected by the model as support vectors.

The ``C`` parameter trades off correct classification of training examples
against maximization of the decision function's margin. For larger values of
``C``, a smaller margin will be accepted if the decision function is better at
classifying all training points correctly. A lower ``C`` will encourage a
larger margin, therefore a simpler decision function, at the cost of training
accuracy. In other words``C`` behaves as a regularization parameter in the
SVM.

'''
# print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import scipy.io
from resultSVM import *
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from importData import get_data
from sklearn.externals import joblib
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split
# Utility function to move the midpoint of a colormap to be around
# the values of interest.

class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

# #############################################################################
# Load and prepare data set
#
# dataset for grid search
def load_data():
    features, X, X_tag, X_no_tag, labels, features_names, tag_index ,s = get_data()

    # It is usually a good idea to scale the data for SVM training.
    # We are cheating a bit in this example in scaling all of the data,
    # instead of fitting the transformation on the training set and
    # just applying it on the test set.
    # scaler = StandardScaler()
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X)

    X = scaler.fit_transform(X)
    X_tag = scaler.fit_transform(X_tag)
    X_no_tag = scaler.fit_transform(X_no_tag)

    scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X)
    features = scaling.transform(features.T)


    return features, X, X_tag, X_no_tag, labels,features_names, tag_index, s
#######################################################
# Chose the best features separator - the ones that return the best score in the test data

def find_best_features(X,y, features_names):
    num_of_features = X.shape[1]
    features = range(num_of_features)
    all_possible_pairs = [(features[i], features[j]) for i in range(len(features)) for j in range(i+1, len(features))]
    score_list = []
    grid_list = []
    X = np.nan_to_num(X)
    score_array = np.zeros((num_of_features, num_of_features))
    c_param_array = np.zeros((num_of_features, num_of_features))
    gamma_param_array = np.zeros((num_of_features, num_of_features))
    for pair in all_possible_pairs:
        i = pair[0]
        j = pair[1]
        print("Run SVM for features " + str(i) + ',' + str(j))
        grid = get_svm_best_parm(X[:, [i, j]], y, 10)
        score_array[i][j] = grid.best_score_
        c_param_array[i][j] = grid.best_params_['C']
        gamma_param_array[i][j] = grid.best_params_['gamma']
        score_list.append(grid.best_score_)
        grid_list.append(grid)
    scipy.io.savemat("score_array2.mat", mdict={'out': score_array}, oned_as='row')
    scipy.io.savemat("score_list2.mat", mdict={'out': score_list}, oned_as='row')

    print("Finish running for all the pairs")
    max_score = max(score_list)
    print("The best score is %f" % (max_score))
    best_score_index = [i for i, j in enumerate(score_list) if j == max_score]
    print(best_score_index)
    scipy.io.savemat("c_param_array2.mat", mdict={'out': c_param_array}, oned_as='row')
    scipy.io.savemat("gamma_param_array2.mat", mdict={'out': gamma_param_array}, oned_as='row')
    scipy.io.savemat("best_score_index2.mat", mdict={'out': best_score_index}, oned_as='row')

    for index in best_score_index:
        pair = all_possible_pairs[index]
        print("Pair2 with best score:" + str(pair))
        grid = grid_list[index]
        new_plot(grid.best_params_['C'], grid.best_params_['gamma'], X[:, [pair[0], pair[1]]], labels, pair, features_names, grid.best_score_)

    return best_score_index


# #############################################################################
# Train classifiers
#
# For an initial search, a logarithmic grid with basis
# 10 is often helpful. Using a basis of 2, a finer
# tuning can be achieved but at a much higher cost.
def find_best_features_trip(X, y, features_names):
    num_of_features = X.shape[1]
    features = range(num_of_features)
    all_possible_trip = [(features[i], features[j], features[k]) for i in range(len(features)) for j in
                         range(i + 1, len(features)) for k in range(j + 1, len(features))]
    score_list = []
    grid_list = []
    X = np.nan_to_num(X)
    score_array = np.zeros(shape=(num_of_features, num_of_features,num_of_features))
    c_param_array = np.zeros(shape=(num_of_features, num_of_features, num_of_features))
    gamma_param_array = np.zeros(shape=(num_of_features, num_of_features, num_of_features))
    print(score_array.shape)
    for pair in all_possible_trip:
        i = pair[0]
        j = pair[1]
        k = pair[2]
        print("Run SVM for features " + str(i) + ',' + str(j) + ',' + str(k))
        grid = get_svm_best_parm(X[:, [i, j, k]], y, 10)
        score_array[i][j][k] = grid.best_score_
        c_param_array[i][j][k] = grid.best_params_['C']
        gamma_param_array[i][j][k] = grid.best_params_['gamma']
        score_list.append(grid.best_score_)
        grid_list.append(grid)
    # np.save(score_array, score_array, delimiter=",")
    # np.save(score_list, score_list, delimiter=",")
    #
    scipy.io.savemat("score_array3.mat", mdict={'out': score_array}, oned_as='row')
    scipy.io.savemat("score_list3.mat", mdict={'out': score_list}, oned_as='row')

    print("Finish running for all the pairs")
    max_score = max(score_list)
    print("The best score is %f" % (max_score))
    best_score_index = [i for i, j in enumerate(score_list) if j == max_score]
    print(best_score_index)
    # np.savetxt('c_param_array.csv', c_param_array, delimiter=",")
    # np.savetxt('gamma_param_array.csv', gamma_param_array, delimiter=",")
    scipy.io.savemat("c_param_array3.mat", mdict={'out': c_param_array}, oned_as='row')
    scipy.io.savemat("gamma_param_array3.mat", mdict={'out': gamma_param_array}, oned_as='row')
    scipy.io.savemat("best_score_index3.mat", mdict={'out': best_score_index}, oned_as='row')

    return best_score_index


def find_best_features_four(X, y, features_names,four_list):
    num_of_features = X.shape[1]
    features = range(num_of_features)
    # all_possible_trip = [(features[i], features[j], features[k]) for i in range(len(features)) for j in
    #                      range(i + 1, len(features)) for k in range(j + 1, len(features))]
    score_list = []
    grid_list = []
    X = np.nan_to_num(X)
    score_array = np.zeros(shape=(num_of_features, num_of_features,num_of_features, num_of_features))
    c_param_array = np.zeros(shape=(num_of_features, num_of_features, num_of_features, num_of_features))
    gamma_param_array = np.zeros(shape=(num_of_features, num_of_features, num_of_features, num_of_features))
    print(score_array.shape)
    for pair in four_list:
        i = pair[0]
        j = pair[1]
        k = pair[2]
        l = pair[3]
        print("Run SVM for features " + str(i) + ',' + str(j) + ',' + str(k) + ',' + str(l))
        grid = get_svm_best_parm(X[:, [i, j, k, l]], y, 10)
        score_array[i][j][k][l] = grid.best_score_
        c_param_array[i][j][k][l] = grid.best_params_['C']
        gamma_param_array[i][j][k][l] = grid.best_params_['gamma']
        score_list.append(grid.best_score_)
        grid_list.append(grid)
    # np.save(score_array, score_array, delimiter=",")
    # np.save(score_list, score_list, delimiter=",")
    #
    scipy.io.savemat("score_array4.mat", mdict={'out': score_array}, oned_as='row')
    scipy.io.savemat("score_list4.mat", mdict={'out': score_list}, oned_as='row')

    print("Finish running for all the pairs")
    max_score = max(score_list)
    print("The best score is %f" % (max_score))
    best_score_index = [i for i, j in enumerate(score_list) if j == max_score]
    print(best_score_index)
    # np.savetxt('c_param_array.csv', c_param_array, delimiter=",")
    # np.savetxt('gamma_param_array.csv', gamma_param_array, delimiter=",")
    scipy.io.savemat("c_param_array4.mat", mdict={'out': c_param_array}, oned_as='row')
    scipy.io.savemat("gamma_param_array4.mat", mdict={'out': gamma_param_array}, oned_as='row')
    scipy.io.savemat("best_score_index4.mat", mdict={'out': best_score_index}, oned_as='row')

    return best_score_index

def find_best_features_five(X, y, features_names,five_list):
    num_of_features = X.shape[1]
    features = range(num_of_features)
    # all_possible_trip = [(features[i], features[j], features[k]) for i in range(len(features)) for j in
    #                      range(i + 1, len(features)) for k in range(j + 1, len(features))]
    score_list = []
    grid_list = []
    X = np.nan_to_num(X)
    score_array = np.zeros(shape=(num_of_features, num_of_features,num_of_features, num_of_features, num_of_features))
    c_param_array = np.zeros(shape=(num_of_features, num_of_features, num_of_features, num_of_features, num_of_features))
    gamma_param_array = np.zeros(shape=(num_of_features, num_of_features, num_of_features, num_of_features, num_of_features))
    print(score_array.shape)
    for pair in five_list:
        i = pair[0]
        j = pair[1]
        k = pair[2]
        l = pair[3]
        t = pair[4]
        print("Run SVM for features " + str(i) + ',' + str(j) + ',' + str(k) + ',' + str(l) + ',' + str(t))
        grid = get_svm_best_parm(X[:, [i, j, k, l, t]], y, 10)
        score_array[i][j][k][l][t] = grid.best_score_
        c_param_array[i][j][k][l][t] = grid.best_params_['C']
        gamma_param_array[i][j][k][l][t] = grid.best_params_['gamma']
        score_list.append(grid.best_score_)
        grid_list.append(grid)
    # np.save(score_array, score_array, delimiter=",")
    # np.save(score_list, score_list, delimiter=",")
    #
    scipy.io.savemat("score_array5.mat", mdict={'out': score_array}, oned_as='row')
    scipy.io.savemat("score_list5.mat", mdict={'out': score_list}, oned_as='row')

    print("Finish running for all the pairs")
    max_score = max(score_list)
    print("The best score is %f" % (max_score))
    best_score_index = [i for i, j in enumerate(score_list) if j == max_score]
    print(best_score_index)

    scipy.io.savemat("c_param_array5.mat", mdict={'out': c_param_array}, oned_as='row')
    scipy.io.savemat("gamma_param_array5.mat", mdict={'out': gamma_param_array}, oned_as='row')
    scipy.io.savemat("best_score_index5.mat", mdict={'out': best_score_index}, oned_as='row')

    return best_score_index



def get_svm_best_parm(X, y, n_splits):
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)


    param_grid = dict(gamma=gamma_range, C=C_range)
    cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.4, random_state=None,)
    grid = GridSearchCV(SVC(max_iter=10000), param_grid=param_grid, cv=cv, )
    print("start fit")
    grid.fit(X, y)
    print("finish fit")
    print("The best parameters are %s with a score of %f"
          % (grid.best_params_, grid.best_score_))

    return grid

# #############################################################################
# Visualization
#
# draw visualization of parameter effects


def plot_svms(classifiers, X_2d, y_2d, C_2d_range ,gamma_2d_range, grid):
    C_range = np.logspace(-2, 10, 13)
    gamma_range = np.logspace(-9, 3, 13)


    plt.figure(figsize=(8, 6))
    xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    for (k, (C, gamma, clf)) in enumerate(classifiers):
        # evaluate decision function in a grid
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # visualize decision function for these parameters
        plt.subplot(len(C_2d_range), len(gamma_2d_range), k + 1)
        plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)),
                  size='medium')

        # visualize parameter's effect on decision function
        plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r,
                    edgecolors='k')
        plt.xticks(())
        plt.yticks(())
        plt.axis('tight')

    scores = grid.cv_results_['mean_test_score'].reshape(len(C_range),
                                                         len(gamma_range))

    # Draw heatmap of the validation accuracy as a function of gamma and C
    #
    # The score are encoded as colors with the hot colormap which varies from dark
    # red to bright yellow. As the most interesting scores are all located in the
    # 0.92 to 0.97 range we use a custom normalizer to set the mid-point to 0.92 so
    # as to make it easier to visualize the small variations of score values in the
    # interesting range while not brutally collapsing all the low score values to
    # the same color.

    plt.figure(figsize=(8, 6))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
    plt.yticks(np.arange(len(C_range)), C_range)
    plt.title('Validation accuracy')
    plt.show()


def plot_best_svm(C, gamma, X_2d, y_2d):

    plt.clf()
    clf = SVC(C=C, gamma=gamma)
    clf.fit(X_2d, y_2d)
    # evaluate decision function in a grid
    xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # visualize decision function for these parameters
    # plt.title("Training data, with gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)),
    #           size='medium')
    plt.title("SVM training data",
              size='medium')

    # visualize parameter's effect on decision function
    plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_2d, cmap=plt.cm.RdBu_r,
                edgecolors='k')
    plt.xticks(())
    plt.yticks(())
    plt.axis('tight')
    plt.show()

    # h = clf.predict(X_test)
    # accs = np.mean(h == y_test)
    # print(accs)

    xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # visualize decision function for these parameters
    # plt.title("Test data, with gamma=10^%d, C=10^%d. Accuracy =%d" % (np.log10(gamma), np.log10(C),100*accs),
    #           size='medium')
    # plt.title("SVM test data. Accuracy = %d" % (100*accs),
    #           size='medium')
    # visualize parameter's effect on decision function
    plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.RdBu_r,
                edgecolors='k')
    plt.xticks(())
    plt.yticks(())
    plt.axis('tight')
    # plt.title("gamma=10^%d, C=10^%d" % (np.log10(gamma), np.log10(C)),
    #           size='medium')
    plt.show()

# create the good plot
def new_plot (C, gamma, X, y, pair, features_names, score):

    plt.clf()
    clf = SVC(C=C, gamma=gamma)
    clf.fit(X, y)
    # Circle out the test data
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=plt.cm.Dark2, edgecolor='k')

    plt.axis('tight')
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Dark2)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])
    plt.xlabel("feature %s" % (features_names[pair[0]]))
    plt.ylabel("feature %s" % (features_names[pair[1]]))

    plt.title("SVM for tag data. Features %s, %s, with test score %0.2f" % (features_names[pair[0]], features_names[pair[1]], score*100))
    # plt.legend(("act", "axc"), loc='center right')
    plt.show()


def new_plot_2(C, gamma, X, X_tag, y_tag, pair, features_names):
    colors = [(0, 0, 1), (1, 0, 0)]  # R -> G -> B
    cmap_name = 'my_list'
    cm = LinearSegmentedColormap.from_list(
        cmap_name, colors)

    def discrete_cmap(N, base_cmap=None):
        """Create an N-bin discrete colormap from the specified input map"""

        # Note that if base_cmap is a string or None, you can simply do
        #    return plt.cm.get_cmap(base_cmap, N)
        # The following works for string, None, or a colormap instance:

        base = plt.cm.get_cmap(base_cmap)
        color_list = base((1, 0))
        cmap_name = base.name + str(N)
        return LinearSegmentedColormap.from_list(cmap_name, color_list, N)

    discrete_cmap = discrete_cmap(2, 'Pastel1')

    plt.clf()
    clf = SVC(C=C, gamma=gamma)
    clf.fit(X_tag, y_tag)
    score = clf.score(X_tag, y_tag)
    # Circle out the test data
    plt.scatter(X_tag[1:20, 0], X_tag[1:20, 1], c=y_tag[1:20], zorder=10, cmap=cm, edgecolor='k')

    plt.axis('tight')
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=discrete_cmap)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])
    plt.xlabel("feature %s" % (features_names[pair[0]]))
    plt.ylabel("feature %s" % (features_names[pair[1]]))

    plt.title("SVM for tag data. Features %s, %s, with intrinsic error %0.2f" % (features_names[pair[0]], features_names[pair[1]], (1-score)*100) + "%")

    # plt.legend(("act", "axc"), loc='center right')
    plt.show()

    y = clf.predict(X)
    # Circle out the test data
    #
    # plt.scatter(x1_0, x2_0, c='red', label="exc", marker='.', s=3)
    # plt.scatter(x1_1, x2_1, c='blue', label="act or inh", marker='.', s=3)
    plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=cm, edgecolor='k')

    plt.axis('tight')
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=discrete_cmap)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
                levels=[-.5, 0, .5])
    plt.xlabel("feature %s" % (features_names[pair[0]]))
    plt.ylabel("feature %s" % (features_names[pair[1]]))

    plt.title("SVM for new data with predict labels. features %s, %s" % (
        features_names[pair[0]], features_names[pair[1]]))
    # plt.legend(("act", "axc"), loc='center right')
    plt.show()
    joblib.dump(clf, 'SVM_model_final.pkl')


#
# def plot_svm_3d(C, gamma, X, Y, pair, features_names):
#
#     # Fit the data with an svm
#     svc = SVC(kernel='linear')
#     X = X[:, [pair[0], pair[1], pair[2]]]
#     svc.fit(X, Y)
#
#     # The equation of the separating plane is given by all x in R^3 such that:
#     # np.dot(svc.coef_[0], x) + b = 0. We should solve for the last coordinate
#     # to plot the plane in terms of x and y.
#
#     z = lambda x, y: (-svc.intercept_[0] - svc.coef_[0][0] * x - svc.coef_[0][1] * y) / svc.coef_[0][2]
#
#     tmp = np.linspace(-2, 2, 51)
#     x, y = np.meshgrid(tmp, tmp)
#
#     # Plot stuff.
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(x, y, z(x, y))
#     ax.plot3D(X[Y == 0, 0], X[Y == 0, 1], X[Y == 0, 2], 'ob')
#     ax.plot3D(X[Y == 1, 0], X[Y == 1, 1], X[Y == 1, 2], 'sr')
#     plt.show()


def get_four_list(X):
    num_of_features = X.shape[1]
    matdata = scipy.io.loadmat("score_array2.mat")
    score_array = matdata['out']

    sum = score_array.sum()
    print(score_array.max())
    print("sum: ", sum)
    avr = sum/float(num_of_features*(num_of_features-1)/2)
    # print("avr: ",avr)
    score_array_more_avr = score_array > 0.978
    pairs_bigger_avr = []
    score = []
    for i in range(num_of_features):
        for j in range(i + 1, num_of_features):
            if score_array_more_avr[i][j]:
                pairs_bigger_avr.append((i, j))
                print(score_array[i][j])
                score.append(score_array[i][j])

    print(pairs_bigger_avr)
    print(score)
    print(len(pairs_bigger_avr))

    four_list = []
    for i in (range(len(pairs_bigger_avr))):
        for j in (range(i + 1, len(pairs_bigger_avr))):
            pair1 = pairs_bigger_avr[i]
            pair2 = pairs_bigger_avr[j]
            if (pair1[0] != pair2[0]) and (pair1[0] != pair2[1] and pair1[1] != pair2[0]) and (pair1[1] != pair2[1]):
                four = np.array([pair1[0], pair1[1], pair2[0], pair2[1]])
                # print(sorted(four))
                four_sort = sorted(four)
                if four_sort not in four_list:
                    four_list.append(four_sort)
    print("len(four_list): ", len(four_list))
    return four_list, pairs_bigger_avr

def get_five_list(X):
    num_of_features = X.shape[1]
    features = range(num_of_features)
    matdata = scipy.io.loadmat("score_list3.mat")
    score_list3 = matdata['out']
    score_list3 = score_list3[0]
    max_score = max(score_list3)
    all_possible_trip = [(features[i], features[j], features[k]) for i in range(len(features)) for j in
                         range(i + 1, len(features)) for k in range(j + 1, len(features))]
    best_score_index = [i for i, j in enumerate(score_list3) if j == max_score]
    print(best_score_index)
    five_list = []
    for ind in best_score_index:
        trip = all_possible_trip[ind]
        print("trip:", trip)
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                if(trip[0]!=i and trip[1]!=i and trip[2]!=i and trip[0]!=j and trip[1]!=j and trip[2]!=j):
                    five = np.array([trip[0], trip[1], trip[2], i, j])
                    five_sort = sorted(five)
                    if five_sort not in five_list:
                        five_list.append(five_sort)

    print(five_list)
    print(len(five_list))
    return five_list


if __name__ == '__main__':
    print("get data")
    features, X, X_tag, X_no_tag, labels, features_names, tag_index, all_labels = load_data()

# Code for example
    best_score_index = find_best_features(X_tag, labels, features_names)
    # best_score_index = find_best_features_trip(X_tag, labels, features_names)
    # print("five_list")
    # four_list = get_four_list(X)
    # best_score_index = find_best_features_four(X_tag, labels, features_names, four_list)
    five_list = get_five_list(X)
    # best_score_index = find_best_features_five(X_tag, labels, features_names, five_list)

    join_labels = []
    i=2
    j=9
    X_tag = X_tag[:, [i, j]]
    matdata = scipy.io.loadmat("c_param_array2.mat")
    c_array2 = matdata['out']
    matdata = scipy.io.loadmat("gamma_param_array2.mat")
    gamma_array2 = matdata['out']
    C = c_array2[i][j]
    gamma = gamma_array2[i][j]

    for i in range(50):
        X_train, X_test, y_train, y_test = train_test_split(X_tag, labels, test_size=0.5)
        clf = SVC(C=C, gamma=gamma)
        clf.fit(X_train, y_train)
        label = clf.predict(X_test)

        join_labels.append((y_test, label))
        # join_labels.append(label)

    join_labels = np.array(join_labels).T
    scipy.io.savemat("join_labels_SVM_best.mat", mdict={'out': join_labels}, oned_as='row')
    print("hey")

