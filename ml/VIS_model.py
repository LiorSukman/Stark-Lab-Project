import numpy as np
import matplotlib.pyplot as plt

def make_meshgrid(x, y, h = 5):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def fix_examples(examples, new_shape, locs):
    """
    fix size of examples

    Parameters
    ----------
    examples: two-dimensional examples to predict
    new_shape: number of features to have
    locs: indices of features to visualize

    Returns
    -------
    ret : ndarray; fixed examples
    """
    ret = np.zeros((len(examples), new_shape))
    ret[:,locs[0]] = examples[:, 0]
    ret[:,locs[1]] = examples[:, 1]
    return ret

def plot_contours(clf, xx, yy, num_features, locs, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    predict = fix_examples(np.c_[xx.ravel(), yy.ravel()], num_features, locs)
    Z = clf.predict(predict)
    Z = Z.reshape(xx.shape)
    out = plt.contourf(xx, yy, Z, **params)
    return out

def visualize_model(data, targets, clf, h, feature1 = 0, feature2 = 1):
    X = data
    y = targets

    # title for the plot
    title = 'Classifier Predictions'

    X0, X1 = X[:, feature1], X[:, feature2]
    xx, yy = make_meshgrid(X0, X1, h = h)

    plot_contours(clf, xx, yy, data.shape[1], (feature1, feature2), cmap = plt.cm.coolwarm, alpha = 0.8)
    plt.scatter(X0, X1, c = y, cmap = plt.cm.coolwarm, edgecolors = 'k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)

    plt.show()
