
import pandas
import numpy as np
from runSVM import *
from numpy import genfromtxt
import scipy.io
import heapq
from numpy import median
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from mpl_toolkits.mplot3d import Axes3D
#

def load_final_data():
    matdata = scipy.io.loadmat("score_array2.mat")
    score_array2 = matdata['out']

    # np.savetxt('score_array2_2.csv', score_array2, delimiter=",")

    matdata = scipy.io.loadmat("score_array3.mat")
    score_array3 = matdata['out']
    matdata = scipy.io.loadmat("score_array4.mat")
    score_array4 = matdata['out']
    matdata = scipy.io.loadmat("score_array5.mat")
    score_array5 = matdata['out']

    score_array = [0, 0, score_array2, score_array3, score_array4, score_array5]
    # score_array = [0, 0, score_array2, score_array3, score_array4]

    matdata = scipy.io.loadmat("score_list2.mat")
    score_list2 = matdata['out']
    matdata = scipy.io.loadmat("score_list3.mat")
    score_list3 = matdata['out']
    matdata = scipy.io.loadmat("score_list4.mat")
    score_list4 = matdata['out']
    matdata = scipy.io.loadmat("score_list5.mat")
    score_list5 = matdata['out']
    score_list=[0 , 0,score_list2[0], score_list3[0], score_list4[0], score_list5[0]]
    # score_list=[0 , 0, score_list2[0], score_list3[0], score_list4[0]]

    return score_list, score_array


def plot_svm_result(X, score_list):
    num_of_features = X.shape[1]

    score_list2 = 1-score_list[2]
    score_list3 = 1-score_list[3]
    score_list4 = 1-score_list[4]
    score_list5 = 1-score_list[5]

    print(len(score_list2))
    print(len(score_list3))
    print(len(score_list4))
    print(len(score_list5))

    sum2 = score_list2.sum()
    sum3 = score_list3.sum()
    sum4 = score_list4.sum()
    sum5 = score_list5.sum()

    med2 = median(score_list2)
    med3 = sum3/float(len(score_list3))
    med4 = sum4/float(len(score_list4))
    med5 = sum5/float(len(score_list5))

    print("med2: ", med2)
    print("med3: ", med3)
    print("med4: ", med4)
    print("med5: ", med5)

    med = [0, 0, med2, med3, med4, med5]

    x_2 = [2]*len(score_list2)
    x_3 = [3]*len(score_list3)
    x_4 = [4]*len(score_list4)
    x_5 = [5]*len(score_list5)

    fig, ax = plt.subplots()

    ax.scatter(x_2, score_list2*100)
    ax.scatter(x_3, score_list3*100)
    ax.scatter(x_4, score_list4*100)
    ax.scatter(x_5, score_list5*100)
    ax.set_xlabel("Numbers of features")
    ax.set_ylabel("GENERALIZATION ERROR")
    ax.set_yscale("log")
    ax.set_xticks(np.arange(6))
    ax.set_xticklabels((' ', ' ', '2', '3', '4', '5'))
    ax.set_title("SVM GENERALIZATION ERROR per numbers of features", size=20)

    for i in range(2, 6):
        label = "Median: %0.2f" % (med[i]*100) + "%"
        ax.annotate(label, xy=(i, med[i]*100), xytext=(i+0.3, med[i]*100),
                    arrowprops=dict(facecolor='black', shrink=0.02),
                    )
    fig.tight_layout()

    plt.show()



# def get_five_list(X, score_list):
#     num_of_features = X.shape[1]
#     features = range(num_of_features)
#     score_list3 = score_list[3]
#     max_score = max(score_list3)
#     all_possible_trip = [(features[i], features[j], features[k]) for i in range(len(features)) for j in
#                          range(i + 1, len(features)) for k in range(j + 1, len(features))]
#     best_score_index = [i for i, j in enumerate(score_list3) if j == max_score]
#     print(best_score_index)
#     five_list = []
#     for ind in best_score_index:
#         trip = all_possible_trip[ind]
#         print("trip:", trip)
#         for i in range(len(features)):
#             for j in range(i+1, len(features)):
#                 if(trip[0]!=i and trip[1]!=i and trip[2]!=i and trip[0]!=j and trip[1]!=j and trip[2]!=j):
#                     five = np.array([trip[0], trip[1], trip[2], i, j])
#                     five_sort = sorted(five)
#                     if five_sort not in five_list:
#                         five_list.append(five_sort)
#
#     print(five_list)
#     print(len(five_list))
#     return five_list

def run_final_svm(X,y,i ,j, features_names):
    _, _, X_tag, X_no_tag, labels, features_names, _, _ = load_data()

    X_tag = X_tag[:, [i, j]]
    matdata = scipy.io.loadmat("c_param_array2.mat")
    c_array2 = matdata['out']
    matdata = scipy.io.loadmat("gamma_param_array2.mat")
    gamma_array2 = matdata['out']
    C = c_array2[i][j]
    gamma = gamma_array2[i][j]
    print(C, gamma)

    new_plot_2(C, gamma, X_no_tag[:, [i, j]], X_tag, np.array(labels), (i,j), features_names)


if __name__ == '__main__':
    features, X, X_tag, X_no_tag, labels, features_names, tag_index, s = load_data()
    print(len(features_names))

    # Load all the data from running the SVM
    score_list, score_array = load_final_data()

    # example how to print the SVM in the array list 2 of (2,9)
    # run_final_svm(X_tag, labels, 2, 9, features_names)

    avr_list = plot_svm_result(X, score_list)




