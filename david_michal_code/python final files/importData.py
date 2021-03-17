import scipy.io as sio
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

# return training_data, training_labels, validation_data, validation_labels
def get_data():
    """
     :return:
        features:  the features table - 12 features for 949 units numpyArray(12, 949)
        X:  features table in list format. <class 'list'> (12, 949)
        X_tag:  features table in list format contain only tag units <class 'list'>(12,503)
        labels:  a list of labels for the tag unit <class 'list'>(503)
        features_names: list of the 12 features names <class 'list'>(12)

     """
    p_features = Path("C:\\Users\mich\Google Drive\Brain project\data for python\\features.mat").resolve()
    p_act = Path("C:\\Users\mich\Google Drive\Brain project\data for python\\act.mat").resolve()
    p_inh = Path("C:\\Users\mich\Google Drive\Brain project\data for python\\inh.mat").resolve()
    p_exc = Path("C:\\Users\mich\Google Drive\Brain project\data for python\\exc.mat").resolve()
    p_features_names = Path("C:\\Users\mich\Google Drive\Brain project\data for python\\feature_names.mat").resolve()

    features = sio.loadmat(p_features)
    features = features['features']
    features = features.T
    print("features.shape : " + str(features.shape))

    features_names_tmp = sio.loadmat(p_features_names)
    features_names_tmp = features_names_tmp['feature_names']

    features_names = []

    for i in range(len(features_names_tmp[0])):
        features_names.append(features_names_tmp[0][i][0])

    print("features_names: ", features_names)


    p_act = sio.loadmat(p_act)
    p_act = p_act['act']
    print("act.shape : " + str(p_act.shape))

    exc = sio.loadmat(p_exc)
    exc = exc['exc']
    print("exc.shape : " + str(exc.shape))

    inh = sio.loadmat(p_inh)
    inh = inh['inh']
    print("inh.shape : " + str(inh.shape))

    vec_act_or_inh = []
    vec_inh = []
    vec_exc = []
    for i,act_obj in enumerate(p_act):
        vec_act_or_inh.append(act_obj[0] or inh[i][0])
        vec_exc.append(exc[i][0])
        vec_inh.append(inh[i][0])


    print("sum(vec_act_or_inh): ", sum(vec_act_or_inh))
    print("sum(vec_exc): ", sum(vec_exc))


    tag = [int (p | l) for p, l in zip(vec_act_or_inh, vec_exc)]
    print("sum(tag): ", sum(tag))
    tag = np.array(tag)
    tag_list =list(np.where(tag == 1)[0])
    non_tag_list = list(np.where(tag == 0)[0])
    X_tag = features[:, tag_list].T.tolist()
    X_no_tag = features[:, non_tag_list].T.tolist()
    X = features.T.tolist()



# calc only tags
    vec_act_or_inh = np.array(vec_act_or_inh)
    ind = np.where(vec_act_or_inh == 1)[0]
    vec_act_or_inh[ind] = 2
    s_tag = np.array(vec_exc) + vec_act_or_inh

    # s = 2 * np.array(vec_exc) + np.array(vec_act_or_inh)
    w = list(np.where(s_tag > 0)[0])
    s = s_tag[w]
    ind = np.where(s > 1)[0]
    s[ind] = 0
    labels = s.tolist()
    print("sum(labels): ", sum(labels))
    print("sum(labels): ", sum(labels))
    print("s_tag len: ", len(s_tag))

    return features, X, X_tag, X_no_tag, labels, features_names, features_names, features_names


if __name__ == '__main__':
    print('get data')
    features, X, X_tag, X_no_tag, labels, features_names, tag_list, s_tag = get_data()
    print(features.shape)
