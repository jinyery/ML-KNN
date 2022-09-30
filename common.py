import os
import pickle

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import coverage_error
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import zero_one_loss


def load_csv(file_path):
    if type(file_path) is str:
        return np.loadtxt(file_path, delimiter=",")
    if type(file_path) is list:
        objs = []
        for i in range(len(file_path)):
            objs.append(load_csv(file_path[i]))
        return tuple(objs)


def save_csv(file_path, obj):
    if type(file_path) is str:
        np.savetxt(file_path, obj, delimiter=",", fmt='%s')
    if type(file_path) is list:
        for i in range(len(file_path)):
            save_csv(file_path[i])


def load_pkl(file_path):
    with open(file_path, "rb") as file:
        obj = pickle.load(file)
        return obj


def save_pkl(file_path, obj):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as file:
        pickle.dump(obj, file)


def load_data(path_train_x, path_train_y, path_test_x, path_test_y, invalid_del=False, normalization=False):
    _train_x, _train_y, _test_x, _test_y = \
        load_csv([path_train_x, path_train_y, path_test_x, path_test_y])
    if invalid_del or normalization:
        _train_x, _invalid_idx = del_invalid(_train_x)
        _test_x = del_invalid(_test_x, _invalid_idx)
        if normalization:
            _train_x, _mean, _std = zscore_norm(_train_x)
            _test_x = zscore_norm(_test_x, _mean, _std)
    return _train_x, _train_y, _test_x, _test_y


def check_numpy(*data):
    new_data = []
    for item in data:
        if item is None \
                or type(item) is np.ndarray \
                or type(item) is np.matrix:
            new_data.append(item)
        else:
            new_data.append(np.array(item))

    if len(new_data) == 1:
        return new_data[0]
    return tuple(new_data)


def del_invalid(samples, invalid_cols=None):
    if invalid_cols is not None and len(invalid_cols) == 0:
        return samples
    elif invalid_cols is not None and len(invalid_cols) > 0:
        return np.delete(samples, invalid_cols, axis=1)
    invalid_cols = []
    for i in range(samples.shape[1]):
        uni = np.unique(samples[:, i])
        if uni.shape[0] == 1:
            invalid_cols.append(i)
    return np.delete(samples, invalid_cols, axis=1), invalid_cols


def zscore_norm(samples, mean=None, std=None):
    return_flag = False
    samples = check_numpy(samples).astype(np.float32)
    if mean is None:
        return_flag = True
        mean = np.mean(samples, 0)
    if std is None:
        return_flag = True
        std = np.std(samples, 0)
    if return_flag:
        return _zscore_norm(samples, mean, std), mean, std
    return _zscore_norm(samples, mean, std)


def _zscore_norm(samples, mean, std):
    samples = (samples - mean) / std
    return samples


def evaluate(y_true, y_pred, verbose=True):
    result = dict()
    result["accuracy_score"] = accuracy_score(y_true, y_pred)
    result["precision_score"] = precision_score(y_true, y_pred, average="samples")
    result["recall_score"] = recall_score(y_true, y_pred, average="samples")
    result["f1_score"] = f1_score(y_true, y_pred, average="samples")
    if verbose:
        print("accuracy_score:", result["accuracy_score"])
        print("precision_score:", result["precision_score"])
        print("recall_score:", result["recall_score"])
        print("f1_score:", result["f1_score"])

    result["hamming_loss"] = hamming_loss(y_true, y_pred)
    result["zero_one_loss"] = zero_one_loss(y_true, y_pred)
    result["coverage_error"] = coverage_error(y_true, y_pred)
    result["label_ranking_loss"] = label_ranking_loss(y_true, y_pred)
    result["average_precision_score"] = average_precision_score(y_true, y_pred)
    result["label_ranking_average_precision_score"] = label_ranking_average_precision_score(y_true, y_pred)
    if verbose:
        print("hamming_loss:", result["hamming_loss"])
        print("zero_one_loss:", result["zero_one_loss"])
        print("coverage_error:", result["coverage_error"])
        print("label_ranking_loss:", result["label_ranking_loss"])
        print("average_precision_score:", result["average_precision_score"])
        print("label_ranking_average_precision_score:", result["label_ranking_average_precision_score"])
    return result
