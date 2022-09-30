import numpy as np
import pandas as pd
from scipy.io import arff

import common

if __name__ == '__main__':
    dataset = "scene"
    label_num = 6

    train_file_path = dataset + '/' + dataset + '-train.arff'
    test_file_path = dataset + '/' + dataset + '-test.arff'

    train_data, _ = arff.loadarff(train_file_path)
    test_data, _ = arff.loadarff(test_file_path)
    train_data = np.array(pd.DataFrame(train_data).values).astype(float)
    test_data = np.array(pd.DataFrame(test_data).values).astype(float)

    train_x = train_data[:, :-label_num]
    train_y = train_data[:, -label_num:].astype(int)
    test_x = test_data[:, :-label_num]
    test_y = test_data[:, -label_num:].astype(int)

    common.save_csv(dataset + '/train_x.csv', train_x)
    common.save_csv(dataset + '/train_y.csv', train_y)
    common.save_csv(dataset + '/test_x.csv', test_x)
    common.save_csv(dataset + '/test_y.csv', test_y)
