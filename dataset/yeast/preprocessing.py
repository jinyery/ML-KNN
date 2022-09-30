import common
import numpy as np
import pandas as pd
from scipy.io import arff

train_file_path = './yeast-train.arff'
test_file_path = './yeast-test.arff'

train_data, _ = arff.loadarff(train_file_path)
test_data, _ = arff.loadarff(test_file_path)
train_data = np.array(pd.DataFrame(train_data).values).astype(float)
test_data = np.array(pd.DataFrame(test_data).values).astype(float)

train_x = train_data[:, 0:103]
train_y = train_data[:, 104:117].astype(int)
test_x = test_data[:, 0:103]
test_y = test_data[:, 104:117].astype(int)
print(test_x)

common.save_csv('./train_x.csv', train_x)
common.save_csv('./train_y.csv', train_y)
common.save_csv('./test_x.csv', test_x)
common.save_csv('./test_y.csv', test_y)