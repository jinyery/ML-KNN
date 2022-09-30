import common
from core.mlknn import MLKNN

if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    dataset_path = "./dataset/yeast/"
    train_x, train_y, test_x, test_y = common.load_data(dataset_path + "train_x.csv", dataset_path + "train_y.csv",
                                                        dataset_path + "test_x.csv", dataset_path + "test_y.csv",
                                                        normalization=True)
    clf = MLKNN()
    clf.train(train_x, train_y)
    # clf = common.load_pkl("./clf.pkl")
    predictions = clf.predict(test_x)
    common.evaluate(test_y, predictions)
    # common.save_pkl("./clf.pkl", clf)
