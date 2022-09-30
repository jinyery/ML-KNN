import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm


class MLKNN:
    def __init__(self, n_neighbors=10, smoothing_param=1):
        self.train_x = None
        self.train_y = None
        self.label_num = None
        self.prior_probs = None
        self.posterior_probs_hits = None
        self.posterior_probs_misses = None
        self.n_neighbors = n_neighbors
        self.smoothing_param = smoothing_param

    def _computing_prior_prob(self):
        match_sum = np.sum(self.train_y, axis=0)
        _numerator = self.smoothing_param + match_sum
        _denominator = self.smoothing_param * 2 + len(self.train_y)
        self.prior_probs = _numerator / _denominator

    def _computing_posterior_prob(self):
        self.posterior_probs_hits = np.zeros((self.label_num, self.n_neighbors + 1))
        self.posterior_probs_misses = np.zeros((self.label_num, self.n_neighbors + 1))
        for label_idx in tqdm(range(self.label_num), desc='Train processing'):
            self_hits = np.zeros(self.n_neighbors + 1)
            self_misses = np.zeros(self.n_neighbors + 1)

            for data_idx in range(len(self.train_x)):
                neighbors = self.knn_spin(data_idx, self.train_x, self.n_neighbors)
                neighbors_hits_num = np.sum(self.train_y[neighbors, label_idx]).astype(int)
                if self.train_y[data_idx][label_idx] == 1:
                    self_hits[neighbors_hits_num] += 1
                else:
                    self_misses[neighbors_hits_num] += 1

            self_hits_sum = np.sum(self_hits)
            self_misses_sum = np.sum(self_misses)
            self.posterior_probs_hits[label_idx] = (self.smoothing_param + self_hits) / (
                    self.smoothing_param * (self.n_neighbors + 1) + self_hits_sum)
            self.posterior_probs_misses[label_idx] = (self.smoothing_param + self_misses) / (
                        self.smoothing_param * (self.n_neighbors + 1) + self_misses_sum)

    def train(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y
        self.label_num = train_y.shape[1]

        self._computing_prior_prob()
        self._computing_posterior_prob()

    def predict(self, test_x):
        predictions = np.zeros((len(test_x), self.label_num))
        for i in tqdm(range(len(test_x)), desc="Predict processing"):
            neighbors = self.knn(test_x[i], self.train_x, self.n_neighbors)
            neighbors_hits_num = np.sum(self.train_y[neighbors], axis=0).astype(int)
            # The '_row_idx' is only used to make one-to-one coordinates with neighbors_hits_num
            _row_idx = np.arange(self.label_num)
            probs_hit = self.prior_probs * self.posterior_probs_hits[_row_idx, neighbors_hits_num]
            probs_miss = (1 - self.prior_probs) * self.posterior_probs_misses[_row_idx, neighbors_hits_num]
            predictions[i] = (probs_hit > probs_miss).astype(int)
        return predictions

    @staticmethod
    def knn(target, data, n_neighbors=None):
        dists = euclidean_distances([target], data)[0]
        dists_idx = np.argsort(dists)
        if n_neighbors is None:
            return dists_idx
        return dists_idx[:n_neighbors]

    @staticmethod
    def knn_spin(target_idx, data, n_neighbors):
        dists_idx = MLKNN.knn(data[target_idx], data)
        aim_idx = np.where(dists_idx == target_idx)[0][0]
        return np.delete(dists_idx, aim_idx)[:n_neighbors]
