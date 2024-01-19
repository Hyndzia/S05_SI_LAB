from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class DiscreteNBC(BaseEstimator, ClassifierMixin):
    def __init__(self, domain_sizes, laplace=False):
        self.class_labels_ = None
        self.log_p_ = None
        self.PY_ = None
        self.domain_sizes = domain_sizes
        self.laplace = laplace

    def fit(self, X, y):
        self.log_p_ = []

        self.class_labels_ = np.unique(y)
        m, n = X.shape
        K = self.class_labels_.size
        self.PY_ = np.zeros(K)

        for index, label in enumerate(self.class_labels_):
            where_class_label = y == label
            self.PY_[index] = np.mean(where_class_label)

        self.class_labels_ = np.unique(y)
        m, n = X.shape
        K = self.class_labels_.size
        yy = np.zeros(m, dtype=np.int8)
        self.PY_ = np.zeros(K)

        for index, label in enumerate(self.class_labels_):
            where_class_label = y == label
            yy[where_class_label] = index
            self.PY_[index] = np.mean(where_class_label)

        q_max = np.max(self.domain_sizes)
        self.log_p_ = np.zeros((K, n, q_max))

        for i in range(m):
            for j in range(n):
                self.log_p_[yy[i], j, X[i, j]] += 1

        for k in range(K):
            if not self.laplace:
                self.log_p_[k] = (np.log(self.log_p_[k]))/np.log(self.PY_[k] * m)
            else:
                for j in range(n):
                    self.log_p_[k, j] = np.log((self.log_p_[k, j] + 1) / (self.PY_[k] * m + self.domain_sizes[j]))
        pass

    def predict(self, X):
        return self.class_labels_[np.argmax(self.predict_proba(X), axis=1)]

    def predict_proba(self, X):
        m, n = X.shape
        K = self.class_labels_.size
        log_scores = np.zeros((m, K))
        for i in range(m):
            log_scores[i] = np.log(self.PY_)
            for k in range(K):
                for j in range(n):
                    log_scores[i, k] += self.log_p_[k, j, X[i, j]]

        return log_scores
