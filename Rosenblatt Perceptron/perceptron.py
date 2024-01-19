import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class NonlinearPerceptron(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate, num_centers, sigma, max_iterations):
        self.learning_rate = learning_rate
        self.num_centers = num_centers
        self.sigma = sigma
        self.max_iterations = max_iterations

    def fit(self, X, y):
        self.class_labels_ = np.unique(y)
        m, n = X.shape
        yy = np.ones(m, dtype=np.int8)
        yy[y == self.class_labels_[0]] = -1

        self.centers_ = np.random.uniform(-1, 1, size=(self.num_centers, n))
        X_features = self.gaussian_kernel(X, self.centers_, self.sigma)

        self.w_ = np.zeros(self.num_centers + 1)
        self.k_ = 0

        X_ext = np.c_[np.ones(m), X_features]

        while self.k_ < self.max_iterations:
            s = X_ext.dot(self.w_)
            errors = np.where(s * yy <= 0)[0]
            if errors.size == 0:
                break
            i = np.random.choice(errors)
            self.w_ = self.w_ + self.learning_rate * yy[i] * X_ext[i]
            self.k_ += 1

    def predict(self, X):
        m = X.shape[0]
        X_features = self.gaussian_kernel(X, self.centers_, self.sigma)
        X_ext = np.c_[np.ones(m), X_features]

        sums = X_ext.dot(self.w_)

        predictions = np.empty(m, dtype=self.class_labels_.dtype)
        predictions[sums <= 0.0] = self.class_labels_[0]
        predictions[sums > 0.0] = self.class_labels_[1]

        return predictions

    def gaussian_kernel(self, X, centers, sigma):
        distances = np.sum((X[:, None] - centers) ** 2, axis=2)
        return np.exp(-distances / (2 * sigma ** 2))
