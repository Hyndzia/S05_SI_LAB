import numpy as np
from sklearn.model_selection import train_test_split
from bayes import DiscreteNBC


def discretize(X, bins, mins_ref=None, maxes_ref=None):
    if mins_ref is None:
        mins_ref = np.min(X, axis=0)
        maxes_ref = np.max(X, axis=0)
    X_d = np.clip(((X - mins_ref) / (maxes_ref - mins_ref) * bins).astype(np.int8), 0, bins - 1)
    return X_d, mins_ref, maxes_ref

seed = 0
train_size = 0.75
bins = 25

data = np.genfromtxt("spambase.data", delimiter=",")

X = data[:, :-1]
y = data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=seed, stratify=y)

X_train_d, mins_ref, maxes_ref = discretize(X_train, bins=bins)
X_test_d, _, _ = discretize(X_test, bins=bins, mins_ref=mins_ref, maxes_ref=maxes_ref)


n = X.shape[1]
domain_sizes = bins * np.ones(n, dtype=np.int8)
clf = DiscreteNBC(domain_sizes=domain_sizes, laplace=True)
clf.fit(X_train_d, y_train)
y_pred_train = clf.predict(X_train_d)
acc_train = clf.score(X_train_d, y_train)
acc_test = clf.score(X_test_d, y_test)
print(f"ACC -> TRAIN: {acc_train}, TEST: {acc_test}")


