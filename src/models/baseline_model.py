import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_baseline(X_train, y_train, X_test, y_test):
    """
    Baseline ML model using PCA + Logistic Regression
    """
    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)

    pca = PCA(n_components=50)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_pca, y_train)

    y_pred = clf.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)

    return acc
