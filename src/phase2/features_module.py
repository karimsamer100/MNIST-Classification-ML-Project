import numpy as np


class PCAFromScratch:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None
        self.explained_variance_ratio = None

    def fit(self, X):
        # checks
        if self.n_components <= 0:
            raise ValueError("n_components must be positive")

        if self.n_components > X.shape[1]:
            raise ValueError("n_components cannot exceed number of features")

        # center data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # manual covariance (better than np.cov for project requirement)
        covariance_matrix = np.dot(X_centered.T, X_centered) / (X_centered.shape[0] - 1)

        # eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # sort descending
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # select top components
        self.explained_variance = eigenvalues[:self.n_components]
        self.components = eigenvectors[:, :self.n_components]

        # explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio = self.explained_variance / total_variance

    def transform(self, X):
        if self.mean is None or self.components is None:
            raise ValueError("PCA must be fitted before calling transform")

        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def apply_pca(X_train, X_val, X_test, n_components=50):
    pca = PCAFromScratch(n_components=n_components)

    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_val_pca, X_test_pca, pca