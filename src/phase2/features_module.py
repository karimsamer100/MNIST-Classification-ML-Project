import numpy as np


class PCAFromScratch:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None
        self.explained_variance_ratio = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        covariance_matrix = np.cov(X_centered, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        self.explained_variance = eigenvalues[:self.n_components]
        self.components = eigenvectors[:, :self.n_components]

        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio = self.explained_variance / total_variance

    def transform(self, X):
        X_centered = X - self.mean
        X_projected = np.dot(X_centered, self.components)
        return X_projected

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def apply_pca(X_train, X_val, X_test, n_components=50):
    pca = PCAFromScratch(n_components=n_components)

    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_val_pca, X_test_pca, pca