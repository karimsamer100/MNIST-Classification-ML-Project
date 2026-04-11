import numpy as np


class PCAFromScratch:
    def __init__(self, n_components):
        # number of principal components to keep
        self.n_components = n_components

        # mean of training data
        self.mean = None

        # principal directions
        self.components = None

        # explained variance values
        self.explained_variance = None

        # ratio of explained variance
        self.explained_variance_ratio = None

    def fit(self, X):
        # compute mean of training data
        self.mean = np.mean(X, axis=0)

        # center the data by subtracting the mean
        X_centered = X - self.mean

        # compute covariance matrix
        covariance_matrix = np.cov(X_centered, rowvar=False)

        # compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # sort them in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # keep only the top principal components
        self.explained_variance = eigenvalues[:self.n_components]
        self.components = eigenvectors[:, :self.n_components]

        # compute explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio = self.explained_variance / total_variance

    def transform(self, X):
        # center the data using training mean
        X_centered = X - self.mean

        # project data onto principal components
        X_projected = np.dot(X_centered, self.components)

        return X_projected

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


def apply_pca(X_train, X_val, X_test, n_components=50):
    # create PCA object
    pca = PCAFromScratch(n_components=n_components)

    # fit and transform training data
    X_train_pca = pca.fit_transform(X_train)

    # transform validation and test data
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_val_pca, X_test_pca, pca