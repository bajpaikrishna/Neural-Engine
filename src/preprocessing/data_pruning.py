import numpy as np

def prune_data(X, y, threshold=0.01):
    # Remove features with low variance
    variances = np.var(X, axis=0)
    selected_features = variances > threshold
    X_pruned = X[:, selected_features]
    return X_pruned, selected_features
