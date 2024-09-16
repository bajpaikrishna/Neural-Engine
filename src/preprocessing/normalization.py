from sklearn.preprocessing import StandardScaler

def normalize_data(X):
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    return X_normalized, scaler
