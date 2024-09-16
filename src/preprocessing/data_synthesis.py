from imblearn.over_sampling import SMOTE

def synthesize_data(X, y):
    smote = SMOTE()
    X_synthetic, y_synthetic = smote.fit_resample(X, y)
    return X_synthetic, y_synthetic
