from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd

def select_features(X, y, k=10):
    selector = SelectKBest(chi2, k=k)
    X_new = selector.fit_transform(X, y)
    return X_new, selector.get_support(indices=True)

def apply_feature_selection(X, y, k=10):
    X_new, selected_features = select_features(X, y, k)
    return pd.DataFrame(X_new), selected_features
