import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



# --------------------------
# Normalisation des données (X seulement)
# --------------------------
class NormalisationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column_target = None):
        self.scaler = StandardScaler()
        self.column_target = column_target or 'y'
        

    def fit(self, X, y=None):
        self.is_fitted_ = True  # vérif
        self.scaler.fit(X)
        return self
    
    def transform(self, X):
        if not hasattr(self, 'is_fitted_'):
            raise RuntimeError("NormalisationTransformer n'est pas entrainé.")
        X_copy = X.copy()
        if self.column_target not in X_copy.columns:
            raise KeyError(f"Colonne cible '{self.column_target}' absente du DataFrame. Colonnes disponibles : {list(X_copy.columns)}")
        X_scaled = pd.DataFrame(self.scaler.transform(X_copy), columns = X_copy.columns)
        X_scaled.rename(columns = {self.column_target : f"{self.column_target}_scaled"}, inplace = True)
        X_scaled[self.column_target] = X_copy[self.column_target].values
        return X_scaled
    


