import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# --------------------------
# Transformateur sklearn : inverse les colonnes spécifiées (1/x)
# Utilisé pour transformer les colonnes comme la Température ('T') ou le Rayonnement ('R')
# --------------------------
class InversionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_inverse=['T', 'R']):
        self.columns_to_inverse = columns_to_inverse
        

    def fit(self, X, y=None):
        self.is_fitted_ = True  # vérif
        return self

    def transform(self, X):
        if not hasattr(self, 'is_fitted_'):
            raise RuntimeError("InversionTransformer n'est pas entrainé.")
        X_copy = X.copy()
        for col in self.columns_to_inverse:
            if col not in X_copy.columns:
                raise ValueError(f"Colonne '{col}' absente du DataFrame d'entrée.")
            # Gestion de division par zéro ou très petite valeur
            # Normalement cette situation ne se produit pas, vue que l'entrée est issue d'une décomposition de séries temporelles multiplicatives
            if(X_copy[col].min() == 0):
                X_copy[col] = X_copy[col] + 1 #décalage pour l'inversion 
            with np.errstate(divide='ignore', invalid='ignore'): 
                X_copy[col] = 1 / X_copy[col]
                X_copy[col] = X_copy[col].replace([np.inf, -np.inf], np.nan)
                X_copy[col] = X_copy[col].fillna(X_copy[col].max())

        return X_copy
