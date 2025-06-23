import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# --------------------------
# Fonction de création des séquences
# --------------------------
def create_sequences(X, y, window_size):
    Xs, ys = [], []
    y = y.values if isinstance(y, pd.Series) else np.array(y)  # ⬅️ conversion en array
    for i in range(len(X) - window_size):
        X_seq = X[i:i+window_size]
        y_seq = y[i + window_size]
        if len(X_seq.shape) == 1:
            X_seq = X_seq.reshape(-1, 1)
        Xs.append(X_seq)
        ys.append(y_seq)
    return np.array(Xs), np.array(ys)

# --------------------------
# Transformeur pour générer l'entrée LSTM
# --------------------------
class SequenceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, window_size=48 , column_target = 'y'):
        self.window_size = window_size
        self.X_seq = None
        self.y_seq = None
        self.X_seq_concat = None
        self.column_target = column_target
        

    def fit(self, X, y=None):
        self.is_fitted_ = True  # vérif
        return self

    def transform(self, X, y= None):
        if not hasattr(self, 'is_fitted_'):
            raise RuntimeError("InversionTransformer n'est pas entrainé.")
        self.X_seq, self.y_seq = create_sequences(X,  # on garde la cible dans les exogènes
                                                  y = X[self.column_target],
                                                  window_size=self.window_size)
       
        #(-) return self.X_seq, self.y_seq
 
        # On concatène y_seq avec X_seq sur le dernier axe 
        y_seq_expanded = self.y_seq.reshape(-1, 1, 1)  # (n_samples, 1, 1)
        y_seq_broadcasted = np.repeat(y_seq_expanded, self.window_size, axis=1)  # (n_samples, window_size, 1)
        self.X_seq_concat = np.concatenate((self.X_seq, y_seq_broadcasted), axis=2)
        return self.X_seq_concat  # sklearn est heureux, car on retourne un unique X transformé
        

    def get_target(self):
        return self.y_seq
    
    def get_exogene(self):
        return self.X_seq