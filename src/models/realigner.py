from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class ReAligner(BaseEstimator, TransformerMixin):
    def __init__(self, window_size):
        self.window_size = window_size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        #print('avant', X.index)
        #print('après', X.iloc[self.window_size:,:])
        return X.iloc[self.window_size:,:]  # coupe le début qui vient de l'entrainement