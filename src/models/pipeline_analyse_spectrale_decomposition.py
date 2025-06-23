# ------------------------------------------------------------------------------------------------------------------------------------------------
# Les modules nécessaires internes
# ------------------------------------------------------------------------------------------------------------------------------------------------
import decomposition_serie_temporelle
import analyse_spectrale 

# pour la mise à jour
import importlib
importlib.reload(decomposition_serie_temporelle)
importlib.reload(analyse_spectrale)

from decomposition_serie_temporelle import DecompositionSerieTemporelle
from analyse_spectrale import SpectrogramAnalysis


# Packages 
import pandas as pd
import numpy as np
from scipy.signal import spectrogram, find_peaks
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.signal import find_peaks


class SpectroDecompPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, target_column='y', forme='multiplicative', spectro_kwargs=None):
        self.target_column = target_column  # La colonne cible 'y'
        self.forme = forme
        self.spectro_kwargs = spectro_kwargs or {}
        self.spectrogram = SpectrogramAnalysis(**self.spectro_kwargs)
        self.decomposition = None
        self.periodes_detectees = []
    def fit(self, X, y=None):
        # Étape 1 : détecter les périodes dominantes
        self.spectrogram.fit(X[self.target_column])
        self.spectrogram.transform(X[self.target_column])  # pour alimenter .dominant_periodes_
        self.periodes_detectees = self.spectrogram.dominant_periodes.astype(int).tolist()[0:2]
        self.periodes_detectees = np.array(list(set(self.periodes_detectees) & set([11,14,15,22,44]))) # Limitation à des périodes précises
        if (len(self.periodes_detectees) == 0):
            self.periodes_detectees = [44]
            print("On prend par défaut 44")
        
        # Étape 2 : construire la classe de décomposition avec les périodes détectées
        self.decomposition = DecompositionSerieTemporelle(
            target_column=self.target_column,
            forme=self.forme,
            periodes=self.periodes_detectees
        )
        self.decomposition.fit(X)
        return self
    def transform(self, X):
        # Transformation : appliquer la décomposition 
        return self.decomposition.transform(X)

    def plot(self):
        self.decomposition.plot_decomposition()
