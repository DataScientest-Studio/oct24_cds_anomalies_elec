
# ------------------------------------------------------------------------------------------------------------------------------------------------
# Les modules nécessaires internes
# ------------------------------------------------------------------------------------------------------------------------------------------------


import decomposition_serie_temporelle
import analyse_spectrale 
import sarimax_model

# pour la mise à jour
import importlib
importlib.reload(decomposition_serie_temporelle)
importlib.reload(analyse_spectrale)
importlib.reload(sarimax_model)

# mise à jour des classes
from sarimax_model import SARIMAModel
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

from tensorflow.keras import backend as K
import gc

class SpectrogramToSARIMAPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, spectro_params=None, sarima_params=None):
        self.spectro_params = spectro_params or {}
        self.sarima_params = sarima_params or {}
        self.spectrogram = SpectrogramAnalysis(**self.spectro_params)
        self.sarima = None
        self.period = None

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X_fit = X.iloc[:, 0]  # Prend la première colonne
            #print('1. OK')
        elif isinstance(X, np.ndarray) and X.ndim == 2:
            X_fit = pd.Series(X[:, 0])
            #print('2. OK')
        elif isinstance(X, np.ndarray):
            X_fit = pd.Series(X)
            #print('3. OK')
        elif isinstance(X, pd.Series):
            X_fit = X
        else:
            raise ValueError("Format d'entrée non supporté pour SARIMAX.")
        # Étape 1 : détecter les périodes dominantes
        self.spectrogram.fit(X_fit)
        self.spectrogram.transform(X_fit)

        if self.spectrogram.dominant_periodes is None or len(self.spectrogram.dominant_periodes) == 0:
            raise ValueError("Aucune période dominante détectée dans l'analyse spectrale.")

        self.period = int(self.spectrogram.dominant_periodes[-1])  # utiliser la plus dominante
         

        # Étape 2 : entraîner SARIMA sur la série complète avec cette période
        self.sarima = SARIMAModel(period=self.period, **self.sarima_params)
        self.sarima.fit(X_fit)
        gc.collect()
        K.clear_session()
        return self

    def transform(self, X):
        forecast = self.sarima.predict(X)
        #self.predictions, self.conf_int = forecast.predicted_mean,  forecast.conf_int()
        return pd.DataFrame(forecast.predicted_mean.values, index=forecast.predicted_mean.index.values, columns=[f"pred_sarima_{self.period}"])
        #return forecast.predicted_mean.values.reshape(-1, 1)


