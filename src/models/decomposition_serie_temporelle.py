import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.base import BaseEstimator, TransformerMixin

class DecompositionSerieTemporelle(BaseEstimator, TransformerMixin):
    def __init__(self, target_column, periodes, forme='multiplicative'):
        self.target_column = target_column
        self.periodes = periodes
        self.forme = forme
        # Stockage pour plot/export
        self.tendance_ = None
        self.residuel_ = None
        self.saisonnalite_ = pd.DataFrame()

    def fit(self, X, y=None):
        return self  # Statelss

    def transform(self, X):
        df = X.copy()
        y = df[self.target_column].copy()

        if self.forme == 'multiplicative' and np.min(y) <= 0:
            y = y - np.min(y) + 1  # pour éviter les divisions par zéro

        saisonnalite = pd.DataFrame(index=y.index)
        tendance = None
        residuel = None

        if self.forme == 'additive':
            y_current = y.copy()
            # décompoistion en cascade
            for periode in self.periodes:
                saison_key = f'saisonnalite_{periode}'
                decomposition = seasonal_decompose(y_current, model='additive', period=periode, extrapolate_trend='freq')
                saisonnalite[saison_key] = decomposition.seasonal
                y_current = y_current - decomposition.seasonal
                tendance = decomposition.trend
                residuel = y - saisonnalite.sum(axis=1) - tendance

        
        
        # pour la form multiplicative deux versions sont proposées: 
        # En cascade y = S1*S2*S3*T*R ==> casacde
        # y = S1*T1*R1, Y=S2*T2*R2, Y=S3*T3*R3 puis déduire une nouvelle décomposition y = S1*S2*S3*T3*R où R = Y/(S1*S2*S3*T3) 
        elif self.forme == 'multiplicative':
            # Étape 1 : première décomposition sur y
            for i, periode in enumerate(self.periodes):
                decompo = seasonal_decompose(y, model='multiplicative', period=periode, extrapolate_trend='freq')
                saison_key = f'saisonnalite_{periode}'
                saisonnalite[saison_key] = decompo.seasonal

                if i == 0:
                    residuel = y / decompo.seasonal
                else:
                    residuel = residuel / decompo.seasonal

                tendance = decompo.trend  # toujours la dernière tendance

            residuel = residuel / tendance # Residuel finale
        """
        # Version en casacde
        elif self.forme == 'multiplicative':

            y_current = y.copy()

            for periode in self.periodes:
                saison_key = f'saisonnalite_{periode}'
                decomposition = seasonal_decompose(y_current, model='multiplicative', period=periode, extrapolate_trend='freq')
                saisonnalite[saison_key] = decomposition.seasonal
                y_current = y_current / decomposition.seasonal
                tendance = decomposition.trend
                residuel = y / (saisonnalite.prod(axis=1) * tendance)
        """
        # Ajout des colonnes transformées
        df[f'{self.target_column}_tendance'] = tendance
        df[f'{self.target_column}_residuel'] = residuel
        for col in saisonnalite.columns:
            df[f'{self.target_column}_{col}'] = saisonnalite[col]

        # Stockage pour plot/export
        self.tendance_ = tendance
        self.residuel_ = residuel
        self.saisonnalite_ = saisonnalite

        return df

    def plot_decomposition(self):
        if self.tendance_ is None:
            raise RuntimeError("transform() doit être appelé avant plot_decomposition().")

        n = len(self.saisonnalite_.columns) + 2
        plt.figure(figsize=(12, 2.5 * n))

        plt.subplot(n, 1, 1)
        plt.plot(self.tendance_, color='orange')
        plt.title('Tendance')

        for i, col in enumerate(self.saisonnalite_.columns, start=2):
            plt.subplot(n, 1, i)
            plt.plot(self.saisonnalite_[col])
            plt.title(col)

        plt.subplot(n, 1, n)
        plt.plot(self.residuel_, color='red')
        plt.title('Résiduel')

        plt.tight_layout()
        plt.show()

    def get_components(self):
        if self.tendance_ is None:
            raise RuntimeError("Appeler `.transform()` avant `get_components()`.")
        return {
            "tendance": self.tendance_,
            "residuel": self.residuel_,
            "saisonnalite": self.saisonnalite_
        }
