from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import TimeSeriesSplit,GridSearchCV
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import auto_arima
from tensorflow.keras import backend as K
import gc

class SARIMAModelFitted(BaseEstimator, RegressorMixin):
    """
    Estimateur compatible avec une pipeline sklearn utilisant un modèle SARIMA.
    
    - 'period': saisonnalité (ex: 48 pour données demi-heure avec saisonnalité journalière)
    - 'research_best_model': booléen pour activer la recherche des meilleurs paramètres avec auto_arima
    - 'n_splits': nombre de splits pour la validation croisée (TimeSeriesSplit)

    """

    def __init__(self, 
                 period=44, 
                 order=None, 
                 seasonal_order=None, 
                 model_param= None, 
                 trend='n',
                 enforce_stationarity = False,  
                 enforce_invertibility=False, 
                 index_start = 0
                 ):
        
        self.period = period
        self.order = order
        self.trend = trend
        self.seasonal_order = seasonal_order
        self.model_param = model_param
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.index_start = index_start

    def fit(self, X, y=None):
        self.X = X
      
        self.model = SARIMAX(self.X,
                            order=self.order,
                            seasonal_order=self.seasonal_order,
                            trend=self.trend,
                            enforce_stationarity=self.enforce_stationarity,
                            enforce_invertibility=self.enforce_invertibility,
                            initialization='approximate_diffuse'
                       )
                                   
        
        print(self.order,self.seasonal_order, self.model_param  )
       
        print("rétablissemnt du modèle fited..." , self.model, self.model_param)

        self.fitted_model = self.model.filter(self.model_param)       
        print("fin de rétablissemnt du modèle fited...",self.fitted_model.summary() )  
        return self
        
    def predict(self, X):
        """
        Prédit le même nombre de points que dans X. X peut contenir une séquence temporelle,
        ou être simplement un 'placeholder' pour indiquer combien de pas prédire.
        """
        print('predict...')
        forecast = self.fitted_model.get_forecast(steps=len(X.iloc[self.index_start:]))
        return pd.DataFrame(forecast.predicted_mean.values, index=forecast.predicted_mean.index.values, columns=[f"pred_sarima_{self.period}"])
        #return forecast
    def get_fitted_model(self):
        return self.fitted_model    
        

