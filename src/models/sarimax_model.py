from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import TimeSeriesSplit,GridSearchCV
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima.arima import auto_arima
from tensorflow.keras import backend as K
import gc

class SARIMAModel(BaseEstimator, RegressorMixin):
    """
    Estimateur compatible avec une pipeline sklearn utilisant un modèle SARIMA.
    
    - 'period': saisonnalité (ex: 48 pour données demi-heure avec saisonnalité journalière)
    - 'research_best_model': booléen pour activer la recherche des meilleurs paramètres avec auto_arima
    - 'n_splits': nombre de splits pour la validation croisée (TimeSeriesSplit)

    """
    def __init__(self, period=48, research_best_model = False, n_splits = 5, is_stationary = False, index_start = 0):
        self.period = period
        self.model = None
        self.research_best_model = research_best_model
        self.fitted_model = None
        self.n_splits = n_splits
        self.is_stationary = is_stationary
        self.index_start = index_start

    def fit(self, X, y=None):
        self.X = X
        if self.research_best_model == True:  
            
            best_score = float('+inf')
            best_model = None

            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            for train_index, test_index in tscv.split(self.X):

                train, test = self.X.iloc[train_index], self.X.iloc[test_index]
                if len(train) < 2*self.period:
                    continue
                if self.is_stationary:
                   model = auto_arima(train, 
                                    start_p=0, max_p = 2,
                                    start_q=0, max_q = 2,
                                    start_P=0, max_P = 2,
                                    start_Q=0, max_Q = 2,
                                    d=0, D=1, trace=False,
                                    with_intercept = True,
                                    seasonal=True,
                                    m = self.period,
                                    error_action='ignore',
                                    suppress_warnings=True,
                                    stepwise=False,
                                    n_jobs=12
                                    ) 
                else:
                    model = auto_arima(train, 
                                        start_p=0, max_p = 2,
                                        start_q=0, max_q = 2,
                                        start_P=0, max_P = 2,
                                        start_Q=0, max_Q = 2,
                                        d=1, D=1, trace=False,
                                        with_intercept = True,
                                        seasonal=True,
                                        m = self.period,
                                        error_action='ignore',
                                        suppress_warnings=True,
                                        stepwise=False,
                                        n_jobs=12
                                        )
                predictions = model.predict(n_periods=len(test))
                score = np.mean(np.abs(test.values - predictions))  # Mean Absolute Error
                
                    
                if score < best_score:
                    best_score = score
                    best_model = model
    
            
            # Modèle SARIMAX 
            # Récupération des ordres
            order = best_model.order            # (p,d,q)
            seasonal_order = best_model.seasonal_order  # (P,D,Q,s)
            if self.is_stationary:
                self.model = SARIMAX(self.X,
                                    order=order,
                                    seasonal_order=seasonal_order,
                                    trend='n',
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                    initialization='approximate_diffuse',
                                    low_memory=True
                                    )
            else : 
                self.model = SARIMAX(self.X,
                                    order=order,
                                    seasonal_order=seasonal_order,
                                    trend='n',
                                    enforce_stationarity=True,
                                    enforce_invertibility=True,
                                    initialization='approximate_diffuse',
                                    low_memory=True
                                    )
            
            self.fitted_model = self.model.fit(disp=False,method='powell')
            print("Le meilleur modèle", self.fitted_model.summary())

        else: # Modèle par défaut établi
            print(" Periode utilisée", self.period)
            #print(X.index)
            if self.is_stationary:
                self.model = SARIMAX(self.X,
                                    order=(1,0,1),
                                    seasonal_order=(1,1,1,self.period),
                                    trend='n',
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                    initialization='approximate_diffuse',
                                    low_memory=True)
            else: 
                #print("on passe par ici")
                self.model = SARIMAX(self.X,order=(2,1,1),
                      seasonal_order=(1,1,0,self.period),
                      enforce_stationarity=True,
                      enforce_invertibility=False,
                      initialization='approximate_diffuse',
                      low_memory=True
                      )
                #print(self.model)
            print("entrainement...")
            self.fitted_model = self.model.fit(disp=False,method='powell')
            #self.fitted_model.summary()
              
        return self
        
    def predict(self, X):
        """
        Prédit le même nombre de points que dans X. X peut contenir une séquence temporelle,
        ou être simplement un 'placeholder' pour indiquer combien de pas prédire.
        """
        print('predict...')
        forecast = self.fitted_model.get_forecast(steps=len(X.iloc[self.index_start:]))
        return forecast
        
        


