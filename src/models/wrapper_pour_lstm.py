from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class WrapperforLSTM(BaseEstimator, TransformerMixin):
    def __init__(self, pipeline):
        """
        Wrapper pour un pipeline complet (prétraitement + LSTM),
        compatible avec ColumnTransformer (via fit/transform).
        
        Parameters:
        ----------
        pipeline : sklearn.pipeline.Pipeline
            Un pipeline dont la dernière étape est un LSTMModel avec .fit()/.predict()
        """
        self.pipeline = pipeline

    def fit(self, X, y=None):
        # On entraîne l'ensemble du pipeline avec les données
        self.pipeline.fit(X, y)
        return self

    def transform(self, X):
        y_pred = self.pipeline.predict(X)   
        
        new_index = X.index[-len(y_pred):]  # on garde uniquement les dernières lignes, de la même longueur que y_pred
        # Utilise .predict() pour produire la sortie en tant que transformation
        return pd.DataFrame({"prediction_sarima_tendance": y_pred
                          #f"borne_inf_conf_int": self.conf_int.iloc[:, 0], 
                             # #f"borne_sup_conf_int": self.conf_int.iloc[:, 1]
                             },index=new_index
                             )
    
