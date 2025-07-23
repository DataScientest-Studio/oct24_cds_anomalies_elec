import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras import Input, backend as K
import os
import gc

class LSTMModel(BaseEstimator, RegressorMixin):
    def __init__(self, 
                 window_size=48, 
                 n_neurons=128, 
                 epochs=100, 
                 batch_size=64, 
                 patience=6, 
                 factor=0.1,
                 loss="mean_absolute_error",
                 min_delta = 0.001,
                 nbfoldcv=5, 
                 optimize_architecture=False, 
                 optimize_lr=True, 
                 use_grid_search=False,
                 save_path=None,
                 activation = 'relu'):

        self.window_size = window_size
        self.model = None
        self.n_neurons = n_neurons
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimize_architecture = optimize_architecture
        self.optimize_lr = optimize_lr
        self.patience = patience
        self.factor = factor
        self.loss = loss
        self.min_delta = min_delta
        self.use_grid_search = use_grid_search
        self.nbfoldcv = nbfoldcv
        self.save_path = save_path
        self.n_features = None
        self.activation = activation

    def fit(self, X, y=None):
        
            

        # Séparation des features et de la target concaténée
        self.X = X[:, :, :-1]
        self.y = X[:, 0, -1]
        self.n_features = self.X.shape[2]
        



        callbacks = [
            EarlyStopping(monitor='val_loss',
                          patience=self.patience,
                          min_delta=self.min_delta, #1e-3,
                          restore_best_weights=True,
                          verbose=1) 
                    ]
        
        if self.save_path is not None: # si un nom de fichier pour sauvegarder le modèle est précisé
            callbacks.append(ModelCheckpoint(filepath=self.save_path,
                                             monitor='val_loss',
                                             save_best_only=True,
                                             verbose=1
                                             )
                                 )


        if self.optimize_lr:
            callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=self.factor, patience=self.patience//4, min_lr=1e-6))
        if self.use_grid_search:
            self.model = self._grid_search_lstm(self.X, self.y)
        
        if self.optimize_architecture:
            self.model = self._optimize_lstm(self.X, self.y, callbacks)
        elif self.save_path is not None and os.path.exists(self.save_path):
              self.model = load_model(self.save_path)
        else:
            self.model = Sequential()
            self.model.add(Input(shape=(self.X.shape[1], self.X.shape[2])))
            self.model.add(LSTM(self.n_neurons, activation=self.activation,return_sequences=True))
            self.model.add(LSTM(self.n_neurons, activation=self.activation, return_sequences=True))
            self.model.add(LSTM(self.n_neurons,  activation=self.activation))
            self.model.add(Dense(units=self.n_neurons, activation="relu"))
            self.model.add(Dense(1))
        


        self.model.compile(optimizer=Adam(learning_rate=5e-4), loss=self.loss)
        self.model.fit(self.X, self.y, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.2, verbose=1, callbacks=callbacks)
        #if self.save_path is not None:
            #self.model.save(self.save_path)


        self.is_fitted_ = True
        #print(self.model.summary())
        
        gc.collect()
        K.clear_session()
        return self

    def _optimize_lstm(self, X_scaled, y_scaled, callbacks):
        best_model = None
        best_score = float('inf')
        tscv = TimeSeriesSplit(n_splits=self.nbfoldcv)

        #for train_index, test_index in tscv.split(X_scaled):
            #X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            #y_train, y_test = y_scaled[train_index], y_scaled[test_index]
        train_index, val_index = next(tscv.split(X_scaled))

        X_train, X_val = X_scaled[train_index], X_scaled[val_index]
        y_train, y_val = y_scaled[train_index], y_scaled[val_index]
            
        for n_neurons in [64,128,256]:
                
            for n_layers in [1,2,3]:
                
                model = Sequential()
                for i in range(n_layers):
                    return_seq = i < (n_layers - 1)
                    if i == 0:
                        model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
                        model.add(LSTM(n_neurons, activation=self.activation, return_sequences=return_seq))
                    else:
                        model.add(LSTM(n_neurons, activation=self.activation, return_sequences=return_seq))
                    
                model.add(Dense(units=n_neurons, activation="relu"))
                model.add(Dense(1))
                model.compile(optimizer=Adam(), loss=self.loss) # Adam (lr = ?)
                model.fit(X_train, y_train, epochs=5, batch_size=self.batch_size, verbose=0, callbacks=callbacks)
                loss = model.evaluate(X_val, y_val, verbose=0)
                if loss < best_score:
                    best_score = loss
                    best_model = model
                    n_neurons_best_architecture = n_neurons
                    n_layers_best_architecture = n_layers
                    
                gc.collect()
                K.clear_session()

        print(f"Best architecture - Neurons: {n_neurons_best_architecture}, Layers: {n_layers_best_architecture}, Loss: {best_score}")
        return best_model

    def _grid_search_lstm(self, X_scaled, y_scaled):
        best_score = float('inf')
        best_model = None
        tscv = TimeSeriesSplit(n_splits=self.nbfoldcv)

        for train_index, test_index in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = y_scaled[train_index], y_scaled[test_index]

         
            for lr in [0.001, 0.01, 0.1]:
                model = self.model
                model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
                model.fit(X_train, y_train, epochs=5, batch_size=self.batch_size, verbose=0)
                loss = model.evaluate(X_test, y_test, verbose=0)
                if loss < best_score:
                    best_score = loss
                    best_model = model
                print(f"Best Grid Search - LR: {lr}, Loss: {best_score}")
                gc.collect()
                K.clear_session()

        
        return best_model

    def predict(self, X):
        if self.model is None:
            if os.path.exists(self.save_path):
                self.model = load_model(self.save_path)
            else:
                raise ValueError("Le modèle n'a pas été ajusté. Utilisez fit() d'abord.")
        if X.shape[2] == self.n_features:
            # déjà les features uniquement
            X_seq = X
        if X.shape[2] == self.n_features + 1:
            # on est en mode entraînement-like → découpe de y_seq
            X_seq = X[:, :, :-1]   
        elif X.shape[2] == self.n_features:
            # déjà les features uniquement
            X_seq = X
        else: 
            raise ValueError(f"Entrée inattendue : la dernière dimension doit être >{self.n_features}")
        
        predictions = self.model.predict(X_seq, verbose=0)
        gc.collect()
        K.clear_session()
        return predictions.flatten()

    def plot_predictions(self, test_data, y_test):
        if self.model is None:
            raise ValueError("Le modèle n'a pas été ajusté. Utilisez fit() d'abord.")
        predictions = self.predict(test_data)

        plt.figure(figsize=(12, 6))
        plt.plot(y_test, label='Test Data', color='green')
        plt.plot(predictions, label='Predictions', color='red')
        plt.title('LSTM Model Predictions')
        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.legend()
        plt.show()
        gc.collect()
        K.clear_session()
