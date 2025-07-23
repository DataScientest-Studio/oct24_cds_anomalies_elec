# ------------------------------------------------------------------------------------------------------------------------------------------------
# 
# Modèle de prédiction de la consommation électrique en France 
# Les données sont issues d'enedis 
# Ces données contiennent la consommation d'un aggrégat de points de soutirage de même profile et de même puissance souscrite.
# Ces données ont été agrégés avec des données méteo : température, humidité, rayonnement solaire et force des vents
# Le modèle établi consiste à 
#          1. effectuer une analyse spectrale de la consommation en déterminant les fréquences (saisonnalités) de celle-ci
#          2. effectuer la décompostion de la consommation en tant que série temporelle en utilisant ces fréquences pour extraire les saisonnalités, la tendance et le résidus. 
#          3. pour chaque  composante saisonalité un modèle SARIMA es conçu
#          4. le résidus est estimé par une modèle LSTM multicouche en fonction des résidus des variable exogènes
#          5. la composante tendance est estimé par LSTM multicouche  en fonction des tendances des variable exogènes
# Les métriques utilisées sont  
#          1. MAPE : Mean Absolute Percentage Error
#          2. MAE : Mean Absolute Error
#          3. RMSE  : Root Mean Square Error 
#  Auteur : Y.S
# ------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------
# Les modules externes nécessaires 
# ------------------------------------------------------------------------------------------------------------------------------------------------

 # Tensorflow
import tensorflow as tf

from tensorflow import keras
# initialisation du seed ==> suite à un problème d'instabilité 
tf.keras.utils.set_random_seed(42) 
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# OS

import os

import requests
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gzip



import datetime
import time





import pandas as pd
import numpy as np



# matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pylab

# Scipy
import scipy.stats as stats
from scipy.stats import shapiro
from scipy.stats import ks_2samp
from scipy.signal.windows import get_window

# Stat
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


# ARIMA
from pmdarima.arima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace import sarimax



# Décomposition série temporelle

from statsmodels.tsa.seasonal import seasonal_decompose 


from tensorflow.keras.models import load_model



#split data
from sklearn.model_selection import train_test_split

#Pipeline
 
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer


#Normalisation
from sklearn import preprocessing


#Gestion de la mémoire
import gc
from tensorflow.keras import backend as K


# Métriques 
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error




# Pkg pour Sauvegarde des modèles
import joblib
import pickle



# expression réguière
import re
# ------------------------------------------------------------------------------------------------------------------------------------------------
# Les modules pour fixer les instablités numériques SARIMAX et compagnies
# ------------------------------------------------------------------------------------------------------------------------------------------------
# Fixer les seeds de tous les générateurs aléatoires pour reproductibilité
import random

# Fixer la seed globale pour reproductibilité complète
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Stabiliser les threads BLAS / OpenMP (évite les instabilités de SARIMAX / scipy)

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# TensorFlow : rendre les opérations déterministes (attention: pas supporté sur tous les GPU)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Gestion mémoire TensorFlow et Garbage Collector
# Limite l'utilisation mémoire GPU (évite l'allocation dynamique trop agressive)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
# Clean TensorFlow session + garbage collector avant chaque gros fit()
def reset_tensorflow_session():
    tf.keras.backend.clear_session()
    gc.collect()


# ------------------------------------------------------------------------------------------------------------------------------------------------
# Les modules developpés pour le projet : chargement avec mise à jour
# ------------------------------------------------------------------------------------------------------------------------------------------------

from . import decomposition_serie_temporelle
from . import analyse_spectrale 
from . import pipeline_analyse_spectrale_decomposition 
from . import analyse_et_sarima
from . import sarimax_model
from . import sarimax_model_fitted
from . import lstm_model
from . import sequence_transformer
from . import inversion_transformer
from . import normalisation_transformation
from . import wrapper_pour_lstm
from . import realigner
# pour la mise à jour
import importlib
importlib.reload(decomposition_serie_temporelle)
importlib.reload(analyse_spectrale)
importlib.reload(pipeline_analyse_spectrale_decomposition)
importlib.reload(lstm_model)
importlib.reload(sarimax_model)
importlib.reload(sarimax_model_fitted)
importlib.reload(analyse_et_sarima)
importlib.reload(sequence_transformer)
importlib.reload(inversion_transformer)
importlib.reload(normalisation_transformation)
importlib.reload(wrapper_pour_lstm)
importlib.reload(realigner)

from .decomposition_serie_temporelle import DecompositionSerieTemporelle
from .analyse_spectrale import SpectrogramAnalysis
from .sarimax_model import SARIMAModel
from .sarimax_model_fitted import SARIMAModelFitted

from .lstm_model import LSTMModel
from .pipeline_analyse_spectrale_decomposition import SpectroDecompPipeline
from .analyse_et_sarima import SpectrogramToSARIMAPipeline
from .sequence_transformer import SequenceTransformer
from .inversion_transformer import InversionTransformer
from .normalisation_transformation import NormalisationTransformer
from .wrapper_pour_lstm import WrapperforLSTM
from .realigner import ReAligner 
 

# ------------------------------------------------------------------------------------------------------------------------------------------------
# Dossier pour le téléchargement des fichiers et extraction du dataframe 
# ------------------------------------------------------------------------------------------------------------------------------------------------

# dossier contenant les modèles 
folder_all_models = r"D:\MesDocuments\Formation\DataScientist_PSL\Projet\BD\models" # repertoire de la base de données
folder_models= os.path.join(folder_all_models, "ARA") # 


# ------------------------------------------------------------------------------------------------------------------------------------------------
# Constantes et Variables globales servant de paramètres par défaut pour les constructeurs 
# ------------------------------------------------------------------------------------------------------------------------------------------------
TARGET  = 'Total énergie soutirée (Wh)'


COLUMNS_A_DECOMPOSER = list(['Total énergie soutirée (Wh)', 'T_moyenne', 'U_moyenne', 'Rayonnement solaire global (W/m2)']) 
FOMRES_DECOMPOSITION =  list(["multiplicative","multiplicative","multiplicative", "multiplicative"])
COLUMNS_TENDANCE = list(['Total énergie soutirée (Wh)_tendance','T_moyenne_tendance', 'U_moyenne_tendance', 'Rayonnement solaire global (W/m2)_tendance'])
COLUMNS_RESIDUEL= list(['Total énergie soutirée (Wh)_residuel','T_moyenne_residuel', 'U_moyenne_residuel', 'Rayonnement solaire global (W/m2)_residuel']) 
COLUMNS_TO_INVERSE_TENDANCE = list(['T_moyenne_tendance', 'U_moyenne_tendance', 'Rayonnement solaire global (W/m2)_tendance'])
COLUMNS_TO_INVERSE_RESIDUEL = list(['T_moyenne_residuel', 'U_moyenne_residuel', 'Rayonnement solaire global (W/m2)_residuel'])
FS = 1/1800 # fréquence d'échantillonnage pas  = 30 minutes
NOMBRE_POINTS_PAR_SEGMENT = 30*48  # pour l'analyse spectrale une semaine
OVERLAP = 12 # Recouvrement entre fenêtres
WINDOWS_SIZE = 24 # Taile de fenêtre pour LSTM
NB_PAS_JOUR = 48 # pas  = 30 minutes ==> 48 par jour
NOMBRE_JOUR_PREDICTION = 7 #a évaluer
NOMBRE_JOUR_TRAIN = 365 # une année pour le train
NOMBRE_JOUR_TOTAL = NOMBRE_JOUR_TRAIN + NOMBRE_JOUR_PREDICTION
TEST_PROPORTION = NOMBRE_JOUR_PREDICTION / NOMBRE_JOUR_TOTAL
TRAIN_SIZE = NB_PAS_JOUR*NOMBRE_JOUR_TRAIN
TOTAL_SIZE = NB_PAS_JOUR*NOMBRE_JOUR_TOTAL
TOTAL_SIZE_SARIMAX = 90 * NB_PAS_JOUR # trois mois pour entrainer le sarimax ce qui permet d'éviter l'explosion de la mémoir pour le filtre de Kalman
THRESHOLD = 0.3 # Seuli de détection pour le spectre
NB_COMPOSANTES_SPECTRALES = 2 # Le nombre de composante spectrale à garder


# Paramètres de la décomposition spectrale
spectro_params_default ={"fs": FS,                                     # Fréquence d'échantillonnage (1pas = 30 minutes :  Ts = 1800s, Fs = 1/1800 Hz)
                         "window": "hann",                             # Fenêtre de Hann 
                         "nperseg": NOMBRE_POINTS_PAR_SEGMENT,         # Longueur de la fenêtre d'analyse spectrale
                         "noverlap": OVERLAP,                          # Recouvrement entre fenêtres
                         "threshold": THRESHOLD                           # Seuil élevé pour ne détecter qu'une période dominante
                        } 
                        
# Paramètres pour LSTM : prédiction de la composante tendance
lstm_params_tendance =  { 'window_size' : WINDOWS_SIZE, 
                         'n_neurons': 256,  
                         'factor' : 0.1,
                         'patience' : 30,
                         'epochs' : 100, 
                         'batch_size' : 32, 
                         'loss' : "mean_absolute_percentage_error",#"mean_absolute_error",
                         'min_delta' : 5e-3,
                         'nbfoldcv' : 5, 
                         'optimize_architecture':True, # Pour chercher l'architecture optimale
                         'optimize_lr':True,           # Pour chehercher le taux optimal
                         'use_grid_search':False,      # Utilsation du grid_search pour les hyperparamètres
                         'save_path' : None,           # 'best_lstm_model_tendance.keras'
                         'activation': 'relu'  
                        }
# Paramètres pour LSTM : prédiction de la composante résisiduelle                        
lstm_params_residuel =  { 'window_size' : WINDOWS_SIZE, 
                         'n_neurons': 256,  
                         'factor' : 0.1,
                         'patience' : 30,
                         'epochs' : 100, 
                         'batch_size' : 32, 
                         'loss' : "mean_absolute_error", #"mean_absolute_percentage_error",
                         'min_delta' : 5e-3,
                         'nbfoldcv' : 5, 
                         'optimize_architecture':True,  # Pour chercher l'architecture optimale
                         'optimize_lr':True,            # Pour chehercher le taux optimal
                         'use_grid_search':False,       # Utilsation du grid_search pour les hyperparamètres
                         'save_path' : None,            # 'best_lstm_model_tendance.keras' 
                         'activation': 'relu' #'tanh'
                        }


COLUMNS = ["T_moyenne", 
               "U_moyenne",
               "FF_moyenne",
               "Rayonnement solaire global (W/m2)", 
               "Nb points soutirage" ,
               "Total énergie soutirée (Wh)"
               ]


# ------------------------------------------------------------------------------------------------------------------------------------------------
# Fonctions auxilaires pour le prétraitement 
# ------------------------------------------------------------------------------------------------------------------------------------------------
# Nettoyage des nom de profil et région et puissance dess tout caractère spécial pour nommer les modèles

def clean_filename_part(s):
    """
    Nettoie une chaîne pour un nom de fichier :
    - Autorise lettres (y compris accentuées), chiffres, _ et -
    - Remplace les autres caractères par _
    - Compacte les _ consécutifs en un seul
    """
    # Remplace les caractères interdits par _
    s = re.sub(r'[^\w-]', '_', s, flags=re.UNICODE)
    # Compacte les __ en _
    s = re.sub(r'_+', '_', s)
    return s


# Pour régler les irrégularité d'indexe dans les données en tant que séries temporelles
def force_datetime_index(df, freq='30min', start_default = '2023-01-01'):
    """
    Force un DataFrame à avoir un DatetimeIndex régulier.

    Parameters
    ----------
    df : pd.DataFrame ou pd.Series
        Données d'entrée sans index temporel.
    freq : str
        Fréquence du pas temporel (ex : '30min', '1H').
    start_time : str or pd.Timestamp
        Date de départ pour créer l'index.

    Returns
    -------
    df_copy : pd.DataFrame or pd.Series
        Données avec un DatetimeIndex régulier.
    """
    h_str = df['h'].astype(int).astype(str).str.zfill(2)
    mn_str = df['mn'].astype(int).astype(str).str.zfill(2)

    date = pd.to_datetime(df['date'] + ' ' + h_str + ':' + mn_str, format='%Y-%m-%d %H:%M')
    #date = pd.to_datetime(df['date'] + ' ' + df['h'].astype(str) + ':' + df['mn'].astype(str))
    
    if df['date'].min() is pd.NaT:
        start_time = date.min() or start_default
    else:
        start_time = start_default

    df_copy = df.copy()
    new_index = pd.date_range(start=start_time, periods=len(df_copy), freq=freq)
    df_copy.index = new_index
    return df_copy

def imputer_series(s, method='ffill', window=3):
    """
    Impute les valeurs manquantes d'une série temporelle.

    Parameters
    ----------
    s : pd.Series
        Série temporelle avec un DatetimeIndex.
    method : str
        Méthode d'imputation : 'interpolate', 'ffill', 'bfill', 'rolling'.
    window : int
        Taille de la fenêtre pour la moyenne glissante (si method='rolling').

    Returns
    -------
    s_filled : pd.Series
        Série avec trous imputés.
    """
    s = s.sort_index()

    if method == 'ffill':
        return s.ffill()
    elif method == 'bfill':
        return s.bfill()
    elif method == 'rolling':
        return s.rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill')
    else:
        raise ValueError("Méthode d'imputation non reconnue : utilisez 'interpolate', 'ffill', 'bfill' ou 'rolling'.")

# Pour la sépartion des données d'entrainement et de test en respectant l'ordre chronologique
def split_time_series(df, TEST_PROPORTION=0.02):
    """
    Découpe un DataFrame temporel en train/test sans mélanger l’ordre.
    
    Args:
        df (pd.DataFrame): données temporelles (index = datetime).
        TEST_PROPORTION (float): proportion à réserver pour le test.
        
    Returns:
        df_train, df_test
    """
    split_idx = int(len(df) * (1 - TEST_PROPORTION))
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]
    return df_train, df_test     

# ------------------------------------------------------------------------------------------------------------------------------------------------
# Step 1 Pipeline 1 chainage d'analyse spectrale et décomposition  de la consommation
# ------------------------------------------------------------------------------------------------------------------------------------------------
# Cette fonction effectue une première analyse spectrale et retourne les périodes détéctées dans une série temporelle 
# Elle extrait les deux périodes dominantes pour la décomposition : demi-journalier (alternance jour/nuit) et journalier

def premiere_analyse(df,spectro_params = None, column_target = None):
        column_target = column_target 
        spectro_params = spectro_params 
        spectrogram_analyzer = SpectrogramAnalysis(**spectro_params) 
        y = df[column_target]
        spectrogram_analyzer.fit(y)
        periodes = spectrogram_analyzer.transform(y) 
        return periodes.values.flatten().astype(int)[0:NB_COMPOSANTES_SPECTRALES]
        #return spectrogram_analyzer.dominant_periodes.astype(int).tolist()[0:2] # les deux premières

# Pipeline  de la première étape : analyse spectrale et décomposition  de la consommation
# Construction de la première pipeline 
def constructeur_pipeline_etape_0(target_column, liste_columns, liste_forme_in, periodes, spectro_params = None):
    spectro_params = spectro_params 

    analyse_decomposition_columns = {}
    for column, methode in zip(liste_columns, liste_forme_in): #liste des columns et la forme pour la décomposition : multiplicative ou additive
        
        if column == target_column:
            analyse_decomposition_columns[f"analyse_decomposition_{column}"] = SpectroDecompPipeline(target_column = column,
                                                                                                     forme = methode, #'multiplicative' ou 'additive
                                                                                                     spectro_kwargs= spectro_params)
        else : 
            analyse_decomposition_columns[f"analyse_decomposition_{column}"] = DecompositionSerieTemporelle(target_column=column,
                                                                                                            forme=methode,
                                                                                                            periodes=periodes
                                                                                                            )
    
    pipeline_etape_1 = Pipeline(steps=list(analyse_decomposition_columns.items()))
    return pipeline_etape_1

# ------------------------------------------------------------------------------------------------------------------------------------------------
# Step 2 Pipeline 2 chainage d'analyse spectrale et décomposition  pour les composantes de saisonnalités, tendance et résidus 
# ------------------------------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------------------------------
# Pipeline lstm pour la composante tendance Yt
# on a constaté que Yt est corrélée aux inverses des tendances de la température et du rayonnement; i.e. 1/Tt et 1/Rt 
# Pipeline de trois opérations : inversion des colonnes 'T' et 'R', normalisation, création de séaunce pour alimenter lstm et modèle lstm
# ------------------------------------------------------------------------------------------------------------------------------------------------

# def constructeur_pipeline_composante_lstm(lstm_params=None, scaler_method=None, columns_to_inverse = None, column_target = None):
    
#     lstm_params = lstm_params 
#     columns_to_inverse = columns_to_inverse 
#     column_target = column_target 
#     scaler_method = scaler_method


#     inversion = InversionTransformer(columns_to_inverse=columns_to_inverse)
#     #print(column_target)
#     scaler = NormalisationTransformer(scaler=scaler_method, column_target=column_target) # Normalisation
    
#     create_sequences = SequenceTransformer(window_size=lstm_params['window_size'],column_target=column_target) # Séquencement pour LSTM

#     lstm_model = LSTMModel( **lstm_params) # Modèle LSTM

#     pipeline_lstm = Pipeline(steps = [
#         ('inversion',inversion ),
#         ('normalize', scaler),
#         ('seq_transform', create_sequences),
#         ('lstm', lstm_model)
#         ])
#     return pipeline_lstm


# ------------------------------------------------------------
# Constructeur de pipelines SARIMA par composante périodique + LSTM
# ------------------------------------------------------------
# Pour chaque période détectée (via spectrogramme), on crée une pipeline SARIMA
# Les paramètres sont fixés via spectro_params et sarima_params
# Dans cette version sans pipeline on appelle pour chaque composant un modèle sarima
sarima_params_composantes_periodiques= {
        "research_best_model": True,# auto-ajustement des hyperparamètres
        "is_stationary" : False, 
        "index_start" : 0 #windows_size
    }


# ------------------------------------------------------------------------------------------------------------------------------------------------
# traitement d'une configuration profile-puissance
# ------------------------------------------------------------------------------------------------------------------------------------------------
def traiter_profile_puissance(df_profile_puissance , profile, puissance, reg):

    try:
        # des graines pour stabilisé les noyau tf, np
        seed = 42 # figé pour la démo
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        tf.random.set_seed(seed)
 
        INDEX_DEBUT = NB_PAS_JOUR * np.random.randint(365) # debut aléatoire à chaque simulation
        print("on traite ...", reg)
        print(INDEX_DEBUT)

 
        

        if df_profile_puissance.empty:
            print(f"Données vides pour profil={profile}, puissance={puissance}")
            return None
        
        #----------------------------------------------------------------------------------------------------------------------------
        # Prétraitement qui complète le netoyage et le traitement déjà effectués lors de la création de la base de données
        # 1. On ne garde que les colonnes température, force du vent FF, humidité U, Rayonnement R, nombre de points de soutirage et consom 
        # (cf. étude et analyse dede corrélation)
        # 2. régler les irrégularité dans les dates en ajoutant les dates manquantes
        # 3. Imputer les valeurs manquantes d'une série temporelle.
        #----------------------------------------------------------------------------------------------------------------------------
        # On force les données à avoir des indexes continues : nécessaire pour les estimations
        df_1 = force_datetime_index(df_profile_puissance, freq='30min')
        #print("apres force_dateime", len(df_1))
        # On complète les éventuels trous crées lors de l'étape précédente
        df_2 = imputer_series(df_1, method='ffill', window=3)

        #print('après imputer', len(df_2))
        # On prend une année + partie de tester (1 jours à 7 jours) 
        #==> une évaluation de limite sera étudié 
        sub_df =  df_2[COLUMNS].iloc[INDEX_DEBUT:INDEX_DEBUT+TOTAL_SIZE, :]# On se limite à une année pour l'entrainement       
        # On divise par le nombre de points de soutirage : on prédit la moyenne    
        sub_df['Total énergie soutirée (Wh)'] = sub_df['Total énergie soutirée (Wh)']/sub_df['Nb points soutirage']
        
        # Nettoayge
        del df_1, df_2 
        
        # Initialiser et ajuster l'analyse du spectrogramme pour detecter les périodes 
        # L'analyse globale pour toutes les régions, profiles et puissances a permis d'identifier la liste des périodes < une journées (= 48 pas) 
        # L =[11, 14, 15 ,22,44]
        periodes =  premiere_analyse(sub_df,
                                    spectro_params_default,
                                    column_target = 'Total énergie soutirée (Wh)')
        
        periodes = np.array(list(set(periodes) & set([11,14,15,22,44])))
        if (len(periodes) == 0):
            periodes = [44]
            print(f"Aucune période détectée pour profil={profile}, puissance={puissance}")
            print("On prend par défaut 44")
        # appel des constructeurs de pipeline  
        # pipeline 0 : analyse spectrale + décomposition 
        pipeline_1 = constructeur_pipeline_etape_0( target_column = 'Total énergie soutirée (Wh)', 
                                                liste_columns = COLUMNS_A_DECOMPOSER, 
                                                liste_forme_in = FOMRES_DECOMPOSITION, 
                                                periodes = periodes,
                                                spectro_params = spectro_params_default)
                
        #####################################################################################################################################                                
        # Séparation des données après le passage par le premier étage de décomposition
        # On décompose les données d'entrainements
        #####################################################################################################################################
        reset_tensorflow_session()
        fitted_pipeline_1 = pipeline_1.fit(sub_df)
        sub_df_decomposed  = fitted_pipeline_1.transform(sub_df) 
        X_train, X_test = split_time_series(sub_df_decomposed, TEST_PROPORTION = TEST_PROPORTION ) 
        # Pour lstm on prend une partie du train --> alignement avec sarima
        X_test_append = pd.concat([X_train.iloc[-WINDOWS_SIZE:], X_test])
        
        # fit et prédiction de la composante saisonnalité 1
        X_prediction_composant = {} 

        #####################################################################################################################################
        # # Reconstitue les modèles à partir des fichiers sauvegardés.
        # - Détecte automatiquement les périodes des modèles SARIMAX.
        # - Restaure les pipelines et LSTM pour tendance et résiduels.
        
        # Retourne un dictionnaire contenant :
        # - 'saisonnalite' : dict[periode] → dict info modèle SARIMAX
        # - 'tendance' / 'tendance_lstm' : pipeline + keras model
        # - 'residuel' / 'residuel_lstm' : pipeline + keras model
        # 
        # ##################################################################################################################################### 
 
       
        safe_profile = clean_filename_part(profile)
        safe_puissance = clean_filename_part(puissance)
        safe_region = clean_filename_part(reg)

        models = {
            'saisonnalite': {},
            'tendance': None,
            'tendance_lstm': None,
            'residuel': None,
            'residuel_lstm': None
        }

        df_result = pd.DataFrame({})
        estimateur_composantes_periodiques = {}             
        for periode in periodes:
            path_model = os.path.join(folder_models, f"model_sarimax_saisonnalite_{periode}_{safe_profile}_{safe_puissance}_{safe_region}.sm")
            if not os.path.exists(path_model):
                raise ValueError("chemin inexistant.")
            

           
            with open(path_model, 'rb') as f:
                model_info = pickle.load(f)


            params = model_info["params"]

            if params is None:
                raise ValueError("Paramètres SARIMAX manquants.")

            if isinstance(params, pd.Series):
                params = np.array(params.values)
            elif not isinstance(params, np.ndarray):
                params = np.array(params)

            
            
            sarima_params_composantes = {
            'period' : periode,
            'order' : model_info['order'],
            'seasonal_order' : model_info['seasonal_order'],
            'model_param' : params,
            'trend' : model_info['trend'],
            'enforce_stationarity' : False,
            'enforce_invertibility' : False
            }  
            print(sarima_params_composantes)  
            estimateur_composantes_periodiques[periode] = SARIMAModelFitted(**sarima_params_composantes)
            estimateur_composantes_periodiques[periode].fit(X_train[f"{TARGET}_saisonnalite_{int(periode)}"].iloc[-TOTAL_SIZE_SARIMAX:].asfreq('30T')) # on se limte à trois mois pour ne pas saturer la memoire
            X_prediction_composant[periode] = estimateur_composantes_periodiques[periode].predict(X_test[f"{TARGET}_saisonnalite_{int(periode)}"]) 
            print('fin prédiction saisonnalite : ', periode, '...')

            # Nettoyage de la mémoire
            K.clear_session()
            gc.collect()
            
            #del estimateur_composantes_periodiques[periode]
            mape_s = mean_absolute_percentage_error(X_test[f"{TARGET}_saisonnalite_{int(periode)}"], X_prediction_composant[periode])
            mae_s = mean_absolute_error(X_test[f"{TARGET}_saisonnalite_{int(periode)}"], X_prediction_composant[periode])
            rmse_s =np.sqrt(mean_squared_error(X_test[f"{TARGET}_saisonnalite_{int(periode)}"], X_prediction_composant[periode]))
            print(f"saisonnalite_{int(periode)}:")
            print(f"MAPE: {mape_s:.2%}")
            print(f"MAE: {mae_s:.2}")
            print(f"RMSE: {rmse_s:.2}")

            df_result = pd.concat([df_result, pd.DataFrame({'region' : [reg], 
                'Profil' : [profile],
                'Puissance' :  [puissance],
                'Composante' : f"saisonnalite_{int(periode)}",
                'MAPE (%)' : [100*mape_s],
                'MAE (Wh)' : [mae_s],
                'RMSE (Wh)' : [rmse_s],
                })] )
            
        ######################################################################################################
        # Prediction partie résiduelle
        ######################################################################################################
        path_resid_pipeline = os.path.join(folder_models, f"model_composante_residuelle_sans_lstm_{safe_profile}_{safe_puissance}_{safe_region}.joblib")
        path_resid_lstm = os.path.join(folder_models, f"model_composante_residuelle_lstm_{safe_profile}_{safe_puissance}_{safe_region}.keras")

        if os.path.exists(path_resid_pipeline) and os.path.exists(path_resid_lstm):
            pipeline_resid = joblib.load(path_resid_pipeline)
            model_resid_lstm = load_model(path_resid_lstm)
            pipeline_resid.named_steps['lstm'].model = model_resid_lstm
        
            models['residuel'] = pipeline_resid
            models['residuel_lstm'] = model_resid_lstm
          

  
        # Prediction
        X_test_input_residuel  = X_test_append[COLUMNS_RESIDUEL]
        X_prediction_composant['residuel'] = models['residuel'].predict(X_test_input_residuel)  
        print('fin prédiction résiduel ...')
        mape_s = mean_absolute_percentage_error(X_test[f"{TARGET}_residuel"], X_prediction_composant['residuel'])
        mae_s = mean_absolute_error(X_test[f"{TARGET}_residuel"], X_prediction_composant['residuel'])
        rmse_s =np.sqrt(mean_squared_error(X_test[f"{TARGET}_residuel"], X_prediction_composant['residuel']))
        print("residuel")
        print(f"MAPE: {mape_s:.2%}")
        print(f"MAE: {mae_s:.2}")
        print(f"RMSE: {rmse_s:.2}")
        df_result = pd.concat([df_result, pd.DataFrame({'region' : [reg], 
                'Profil' : [profile],
                'Puissance' :  [puissance],
                'Composante' : 'résidu',
                'MAPE (%)' : [100*mape_s],
                'MAE (Wh)' : [mae_s],
                'RMSE (Wh)' : [rmse_s],
                })])

        ######################################################################################################
        # Prédiction partie tendancielle
        ######################################################################################################
        # --- Modèles Tendance ---
        path_tendance_pipeline = os.path.join(folder_models, f"model_composante_tendance_sans_lstm_{safe_profile}_{safe_puissance}_{safe_region}.joblib")
        path_tendance_lstm = os.path.join(folder_models, f"model_composante_tendance_lstm_{safe_profile}_{safe_puissance}_{safe_region}.keras")

        if os.path.exists(path_tendance_pipeline) and os.path.exists(path_tendance_lstm):
            pipeline_tendance = joblib.load(path_tendance_pipeline)
            model_tendance_lstm = load_model(path_tendance_lstm)
            pipeline_tendance.named_steps['lstm'].model = model_tendance_lstm

            models['tendance'] = pipeline_tendance
            models['tendance_lstm'] = model_tendance_lstm
                # Prediction
        
        X_test_input_tendance  = X_test_append[COLUMNS_TENDANCE]
        X_prediction_composant['tendance'] = models['tendance'].predict(X_test_input_tendance)  
        print('fin prédiction tendance ...')
        mape_s = mean_absolute_percentage_error(X_test[f"{TARGET}_tendance"], X_prediction_composant['tendance'])
        mae_s = mean_absolute_error(X_test[f"{TARGET}_tendance"], X_prediction_composant['tendance'])
        rmse_s =np.sqrt(mean_squared_error(X_test[f"{TARGET}_tendance"], X_prediction_composant['tendance']))
        print("tendance")
        print(f"MAPE: {mape_s:.2%}")
        print(f"MAE: {mae_s:.2}")
        print(f"RMSE: {rmse_s:.2}")
        df_result = pd.concat([df_result, pd.DataFrame({'region' : [reg], 
                'Profil' : [profile],
                'Puissance' :  [puissance],
                'Composante' : 'tendance',
                'MAPE (%)' : [100*mape_s],
                'MAE (Wh)' : [mae_s],
                'RMSE (Wh)' : [rmse_s],
                })] )


        ######################################################################################################
        # Reconstitution
        ######################################################################################################
     
        y_test = X_test[TARGET]* X_test['Nb points soutirage']
        y_prediction = X_prediction_composant['tendance'] * X_prediction_composant['residuel']*X_test['Nb points soutirage']
        for periode in periodes:
                y_prediction = y_prediction * X_prediction_composant[periode].values.flatten()

        y_prediction = pd.Series(y_prediction, index = y_test.index)
        

        ######################################################################################################
        # Evaluation par rapport aux métriques MAPE, MAE et RMSE
        ######################################################################################################
        
        mape = mean_absolute_percentage_error(y_test, y_prediction)
        mae = mean_absolute_error(y_test, y_prediction)
        rmse =np.sqrt(mean_squared_error(y_test, y_prediction))
        print(f"MAPE: {mape:.2%}")
        print(f"MAE: {mae:.2}")
        print(f"RMSE: {rmse:.2}")
          

        ######################################################################################################
        # Sauvegarde des fichiesr des scores par rapport aux métriquex sont stockés dans une df_result
        # suite au problème de caractère spéciaux pour l'enregistrement des fichier
        # on ajoute une fonction pour supprimer les caractères speciax
        ######################################################################################################
        df_result =pd.concat([df_result, pd.DataFrame({'region' : [reg], 
                'Profil' : [profile],
                'Puissance' :  [puissance],
                'Composante' : 'serie observée',
                'MAPE (%)' : [100*mape],
                'MAE (Wh)' : [mae],
                'RMSE (Wh)' : [rmse],
                })] )
     
        

        gc.collect()
        K.clear_session() 
        print("✅ Résultat prêt à retourner")
        return df_result  
    except Exception as e:
        print(f"Erreur rencontrée pour {reg} - {profile} - {puissance}: {str(e)}")
        return None
       
       