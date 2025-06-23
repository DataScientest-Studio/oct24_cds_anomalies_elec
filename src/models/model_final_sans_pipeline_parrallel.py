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



# OS

import os
import requests
import sys
import gzip

import datetime
import time





import pandas as pd
import numpy as np



# matplotlib
import matplotlib.pyplot as plt

from ipywidgets import interact, IntSlider, FloatSlider, Dropdown, Button, HBox, VBox
import ipywidgets as widgets
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

# Métriques 
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


# Décomposition série temporelle

from statsmodels.tsa.seasonal import seasonal_decompose 






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

# Gestion d'erreur / fichier log
import logging
logging.basicConfig(filename='erreurs.log', level=logging.ERROR)

#Parrallèlisme
from joblib import Parallel, delayed


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

# Configuration du Logger général du projet
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("projet_execution.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)
logger.info("Environnement initialisé correctement avec la seed %d", SEED)

try:
    from IPython import get_ipython
    if get_ipython() is None:
        from IPython import embed
        logger.info("Lancement du kernel interactif IPython")
        embed()
except ImportError:
    logger.warning("IPython non disponible")

# Configuration du Logger général du projet

from log_manager import setup_logger
# ------------------------------------------------------------------------------------------------------------------------------------------------
# Les modules developpés pour le projet : chargement avec mise à jour
# ------------------------------------------------------------------------------------------------------------------------------------------------

import decomposition_serie_temporelle
import analyse_spectrale 
import pipeline_analyse_spectrale_decomposition 
import analyse_et_sarima
import sarimax_model
import lstm_model
import sequence_transformer
import inversion_transformer
import normalisation_transformation
import wrapper_pour_lstm
import realigner
# pour la mise à jour
import importlib
importlib.reload(decomposition_serie_temporelle)
importlib.reload(analyse_spectrale)
importlib.reload(pipeline_analyse_spectrale_decomposition)
importlib.reload(lstm_model)
importlib.reload(sarimax_model)
importlib.reload(analyse_et_sarima)
importlib.reload(sequence_transformer)
importlib.reload(inversion_transformer)
importlib.reload(normalisation_transformation)
importlib.reload(wrapper_pour_lstm)
importlib.reload(realigner)

from decomposition_serie_temporelle import DecompositionSerieTemporelle
from analyse_spectrale import SpectrogramAnalysis
from sarimax_model import SARIMAModel
from lstm_model import LSTMModel
from pipeline_analyse_spectrale_decomposition import SpectroDecompPipeline
from analyse_et_sarima import SpectrogramToSARIMAPipeline
from sequence_transformer import SequenceTransformer
from inversion_transformer import InversionTransformer
from normalisation_transformation import NormalisationTransformer
from wrapper_pour_lstm import WrapperforLSTM
from realigner import ReAligner 
 

# ------------------------------------------------------------------------------------------------------------------------------------------------
# Dossier pour le téléchargement des fichiers et extraction du dataframe 
# ------------------------------------------------------------------------------------------------------------------------------------------------
folder_projet_BD = r"D:\MesDocuments\Formation\DataScientist_PSL\Projet\BD" # repertoire de la base de données
folder_BD_propre= os.path.join(folder_projet_BD, "conso-inf36-meteo-rayonnement-region-propre") # 

liste_file = os.listdir(folder_BD_propre)

# création d'un dossier pour stocker les résultats (scores)
folder_resultats = os.path.join(folder_projet_BD, "resultats") 
folder_models = os.path.join(folder_projet_BD, "models") 
if not os.path.isdir(folder_resultats):   
    os.mkdir(folder_resultats)
if not os.path.isdir(folder_models):   
    os.mkdir(folder_models)

folder_log = os.path.join(folder_projet_BD, "fichiers_log") 
if not os.path.isdir(folder_log):
    os.mkdir(folder_log)

# Initaition du logger
logger = setup_logger(log_dir=folder_log, log_file="predictions_conso_elec.log")
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

NB_COMPOSANTES_SPECTRALES = 2 # Le nombre de composante spectrale à garder
# On se restreint au profiles de maisons résenditielles :objectif initial du projet
LISTE_PROFILE = list(['RES1 (+ RES1WE)', 'RES11 (+ RES11WE)', 'RES2 (+ RES5)', 'RES2WE','RES3', 'RES4'])


# Paramètres de la décomposition spectrale
spectro_params_default ={"fs": FS,                                     # Fréquence d'échantillonnage (1pas = 30 minutes :  Ts = 1800s, Fs = 1/1800 Hz)
                         "window": "hann",                             # Fenêtre de Hann 
                         "nperseg": NOMBRE_POINTS_PAR_SEGMENT,         # Longueur de la fenêtre d'analyse spectrale
                         "noverlap": OVERLAP,                          # Recouvrement entre fenêtres
                         "threshold": 0.2                             # Seuil élevé pour ne détecter qu'une période dominante
                        } 
                        
# Paramètres pour LSTM : prédiction de la composante tendance
lstm_params_tendance =  { 'window_size' : WINDOWS_SIZE, 
                         'n_neurons': 256,  
                         'factor' : 0.1,
                         'epochs' : 100, 
                         'batch_size' : 32, 
                         'loss' : "mean_absolute_error",
                         'nbfoldcv' : 3, 
                         'optimize_architecture':True, # Pour chercher l'architecture optimale
                         'optimize_lr':True,           # Pour chehercher le taux optimal
                         'use_grid_search':False,      # Utilsation du grid_search pour les hyperparamètres
                         #'save_path' : None           # 'best_lstm_model_tendance.keras'  
                        }
# Paramètres pour LSTM : prédiction de la composante résisiduelle                        
lstm_params_residuel =  { 'window_size' : WINDOWS_SIZE, 
                         'n_neurons': 256,  
                         'factor' : 0.1,
                         'epochs' : 100, 
                         'batch_size' : 32, 
                         'loss' : "mean_absolute_error",
                         'nbfoldcv' : 3, 
                         'optimize_architecture':False,  # Pour chercher l'architecture optimale
                         'optimize_lr':True,            # Pour chehercher le taux optimal
                         'use_grid_search':False,       # Utilsation du grid_search pour les hyperparamètres
                         #'save_path' : None            # 'best_lstm_model_tendance.keras' 
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
    
# Cette fonction permet de vérifier que la prédiction ne contient pas des nan ou des valeurs abberant. 
# Ceci sert à sauter les cas qui pose problème pour continuer l'exécution et les examiner plus tard
# Ce problème a été résolue en imposant des graines d'initilisation pour les noyau keras, numpy et tensorflow
def check_prediction(y_pred, step_name=""):
    """
    Vérifie qu'une prédiction ne contient pas de NaN ou de valeurs aberrantes.
    """
    if y_pred is None:
        logging.error(f"{step_name}: prédiction retournée None")
        return False
    if np.any(pd.isna(y_pred)):
        logging.error(f"{step_name}: présence de NaN dans la prédiction")
        return False
    if np.any(np.isinf(y_pred)):
        logging.error(f"{step_name}: présence de inf ou -inf dans la prédiction")
        return False
    if np.isnan(y_pred).all():
        logging.error(f"{step_name}: toutes les valeurs sont NaN")
        return False
    return True


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

def constructeur_pipeline_composante_lstm(lstm_params=None, columns_to_inverse = None, column_target = None):
    
    lstm_params = lstm_params 
    columns_to_inverse = columns_to_inverse 
    column_target = column_target 



    inversion = InversionTransformer(columns_to_inverse)
    #print(column_target)
    scaler = NormalisationTransformer(column_target=column_target) # Normalisation
    
    create_sequences = SequenceTransformer(window_size=lstm_params['window_size'],column_target=column_target) # Séquencement pour LSTM

    lstm_model = LSTMModel( **lstm_params) # Modèle LSTM

    pipeline_lstm = Pipeline(steps = [
        ('inversion',inversion ),
        ('normalize', scaler),
        ('seq_transform', create_sequences),
        ('lstm', lstm_model)
        ])
    return pipeline_lstm

# ------------------------------------------------------------
# Constructeur de pipelines SARIMA par composante périodique + LSTM
# ------------------------------------------------------------
# Pour chaque période détectée (via spectrogramme), on crée une pipeline SARIMA
# Les paramètres sont fixés via spectro_params et sarima_params
# Dans cette version sans pipeline on appelle pour chaque composant un modèle sarima
sarima_params_composantes_periodiques= {
        "research_best_model": False,# Pas d'auto-ajustement des hyperparamètres
        "is_stationary" : False, 
        "index_start" : 0 #windows_size
    }
sarima_params_composantes_residuelle= {
        "research_best_model": False, # Pas d'auto-ajustement des hyperparamètres
        "is_stationary" : True, 
        "index_start" : 0 # windows_size
    } 



# ------------------------------------------------------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------------------------------------------------------
def traiter_profile_puissance(df_profile_puissance , profile, puissance, reg, file_out):
    try:
        seed = 42
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        INDEX_DEBUT = NB_PAS_JOUR * np.random.randint(365) # debut aléatoire à chaque simulation
        print("on traite ...", reg)
        print(len(df_profile_puissance))

 
        
        
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
        print("apres force_dateime", len(df_1))

        # On complète les éventuels trous crées lors de l'étape précédente
        df_2 = imputer_series(df_1, method='ffill', window=3)

        print('après imputer', len(df_2))
        # On prend une année + partie de tester (1 jours à 7 jours) 
        #==> une évaluation de limite sera étudié 
        sub_df =  df_2[COLUMNS].iloc[INDEX_DEBUT:INDEX_DEBUT+TOTAL_SIZE, :]# On se limite à une année pour l'entrainement
        print('après extract_culumns', len(sub_df))
        # Nettoyage
        # del df_1, df_2 
        # On divise par le nombre de points de soutirage : on prédit la moyenne
            
        sub_df['Total énergie soutirée (Wh)'] = sub_df['Total énergie soutirée (Wh)']/sub_df['Nb points soutirage']
        sub_df.to_csv('test.csv', index=False)
            
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

        # pipeline 1 : prédiction pour chaque composante
        # Ajout de la composante résiduelle
       
        lstm_params_residuel['save_path'] = os.path.join(folder_models,f"model_lstm_residuel_{profile}_{puissance}_{reg}.keras")
        pipeline_lstm_residuel = constructeur_pipeline_composante_lstm(lstm_params =lstm_params_residuel , 
                                                                    columns_to_inverse = COLUMNS_TO_INVERSE_RESIDUEL, 
                                                                    column_target = 'Total énergie soutirée (Wh)_residuel')

        # Ajout de la composante tendance
        lstm_params_tendance['save_path'] = os.path.join(folder_models,f"model_lstm_tendance_{profile}_{puissance}_{reg}.keras") 

        pipeline_lstm_tendance = constructeur_pipeline_composante_lstm(lstm_params = lstm_params_tendance, 
                                                                    columns_to_inverse = COLUMNS_TO_INVERSE_TENDANCE, 
                                                                    column_target = 'Total énergie soutirée (Wh)_tendance')
                                        
                                    
                                        
        # Séparation des données après le passage par le premier étage de décomposition
        # On décompose les données d'entrainements
        reset_tensorflow_session()
        fitted_pipeline_1 = pipeline_1.fit(sub_df)
        sub_df_decomposed  = fitted_pipeline_1.transform(sub_df) 
        X_train, X_test = split_time_series(sub_df_decomposed, TEST_PROPORTION = TEST_PROPORTION ) 
            # Pour lstm on prend une partie du train --> alignement avec sarima
        X_test_append = pd.concat([X_train.iloc[-WINDOWS_SIZE:], X_test])
        
        #Nettoyage df
        #del sub_df, fitted_pipeline_1, sub_df_decomposed

        
        ######################################################################################################
        # fit et prédiction de la composante saisonnalité 1
        X_prediction_composant = {} 
        estimateur_composantes_periodiques = {}
        ######################################################################################################
        # des pipelines pour chaque composante saisonnière détectée
        ######################################################################################################
        print('on est la')
        for periode in periodes:
            reset_tensorflow_session() # nettoyage
            estimateur_composantes_periodiques[periode] = SpectrogramToSARIMAPipeline(spectro_params=spectro_params_default, sarima_params=sarima_params_composantes_periodiques)
            estimateur_composantes_periodiques[periode].fit(X_train[f"{TARGET}_saisonnalite_{int(periode)}"].iloc[-TOTAL_SIZE_SARIMAX:].asfreq('30T')) # on se limte à trois mois pour ne pas saturer la memoire
            X_prediction_composant[periode] = estimateur_composantes_periodiques[periode].transform(X_test[f"{TARGET}_saisonnalite_{int(periode)}"]) 
            print('fin prédiction saisonnalite : ', periode, '...')
            # Enregistrement du modèle
            # ....
            # Nettoyage de la mémoire
            K.clear_session()
            gc.collect()
            del estimateur_composantes_periodiques[periode] 
            #del estimateur_composantes_periodiques[periode]
        
            
        ######################################################################################################
        # Prediction partie résiduelle
        ######################################################################################################
        X_input_residuel  = X_train[COLUMNS_RESIDUEL]
        y_target_residuel = X_train[f"{TARGET}_residuel"]

        # Fit
        reset_tensorflow_session()
        pipeline_lstm_residuel_fitted = pipeline_lstm_residuel.fit(X_input_residuel,y_target_residuel) 
        reset_tensorflow_session()
        # Préparation des autres colonnes
        X_test_input_residuel  = X_test_append[COLUMNS_RESIDUEL]
        y_test_target_residuel = X_test_append[f"{TARGET}_residuel"]
        # Prediction
        X_prediction_composant['residuel'] = pipeline_lstm_residuel_fitted.predict(X_test_input_residuel)  
        print('fin prédiction résiduel ...')
        ######################################################################################################
        # Prédiction partie tendancielle
        ######################################################################################################
        X_input_tendance = X_train[COLUMNS_TENDANCE]
        y_target_tendance =X_train[f"{TARGET}_tendance"]
        # Fit
        reset_tensorflow_session()
        pipeline_lstm_tendance_fitted = pipeline_lstm_tendance.fit(X_input_tendance, y_target_tendance)
        reset_tensorflow_session()
        # Préparation des autres colonnes
        X_test_input_tendance  = X_test_append[COLUMNS_TENDANCE]
        y_test_target_tendance = X_test_append[f"{TARGET}_tendance"]
        # Prediction
        X_prediction_composant['tendance'] = pipeline_lstm_tendance_fitted.predict(X_test_input_tendance)
        print('fin prédiction tendance ...')
            


        # Reconstitution
        y_test = X_test[TARGET]* X_test['Nb points soutirage']
        y_prediction = X_prediction_composant['tendance'] * X_prediction_composant['residuel']*X_test['Nb points soutirage']
        for periode in periodes:
                y_prediction = y_prediction * X_prediction_composant[periode].values.flatten()

        y_prediction = pd.Series(y_prediction, index = y_test.index)


        # Verification :
        if not check_prediction(y_prediction, step_name="Final Reconstruction"):
            logging.warning(f"Impossible de calculer les métriques pour le profil {profile}, puissance {puissance}.")
            return None
        

        # Si on arrive ici fait l'évaluation par rapport au métrique MAPE, MAE et RMSE
        mape = mean_absolute_percentage_error(y_test, y_prediction)
        mae = mean_absolute_error(y_test, y_prediction)
        rmse = mean_squared_error(y_test, y_prediction)
        print(f"MAPE: {mape:.2%}")
        print(f"MAE: {mae:.2}")
        print(f"RMSE: {np.sqrt(rmse):.2}")
            
        # les scores par rapport aux métriquex sont stockés dans une df_result
        df_result = pd.DataFrame({'region' : [reg], 
                'Profil' : [profile],
                'Puissance' :  [puissance],
                'MAPE (%)' : [100*mape],
                'MAE (Wh)' : [mae],
                'RMSE (Wh)' : [rmse]
                }) 
        df_result.to_csv(file_out, mode='a', header=not os.path.exists(file_out), index=False)    
        gc.collect()
        K.clear_session() 
        return df_result  
    except Exception as e:
        print(f"Erreur rencontrée pour {reg} - {profile} - {puissance}: {str(e)}")
        return None
       
       

# -----------------------------------------------
# Exécution parallèle principale
# -----------------------------------------------
if __name__ == '__main__':
    for file_name in liste_file:  # boucle séquentielle sur les régions
        file = os.path.join(folder_BD_propre, file_name)
        df = pd.read_csv(file, low_memory=False)
        print(' la vraie longueur est !!', len(df))
        reg = df['Région'].unique()[0]
        print("Nous traitons la région ....", reg)
        
        
        # sauveagrde des résultats pour chaque région dans un csv 
        filename = f"resultats_{reg}.csv" 
        file_out = os.path.join(folder_resultats, filename)    

        

        # Générer les jobs (profile, puissance)
        jobs = []
        #for profile in set(LISTE_PROFILE) & set(df['Profil'].unique()):
        for profile in df['Profil'].unique():
            liste_puissances_souscrites = df.loc[(df['Profil'] == profile), 'Plage de puissance souscrite'].unique()
            for puissance in liste_puissances_souscrites:
                df_profile_puissance = df.loc[(df['Profil']==profile) & (df['Plage de puissance souscrite']==puissance)]  # extraction de la partie de la base
                jobs.append((df_profile_puissance,profile, puissance, reg,file_out))

        # Parallélisation 
        n_jobs = os.cpu_count() - 1
        results = Parallel(n_jobs)( delayed(traiter_profile_puissance)
                                      (df_, profile_, puissance_, reg_, file_out_) 
                                      for (df_,profile_, puissance_, reg_,file_out_) in jobs
                                      )

        # Collecte des résultats
        #df_result = pd.DataFrame([res for res in results if res is not None])
        
        # Sauvegarde région
        #filename = f"resultats_{reg}.csv"
        #file_out = os.path.join(folder_resultats, filename)
        #df_result.to_csv(file_sortie, index=False)

        print("Fin du traitement pour la région:", reg)

        # Nettoyage mémoire entre chaque fichier
        K.clear_session()
        gc.collect() 

