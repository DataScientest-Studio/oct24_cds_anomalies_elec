import streamlit as st
import pandas as pd
import numpy as np

import os
import sys
import io
from pathlib import Path
import altair as alt

# Ajoute le répertoire Scripts au sys.path pour rendre les modules importables
SCRIPTS_DIR = Path(r"D:\MesDocuments\Formation\DataScientist_PSL\Projet\Scripts")
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))



import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import contextlib

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

from sklearn.preprocessing import MinMaxScaler




# importer des modules depuis src/models/

import decomposition_serie_temporelle
import analyse_spectrale 
#import model_final_demo_prediction_streamlit
# pour la mise à jour
import importlib
importlib.reload(decomposition_serie_temporelle)
importlib.reload(analyse_spectrale)
#importlib.reload(model_final_demo_prediction_streamlit)

from decomposition_serie_temporelle import DecompositionSerieTemporelle
from analyse_spectrale import SpectrogramAnalysis
from sarimax_model_fitted import SARIMAModelFitted
#from model_final_demo_prediction_streamlit import traiter_profile_puissance

 



# --- Entête ----
def show_header():
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image("logos/dst-logo.svg", width=100)
    with col2:
        st.markdown("""
            # 🔌 Prévision de la consommation d'électricité en France
            **Projet DataScientest | Youssef SERRESTOU**
        """)

# -- Layout de la page ---        
def set_full_width():
    st.markdown("""
        <style>
            /* Supprimer les marges sur les côtés */
            .appview-container .main .block-container {
                padding-left: 2rem;
                padding-right: 2rem;
                max-width: 100% !important;
            }
        </style>
    """, unsafe_allow_html=True)




# --- Dossiers de données ---
BASE_DIR = Path(r"D:\MesDocuments\Formation\DataScientist_PSL\Projet\BD")
FOLDERS = {
    "🌦️ Météo (CSV brut)": BASE_DIR / "Meteo" / "CSV",
    "⚡ Consommation par région": BASE_DIR / "conso-inf36-region",
    "🌞 Rayonnement par région": BASE_DIR / "Meteo" / "rayonnement",
    "⚡+🌦️ Consommation-météo par région": BASE_DIR / "conso-inf36-meteo-rayonnement-region-propre"
}

FOLDERS_Fusion = {
    "⚡+🌦️ Consommation-météo par région": BASE_DIR / "conso-inf36-meteo-rayonnement-region-propre"
}
# séparateurs 
SEPARATORS = {
    "🌦️ Météo (CSV brut)": ';',
    "⚡ Consommation par région": ';',
    "🧪 Rayonnement par région": ',' ,
    "⚡+🌦️ Consommation-météo par région": ','  
}

FOLDER_RESULT = {
    "Résultats globaux": BASE_DIR / "resultats"
}
# --- Fonctions ---

@st.cache_data
def list_csv_files(folder: Path):
    """Liste les fichiers CSV dans un dossier donné."""
    return sorted([f for f in folder.glob("*.csv")])

def get_file_size(path: Path) -> float:
    """Retourne la taille d’un fichier en Mo."""
    return os.path.getsize(path) / (1024 * 1024)

@st.cache_data(show_spinner=True)
def load_csv_preview(file_path: Path, sep: str, nrows: int = 20) -> pd.DataFrame:
    """Charge les n premières lignes d’un fichier CSV avec séparateur ;"""
    if sep is not None:
        return pd.read_csv(file_path, sep=';', nrows=nrows, low_memory=False)
    else : 
        return pd.read_csv(file_path,nrows=nrows, low_memory=False)

def show_file_info(file: Path, sep: str):
    size_mb = get_file_size(file)
    st.markdown(f"**Nom :** `{file.name}` — **Taille :** {size_mb:.2f} Mo — **Séparateur :** `{sep}`")

    if st.button("📥 Charger un aperçu de 20 lignes", key=file.name):
        try:
            df = load_csv_preview(file, sep=sep)
            st.dataframe(df)
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement : {e}")

@st.cache_data(show_spinner=True)
def load_csv_entier(file_path: Path) -> pd.DataFrame:
    """Charge les n premières lignes d’un fichier CSV avec séparateur ;"""
    return pd.read_csv(file_path, low_memory=False)
 
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
# Pour régler certaines irrégularité dans les données vues en tant que séries temporelles
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


def load_and_filter_df_fusion(FOLDERS_Fusion: dict):
    st.markdown("📂 **Sélection de la série à analyser**")
    
    folder_label = st.selectbox("📁 Choisissez un répertoire :", list(FOLDERS_Fusion.keys()))
    folder_path = FOLDERS_Fusion[folder_label]

    if not folder_path.exists():
        st.error(f"❌ Le dossier `{folder_path}` n’existe pas.")
        return None

    csv_files = list_csv_files(folder_path)
    if not csv_files:
        st.warning("⚠️ Aucun fichier CSV trouvé dans ce dossier.")
        return None

    selected_file = st.selectbox("📄 Choisissez un fichier CSV :", csv_files)
    df_fusion = load_csv_entier(selected_file)

    required_cols = ["Profil", "Plage de puissance souscrite"]
    if not all(col in df_fusion.columns for col in required_cols):
        st.error("❌ Les colonnes 'Profil' et 'Plage de puissance souscrite' sont manquantes.")
        return None

    # Sélection du profil
    profils_disponibles = sorted(df_fusion["Profil"].dropna().unique())
    profil_selectionne = st.selectbox("👤 Choisissez un profil :", profils_disponibles)

    # Sélection de la plage de puissance
    puissances_disponibles = sorted(
        df_fusion[df_fusion["Profil"] == profil_selectionne]["Plage de puissance souscrite"].dropna().unique()
    )
    puissance_selectionnee = st.selectbox("⚡ Choisissez une plage de puissance :", puissances_disponibles)

    # Filtrage
    df_fusion_filtred = df_fusion[
        (df_fusion["Profil"] == profil_selectionne) &
        (df_fusion["Plage de puissance souscrite"] == puissance_selectionnee)
    ]

    st.success(f"✅ {len(df_fusion_filtred)} lignes sélectionnées pour {profil_selectionne} / {puissance_selectionnee}")

    return df_fusion_filtred
def stationnarity_test(serie, test_type='ADF'):
    if test_type == 'ADF':
        result = adfuller(serie.dropna())
        st.markdown("**Test ADF (Augmented Dickey-Fuller)**")
        st.write(f"Statistique de test : {result[0]:.4f}")
        st.write(f"p-value : {result[1]:.4f}")
        st.write(f"Nombre de retards : {result[2]}")
        st.write(f"Nombre d’observations : {result[3]}")
        st.write("Valeurs critiques :")
        for key, value in result[4].items():
            st.write(f"  - {key}: {value:.3f}")
        if result[1] < 0.05:
            st.success("✅ La série est stationnaire (p-value < 0.05)")
        else:
            st.warning("⚠️ La série est non stationnaire (p-value ≥ 0.05)")

    elif test_type == 'KPSS':
        result = kpss(serie.dropna(), regression='c', nlags="auto")
        st.markdown("**Test KPSS (Kwiatkowski–Phillips–Schmidt–Shin)**")
        st.write(f"Statistique de test : {result[0]:.4f}")
        st.write(f"p-value : {result[1]:.4f}")
        st.write(f"Nombre de lags utilisés : {result[2]}")
        st.write("Valeurs critiques :")
        for key, value in result[3].items():
            st.write(f"  - {key}: {value:.3f}")
        if result[1] < 0.05:
            st.warning("⚠️ La série est non stationnaire (p-value < 0.05)")
        else:
            st.success("✅ La série est stationnaire (p-value ≥ 0.05)")
       
def show_exploratory_analysis(df_fusion):
    # Configuration de la grille de subplots
    fig, axs = plt.subplots(ncols=4, nrows=4, figsize=(30, 30))
    fig.suptitle("Total énergie soutirée normalisée (Wh) en fonction de l'heure (h)", fontsize="x-large")

    col = 0
    row = 0

    profils = df_fusion['Profil'].unique()
    max_plots = min(len(profils), 16)

    for i, p in enumerate(profils[:max_plots]):
        df_conso_temp = df_fusion.loc[df_fusion['Profil'] == p, 
                                      ['Plage de puissance souscrite', 'h', 'Nb points soutirage',
                                       "Total énergie soutirée (Wh)", 'day_n']].copy()

        # Normalisation de la consommation
        df_conso_temp["Total énergie soutirée normalisée (Wh)"] = (
            df_conso_temp["Total énergie soutirée (Wh)"] / df_conso_temp["Nb points soutirage"]
        )

        sns.set_theme(style="white")

        sns.lineplot(
            x='h',
            y="Total énergie soutirée normalisée (Wh)",
            hue='day_n',
            data=df_conso_temp,
            ax=axs[row, col]
        )

        axs[row, col].set_title(f"Profil : {p}")
        axs[row, col].legend(title="Jour", loc="upper right", fontsize='small')

        col += 1
        if col == 4:
            col = 0
            row += 1

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Intégration dans Streamlit
    st.pyplot(fig)

def spectral_analysis_streamlit(serie):
    st.markdown("#### 🎵 Spectrogramme de la série sélectionnée")
   
    spectrogram_analyzer = SpectrogramAnalysis(window='hann', nperseg=10*48, noverlap=2*48, fs= 1/1800, threshold=0.5)
    spectrogram_analyzer.fit(serie['Total énergie soutirée (Wh)'].dropna())
    TT = spectrogram_analyzer.transform(serie['Total énergie soutirée (Wh)'].dropna())
    
    fig = plt.figure(figsize=(10, 4))
    spectrogram_analyzer.plot_spectrogramme(fig=fig)
    
    st.pyplot(fig)
    
def acf_pacf_streamlit(serie):
    st.markdown("#### 🎵 ACF et PACF de la série sélectionnée")
   
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,6))

    plot_acf(serie , lags = 2*48, ax=ax1)

    ax1.set_title("Fonction d'autocorrélation")

    plot_pacf(serie , lags = 2*48, ax=ax2)
    ax2.set_title("Fonction d'autocorrélation partielle")
    
    st.pyplot(fig)

def decomposition_streamlit(serie):
    # Analyse spectrale
    spectrogram_analyzer = SpectrogramAnalysis(window='hann', nperseg=10*48, noverlap=2*48, fs= 1/1800, threshold=0.5)
    spectrogram_analyzer.fit(serie['Total énergie soutirée (Wh)'].dropna())
    TT = spectrogram_analyzer.transform(serie['Total énergie soutirée (Wh)'].dropna())
    P = int(TT.iloc[-1,-1])
    #serie_diff = serie.diff().dropna()
    
    decomposition = seasonal_decompose(serie, period=P,model='multiplicative',  extrapolate_trend='freq')
    serie_T = decomposition.trend
    serie_S = decomposition.seasonal
    serie_R = decomposition.resid
    # Tracé manuel avec couleurs
    fig, axs = plt.subplots(4, 1, figsize=(8, 6), sharex=True)
    axs[0].plot(serie, color='black')
    axs[0].set_ylabel('Observé')

    axs[1].plot(serie_T, color='blue')
    axs[1].set_ylabel('Tendance',fontsize=10)

    label = f"Période = {P}"
    axs[2].plot(serie_S, label=label, color='green')
    axs[2].set_ylabel('Saisonnalité',fontsize=10)

    axs[3].plot(serie_R, color='red')
    axs[3].set_ylabel('Résidu',fontsize=10)
    axs[3].set_xlabel("Temps", fontsize=10)
    axs[3].tick_params(axis='x', labelrotation=45) 
    
    
    for ax in axs:
        ax.legend(loc='upper right')
        ax.grid(True)

    fig.suptitle("Décomposition multiplicative de la  série sélectionnée", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Pour ne pas écraser le titre
    st.pyplot(fig)
def plot_correlation_tendances(serie_conso, serie_temp, var : str):
    """
    Affiche la corrélation visuelle entre la tendance de la consommation
    et la tendance inversée de la température, normalisées.
    
    - serie_conso : série pandas (consommation en Wh)
    - serie_temp : série pandas (température en °C)
    """
    scaler = MinMaxScaler()

    # Moyenne mobile sur 7 jours (48 pas/jour)
    moyenne_mobile = serie_conso.rolling(window=48*7).mean()
    

    if var == 'U':
        # Masque pour les valeurs entre 80 et 95
        moyenne_mobile_T = 1/serie_temp.rolling(window=48*7).mean()
        #moyenne_mobile_T = 1/ (moyenne_mobile_T + np.max(moyenne_mobile_T))
        #mask_mid = (moyenne_mobile_T > 80) & (moyenne_mobile_T < 95)
        #moyenne_mobile_T[mask_mid] = 1 / (1 + moyenne_mobile_T[mask_mid])
            
    else : 
        moyenne_mobile_T = 1 / (serie_temp.rolling(window=48*7).mean() + 1 - np.min(serie_temp))

    moyenne_mobile_normalise = scaler.fit_transform(moyenne_mobile.values.reshape(-1,1))
    moyenne_mobile_T_normalisee = scaler.fit_transform(moyenne_mobile_T.values.reshape(-1,1))
    
    
    # Corrélation de Pearson
    # Convertir en array et retirer les NaN
    x = np.array(moyenne_mobile_normalise)
    y = np.array(moyenne_mobile_T_normalisee)

    # Supprimer les valeurs où au moins un est NaN
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean, y_clean = x[mask], y[mask]

    # Corrélation propre
    coeff = np.corrcoef(x_clean, y_clean)[0, 1]
        
    #fig = plt.figure(figsize=(12, 6))
    # Tracé
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(moyenne_mobile_normalise, 
             label='Tendance normalisée de la consommation', linestyle='--', color='black')
    ax.plot(moyenne_mobile_T_normalisee, 
             label='Tendance transformée et normalisée de la variable météo cible', linestyle='--', color='red')

    
    
    ax.set_title("Corrélation des tendances")
    ax.set_xlabel("Temps – pas = 30 minutes")
    ax.set_ylabel("Valeur (échelle normalisée)")
    ax.legend()
    ax.grid(True)
    
    # Affichage du coefficient en haut à gauche
    ax.text(0.01, 0.95, f"r = {coeff:.2f}", transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))
    
    st.pyplot(fig)

def plot_correlation_residu(serie_conso, serie_meteo, var : str):
    """
    Affiche la corrélation visuelle entre la tendance de la consommation
    et la tendance inversée de la variable météo cible, normalisées,
    et affiche le coefficient de corrélation de Pearson.
    
    - serie_conso : série pandas (consommation en Wh)
    - serie_temp : série pandas (température, humidité, etc.)
    """
    # Analyse spectrale
    spectrogram_analyzer = SpectrogramAnalysis(window='hann', nperseg=10*48, noverlap=2*48, fs= 1/1800, threshold=0.5)
    spectrogram_analyzer.fit(serie['Total énergie soutirée (Wh)'].dropna())
    TT = spectrogram_analyzer.transform(serie['Total énergie soutirée (Wh)'].dropna())
    P = int(TT.iloc[0,0])
    decomposition = seasonal_decompose(serie_conso, period=P,model='multiplicative',  extrapolate_trend='freq')
    serie_conso_resid = decomposition.resid
    
    serie_meteo = serie_meteo + 1 - np.min(serie_meteo)
    decomposition = seasonal_decompose(serie_meteo, period=P, model='multiplicative',  extrapolate_trend='freq')
    serie_meteo_resid = decomposition.resid
    
    
    scaler = MinMaxScaler()



    # Aligner les deux séries (évite NaN)
  

    # Normalisation
    conso_norm = scaler.fit_transform(serie_conso_resid.values.reshape(-1,1))
    meteo_norm = scaler.fit_transform(serie_meteo_resid.values.reshape(-1,1))

    # Corrélation de Pearson
    coeff = np.corrcoef(conso_norm.ravel(), meteo_norm.ravel())[0, 1]

    # Tracé
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(conso_norm, label='Tendance normalisée de la consommation', linestyle='--', color='black')
    ax.plot(meteo_norm, label='Tendance inversée et normalisée de la variable météo cible', linestyle='--', color='red')

    ax.set_title("Corrélation des résidus")
    ax.set_xlabel("Temps – pas = 30 minutes")
    ax.set_ylabel("Valeur (échelle normalisée)")
    ax.legend()
    ax.grid(True)

    # Affichage du coefficient en haut à gauche
    ax.text(0.01, 0.95, f"r = {coeff:.2f}", transform=ax.transAxes,
            fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))

    st.pyplot(fig)


def run_demo_prediction(df_selection):

    # Sélection des paramètres
    selected_region = df_selection["Région"].unique()[0]
    selected_profil = df_selection["Profil"].unique()[0]
    selected_puissance = df_selection["Plage de puissance souscrite"].unique()[0]
    st.write(selected_profil,selected_puissance,selected_region)

    if st.button("🚀 Lancer la prédiction !"):
        try:
            resultats = traiter_profile_puissance(df_selection, selected_profil, selected_puissance, selected_region)
            #st.write("Résultats bruts :", resultats)

            if resultats is not None and not resultats.empty:
                st.success("✅ Prédiction réussie !")

                # Résultats par composante
                st.markdown("## 🔍 Prédictions par composante")
                composants = resultats["Composante"].unique()
                for composante in composants:
                    st.markdown(f"#### 📈 Composante : {composante}")
                    df_comp = resultats[resultats["Composante"] == composante]
                    # Bloc de métriques
                    with st.container():
                        col1, col2, col3 = st.columns(3)
                        col1.metric("📊 MAPE (%)", f"{df_comp['MAPE (%)'].values[0]:.3f}")
                        col2.metric("📉 MAE (Wh)", f"{df_comp['MAE (Wh)'].values[0]:,.2f}")
                        col3.metric("📈 RMSE (Wh)", f"{df_comp['RMSE (Wh)'].values[0]:,.2f}")

                    st.markdown("---")

                # Affichage tableau complet
                st.markdown("## 📋 Résumé des scores")
                st.dataframe(resultats)

            else:
                st.warning("Aucun résultat retourné.")

        except Exception as e:
            st.error(f"❌ Une erreur est survenue : {e}")



#--------------------------------------------------------
#
#
#--------------------------------------------------------
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

import decomposition_serie_temporelle
import analyse_spectrale 
import pipeline_analyse_spectrale_decomposition 
import analyse_et_sarima
import sarimax_model
import sarimax_model_fitted
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
importlib.reload(sarimax_model_fitted)
importlib.reload(analyse_et_sarima)
importlib.reload(sequence_transformer)
importlib.reload(inversion_transformer)
importlib.reload(normalisation_transformation)
importlib.reload(wrapper_pour_lstm)
importlib.reload(realigner)

from decomposition_serie_temporelle import DecompositionSerieTemporelle
from analyse_spectrale import SpectrogramAnalysis
from sarimax_model import SARIMAModel
from sarimax_model_fitted import SARIMAModelFitted

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

# dossier contenant les modèles 
folder_all_models = r"D:\MesDocuments\Formation\DataScientist_PSL\Projet\BD\models" # repertoire de la base de données
folder_models= os.path.join(folder_all_models, "ARA") # 


# # ------------------------------------------------------------------------------------------------------------------------------------------------
# # Constantes et Variables globales servant de paramètres par défaut pour les constructeurs 
# # ------------------------------------------------------------------------------------------------------------------------------------------------
    # st.markdown("### ⏳ Horizon de prédiction")
    # NOMBRE_JOUR_PREDICTION = st.slider(
    # "Sélectionnez l'horizon de la prédiction en jours =  48 pas",
    # min_value = 1,  # 1 jour
    # max_value = 60, # 1 semaine
    # value = 7,     # valeur par défaut
    # step = 7
     # )

# TARGET  = 'Total énergie soutirée (Wh)'


# COLUMNS_A_DECOMPOSER = list(['Total énergie soutirée (Wh)', 'T_moyenne', 'U_moyenne', 'Rayonnement solaire global (W/m2)']) 
# FOMRES_DECOMPOSITION =  list(["multiplicative","multiplicative","multiplicative", "multiplicative"])
# COLUMNS_TENDANCE = list(['Total énergie soutirée (Wh)_tendance','T_moyenne_tendance', 'U_moyenne_tendance', 'Rayonnement solaire global (W/m2)_tendance'])
# COLUMNS_RESIDUEL= list(['Total énergie soutirée (Wh)_residuel','T_moyenne_residuel', 'U_moyenne_residuel', 'Rayonnement solaire global (W/m2)_residuel']) 
# COLUMNS_TO_INVERSE_TENDANCE = list(['T_moyenne_tendance', 'U_moyenne_tendance', 'Rayonnement solaire global (W/m2)_tendance'])
# COLUMNS_TO_INVERSE_RESIDUEL = list(['T_moyenne_residuel', 'U_moyenne_residuel', 'Rayonnement solaire global (W/m2)_residuel'])
# FS = 1/1800 # fréquence d'échantillonnage pas  = 30 minutes
# NOMBRE_POINTS_PAR_SEGMENT = 30*48  # pour l'analyse spectrale une semaine
# OVERLAP = 12 # Recouvrement entre fenêtres
# WINDOWS_SIZE = 24 # Taile de fenêtre pour LSTM
# NB_PAS_JOUR = 48 # pas  = 30 minutes ==> 48 par jour
# NOMBRE_JOUR_PREDICTION = 15 #a évaluer
# NOMBRE_JOUR_TRAIN = 365 # une année pour le train
# NOMBRE_JOUR_TOTAL = NOMBRE_JOUR_TRAIN + NOMBRE_JOUR_PREDICTION
# TEST_PROPORTION = NOMBRE_JOUR_PREDICTION / NOMBRE_JOUR_TOTAL
# TRAIN_SIZE = NB_PAS_JOUR*NOMBRE_JOUR_TRAIN
# TOTAL_SIZE = NB_PAS_JOUR*NOMBRE_JOUR_TOTAL
# TOTAL_SIZE_SARIMAX = 90 * NB_PAS_JOUR # trois mois pour entrainer le sarimax ce qui permet d'éviter l'explosion de la mémoir pour le filtre de Kalman
# THRESHOLD = 0.3 # Seuli de détection pour le spectre
# NB_COMPOSANTES_SPECTRALES = 2 # Le nombre de composante spectrale à garder


# # Paramètres de la décomposition spectrale
# spectro_params_default ={"fs": FS,                                     # Fréquence d'échantillonnage (1pas = 30 minutes :  Ts = 1800s, Fs = 1/1800 Hz)
                         # "window": "hann",                             # Fenêtre de Hann 
                         # "nperseg": NOMBRE_POINTS_PAR_SEGMENT,         # Longueur de la fenêtre d'analyse spectrale
                         # "noverlap": OVERLAP,                          # Recouvrement entre fenêtres
                         # "threshold": THRESHOLD                           # Seuil élevé pour ne détecter qu'une période dominante
                        # } 
                        
# # Paramètres pour LSTM : prédiction de la composante tendance
# lstm_params_tendance =  { 'window_size' : WINDOWS_SIZE, 
                         # 'n_neurons': 256,  
                         # 'factor' : 0.1,
                         # 'patience' : 30,
                         # 'epochs' : 100, 
                         # 'batch_size' : 32, 
                         # 'loss' : "mean_absolute_percentage_error",#"mean_absolute_error",
                         # 'min_delta' : 5e-3,
                         # 'nbfoldcv' : 5, 
                         # 'optimize_architecture':True, # Pour chercher l'architecture optimale
                         # 'optimize_lr':True,           # Pour chehercher le taux optimal
                         # 'use_grid_search':False,      # Utilsation du grid_search pour les hyperparamètres
                         # 'save_path' : None,           # 'best_lstm_model_tendance.keras'
                         # 'activation': 'relu'  
                        # }
# # Paramètres pour LSTM : prédiction de la composante résisiduelle                        
# lstm_params_residuel =  { 'window_size' : WINDOWS_SIZE, 
                         # 'n_neurons': 256,  
                         # 'factor' : 0.1,
                         # 'patience' : 30,
                         # 'epochs' : 100, 
                         # 'batch_size' : 32, 
                         # 'loss' : "mean_absolute_error", #"mean_absolute_percentage_error",
                         # 'min_delta' : 5e-3,
                         # 'nbfoldcv' : 5, 
                         # 'optimize_architecture':True,  # Pour chercher l'architecture optimale
                         # 'optimize_lr':True,            # Pour chehercher le taux optimal
                         # 'use_grid_search':False,       # Utilsation du grid_search pour les hyperparamètres
                         # 'save_path' : None,            # 'best_lstm_model_tendance.keras' 
                         # 'activation': 'relu' #'tanh'
                        # }


# COLUMNS = ["T_moyenne", 
               # "U_moyenne",
               # "FF_moyenne",
               # "Rayonnement solaire global (W/m2)", 
               # "Nb points soutirage" ,
               # "Total énergie soutirée (Wh)"
               # ]


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

def premiere_analyse(df,spectro_params = None, column_target = None):
        column_target = column_target 
        spectro_params = spectro_params 
        spectrogram_analyzer = SpectrogramAnalysis(**spectro_params) 
        y = df[column_target]
        spectrogram_analyzer.fit(y)
        periodes = spectrogram_analyzer.transform(y) 
        return periodes.values.flatten().astype(int)[0:NB_COMPOSANTES_SPECTRALES]
        #return spectrogram_analyzer.dominant_periodes.astype(int).tolist()[0:2] # les deux premières


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
     

 
        

        if df_profile_puissance.empty:
            st.write(f"Données vides pour profil={profile}, puissance={puissance}")
            return None
            
        #st.write(f"Données non vides pour profil={profile}, puissance={puissance}, region {reg}")
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
            #print(sarima_params_composantes)  
            estimateur_composantes_periodiques[periode] = SARIMAModelFitted(**sarima_params_composantes)
            estimateur_composantes_periodiques[periode].fit(X_train[f"{TARGET}_saisonnalite_{int(periode)}"].iloc[-TOTAL_SIZE_SARIMAX:].asfreq('30T')) # on se limte à trois mois pour ne pas saturer la memoire
            X_prediction_composant[periode] = estimateur_composantes_periodiques[periode].predict(X_test[f"{TARGET}_saisonnalite_{int(periode)}"]) 
            #print('fin prédiction saisonnalite : ', periode, '...')

            # Nettoyage de la mémoire
            K.clear_session()
            gc.collect()
            
            #del estimateur_composantes_periodiques[periode]
            mape_s = mean_absolute_percentage_error(X_test[f"{TARGET}_saisonnalite_{int(periode)}"], X_prediction_composant[periode])
            mae_s = mean_absolute_error(X_test[f"{TARGET}_saisonnalite_{int(periode)}"], X_prediction_composant[periode])
            rmse_s =np.sqrt(mean_squared_error(X_test[f"{TARGET}_saisonnalite_{int(periode)}"], X_prediction_composant[periode]))
            # st.write(f"saisonnalite_{int(periode)}:")
            # st.write(f"MAPE: {mape_s:.2%}")
            # st.write(f"MAE: {mae_s:.2}")
            # st.write(f"RMSE: {rmse_s:.2}")

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
        #st.write('fin prédiction résiduel ...')
        mape_s = mean_absolute_percentage_error(X_test[f"{TARGET}_residuel"], X_prediction_composant['residuel'])
        mae_s = mean_absolute_error(X_test[f"{TARGET}_residuel"], X_prediction_composant['residuel'])
        rmse_s =np.sqrt(mean_squared_error(X_test[f"{TARGET}_residuel"], X_prediction_composant['residuel']))
        # st.write("residuel")
        # st.write(f"MAPE: {mape_s:.2%}")
        # st.write(f"MAE: {mae_s:.2}")
        # st.write(f"RMSE: {rmse_s:.2}")
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
        #st.write('fin prédiction tendance ...')
        mape_s = mean_absolute_percentage_error(X_test[f"{TARGET}_tendance"], X_prediction_composant['tendance'])
        mae_s = mean_absolute_error(X_test[f"{TARGET}_tendance"], X_prediction_composant['tendance'])
        rmse_s =np.sqrt(mean_squared_error(X_test[f"{TARGET}_tendance"], X_prediction_composant['tendance']))
        # st.write("tendance")
        # st.write(f"MAPE: {mape_s:.2%}")
        # st.write(f"MAE: {mae_s:.2}")
        # st.write(f"RMSE: {rmse_s:.2}")
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
        # st.write(f"MAPE: {mape:.2%}")
        # st.write(f"MAE: {mae:.2}")
        # st.write(f"RMSE: {rmse:.2}")
          

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
        #st.write("✅ Résultat prêt à retourner")
        #st.write("Résultat final : ", df_result)
        #st.markdown("#📈 Série réelle vs prédite")
        # df_compare = pd.DataFrame(y_test, columns=['y_test (Wh)'])
        # df_compare['y_prediction (Wh)'] = y_prediction
        # st.line_chart(df_compare)
        
        
        
        # Préparer le DataFrame pour Altair (long format)
        df_compare = pd.DataFrame({
            'Datetime': y_test.index,
            'Série réelle (Wh)': y_test.values,
            'Série prédite (Wh)': y_prediction.values
        })
        df_long = df_compare.melt(id_vars='Datetime', var_name='Série', value_name='Valeur (Wh)')

        # Créer le graphique Altair
        chart = alt.Chart(df_long).mark_line().encode(
            x='Datetime:T',
            y='Valeur (Wh):Q',
            color=alt.Color('Série:N', scale=alt.Scale(domain=['Série réelle (Wh)', 'Série prédite (Wh)'],
                                                       range=['#1f77b4', '#d62728']))  # Bleu et rouge
        ).properties(
            width=700,
            height=400,
            title='📈 Série réelle vs prédite'
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        ).configure_title(
            fontSize=16,
            anchor='start'
        )

        st.altair_chart(chart, use_container_width=True)
        return df_result  
    except Exception as e:
        st.write(f"Erreur rencontrée pour {reg} - {profile} - {puissance}: {str(e)}")
        return None
       
       


def afficher_resultats_globaux(chemin_dossier_csv):
    
    folder_label = list(chemin_dossier_csv.keys())[0]
    folder_path = chemin_dossier_csv[folder_label]

    if not folder_path.exists():
        st.error(f"❌ Le dossier `{folder_path}` n’existe pas.")
        return None

    csv_files = list_csv_files(folder_path)
    if not csv_files:
        st.warning("⚠️ Aucun fichier CSV trouvé dans ce dossier.")
        return None

    selected_file = st.selectbox("📄 Choisissez un fichier CSV :", csv_files)
    df = pd.read_csv(selected_file)
    # Chargement des fichiers CSV du dossier
 

    metriques = ["MAPE (%)"] # "MAE (Wh)", "RMSE (Wh)", "temps execution"]
    for metrique in metriques:
        st.markdown(f"{metrique}")

        # Initialiser le graphe
        fig, ax = plt.subplots(figsize=(8, 4))
        palette = sns.color_palette("husl", n_colors=df["Profil"].nunique())
        sns.set_theme(style="white")

        sns.scatterplot(
            x='Profil',
            y=metrique,
            hue='Puissance',
            data=df,
            palette=palette,
            ax=ax
        )
        if metrique == "MAPE (%)":
            ax.axhline(y=2, color='green', linestyle='--', linewidth=1, label="Seuil 2%")
            ax.axhline(y=4, color='blue', linestyle='--', linewidth=1, label="Seuil 4%")
            ax.axhline(y=8, color='orange', linestyle='--', linewidth=1, label="Seuil 8%")
            ax.axhline(y=20, color='red', linestyle='--', linewidth=1, label="Seuil 20%")
        ax.legend(fontsize=6)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=6)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)
        ax.set_title(f"{metrique} pour les configurations profil - puissance", fontsize=6)
        ax.set_xlabel("Puissance", fontsize=6)
        ax.set_ylabel(metrique, fontsize=6)
        ax.legend(title="Puissance", fontsize=6, title_fontsize=6)
        st.pyplot(fig)


# -----------------------------
# Sidebar navigation
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", [
    "Acceuil",
    "Contexte et problématique",
    "Données utilisées",
    "Fusion des données",
    "Exploration de la base construite",
    "Représentation du problème",
    "Analyse des séries temporelles",
    "Analyse des corrélations",
    "Approche proposée",
    "Réalisation – Implémentation",
    "Démonstration",
    "Résultats",
    "Conclusion"
])

# -----------------------------
# Préambule
# -----------------------------
if page == "Acceuil":
    set_full_width()
    show_header()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        - Projet proposé par **Guillaume ROTH**  
        - Équipe initiale : **Fei YANG**, **Guillaume ROTH**, **Youssef SERRESTOU**
        - Objectif initial :  
          _Détection d'anomalie de la consommation d'électricité à l’échelle d’un habitat individuel_
        """)

        st.markdown("""
        ### Changements importants
        - ❌ Absence de données à l’échelle d’un habitat  
        - 🚶‍♀️ Départ de **Fei YANG** pour un autre projet  
        - ❌ Désengagement progressif de **Guillaume ROTH**
        """)

    with col2:
        st.markdown("""
        ### 🎯 Objectif du projet
        Suite à :  
        - 🤝 des échanges avec **Enedis**  
        - 📚 une **étude de l’état de l’art**  
        - 👨‍🏫 et en concertation avec le **tuteur du projet**

        ---
        ➔
         _Établir un modèle de prévision de la consommation d’électricité à court terme pour les utilisateurs du réseau Enedis en France._
        """)


# -----------------------------
# 1. Contexte et problématique
# -----------------------------
if page == "Contexte et problématique":
    set_full_width()
    show_header()
    
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(""" 
        **Contexte :**  
        Deux catégories de prévisions à distinguer :
        -  **la prévision de la charge électrique** :
            - importante pour les fournisseurs et opérateurs d’électricité  
            - sert à équilibrer l’offre et la demande 
            - largement étudiée
        -  **la prévision de la consommation d'électricité** :
            - importante pour les consommateurs et pour les producteurs
            - sert à prédire la consommation réelle 
            - moins abordée dans les études publiées
        
        **Typologie selon  l'horizon**:
        - Prévision à court terme < 1 semaine
        - Prévision à moyen terme > 1 semaine et < 1 an
        - Prévision à long terme > 1 an
        """)
    
    with col2:
        st.markdown("""     
        **Notre objectif:**  
        Construire un modèle fiable, robuste et précis pour prévoir à court terme la consommation des utilisateurs du réseau Enedis, 
        en fonction des variables explicatives.
        
        
        **Les étapes de réalisation du projet:**  
        - Recherche des données de consommation d'électricité à utiliser ➔ base de données Enedis , data.gouv,  échanges avec Enedis
        - Détermination des facteurs influants sur la consommation ➔  état de l'art
        - Recherche des bases de données pour inclure ces variables ➔ data.gouv, Météo-France
        - Analyse, traitement et fusioner des différentes base de données 
        - Fromalisation et modélisation du problème
        - Proposition d'une nouvelle approche
        - Evaluation de l'approche proposée
        - Présentation écrite et orale
       
         """)


# -----------------------------
# 2. Données utilisées
# -----------------------------
elif page == "Données utilisées":
    set_full_width()
    show_header()
    st.title("📈 Données utilisées")
        
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(""" 
        1. **Données Enedis**:
        
        **Caractéristiques :**
        - Données restituant l'énergie totale soutirée au pas de 30 minutes 
        - Plage de puissance ≤ 36 kVA
        - Agrégées par profil, plage de puissance souscrite et région
        - Période choisie : 2023-2024
            
        2. **Données de Météo-Franceo-France**
            - Température (°C), 
            - Humidité (%), 
            - Vitesse du vent (m/s), 
            

    """)
    with col2:
        st.markdown(""" 
        **Caractéristiques :**
        - Granularité temporelle (échantillonnage temporel) : horaire (1 heure) 
        - Données de toutes les stations regroupés par département, et par lots de période 
        - Période choisie : 2023-2024
        3. **Données de Météo-Franceo-France**
            - nébulosité remplacée par le rayonnement global (W/m2)
        
        **Caractéristiques :**
        - Echantillonnage temporel : tri-horaire 
        - Données régionnales 
        - Période choisie : 2023-2024
      
        4. **Données calendaires (jours spéciaux)**
    """)    
    
    #st.image("figures/schema_donnees.png", caption="Schéma des données fusionnées (exemple)")

    st.markdown("""**Visualisation rapide des CSV par répertoire**""")

    folder_label = st.selectbox("📂 Choisissez un répertoire :", list(FOLDERS.keys()))
    folder_path = FOLDERS[folder_label]
    sep = SEPARATORS.get(folder_label, ';')

    if not folder_path.exists():
        st.error(f"Le dossier `{folder_path}` n’existe pas.")
      

    csv_files = list_csv_files(folder_path)
    if not csv_files:
        st.warning("Aucun fichier CSV trouvé dans ce dossier.")
        

    selected_file = st.selectbox("📄 Choisissez un fichier CSV :", csv_files)
    if selected_file:
        show_file_info(selected_file,sep)

# -----------------------------
# 3. Fusion des données 
# -----------------------------
elif page == "Fusion des données":
    set_full_width()
    show_header()
    st.title("📈 Fusion des données ")
        
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(""" 
        1. **Probème d'échantillonnage temporel**
            - Période d'échantillonnage choisie : 30 minutes
            - sur-échantillonnage par interpolation linéaire des données météorologiques 
            

        """)
        st.markdown("""**Visualisation rapide des CSV par répertoire**""")
        #st.title("📁 Visualisation rapide des CSV par répertoire")
        folder_label = st.selectbox("📂 Choisissez un répertoire :", list(FOLDERS_Fusion.keys()))
        folder_path = FOLDERS_Fusion[folder_label]
        sep = ','
    


        if not folder_path.exists():
            st.error(f"Le dossier `{folder_path}` n’existe pas.")
      
        csv_files = list_csv_files(folder_path)
        if not csv_files:
            st.warning("Aucun fichier CSV trouvé dans ce dossier.")
            
        selected_file = st.selectbox("📄 Choisissez un fichier CSV :", csv_files)
    
    with col2:
        st.markdown(""" 
        2. **Probème d'échantillonnage spatiale** :
        
        - Hypothèse : ergodicité spatiale
        - Analyse des corrélations 
        
        ➔ Pour chaque région on remplace les données de toutes ses stations météo par leurs statistiques:
        
            - moyenne
            - extremums
            - écart-type
            - quartils q1, q2, q3
            - coefficient d’asymétrie (skew)
            - coefficient d’aplatissement (kurtosis)
            
    """)    
    
    #st.image("figures/schema_donnees.png", caption="Schéma des données fusionnées (exemple)")

    if selected_file:
        show_file_info(selected_file, sep = None)
# -----------------------------
# 4. Exploration de la base construite
# -----------------------------
elif page == "Exploration de la base construite":
    set_full_width()
    show_header()
    st.title("🔎 Première analyse exploratoire")

    st.markdown("""
    Cette section illustre l'effet de plusieurs facteurs sur la consommation électrique:
    
        - L'affichage ici est statique pour éviter le problème de temps de génération des figures 
        - Les données  utilisées pour générer les figures sont issues de la région Auvergne-Rhône-Alpes
        - la consommation a été divisée par le nombre de points de soutirage puis normalisée       
    
    """)

    FACTEURS = {
        "🕒 Saisonnalité intra-journalière": "Chap2/conso_par_heure.png",
        "📅 Saisonnalité annuelle": "Chap2/conso_annuelle.png",
        "🧍 Influence du profil": "Chap2/conso_horarire_par_profile.png",
        "⚡ Influence de la puissance souscrite": "Chap2/conso_heure_par_plage.png",
        "🌤️ Influence des facteurs météorologiques": "Chap2/conso_vs_facteurs_meteo.png",
        #"📆 Influence des jours de semaine/week-end": "Chap2/effetjour.png"
    }

    # Convertir les éléments en liste pour itération par 2
    items = list(FACTEURS.items())

    # Affichage en deux colonnes
    cols = st.columns(2)  # Crée deux colonnes

    for i in range(len(items)-1):
        titre, img = items[i]
        col = cols[i % 2]  # Alterne entre la colonne de gauche (0) et celle de droite (1)
        with col:
            with st.expander(titre):
                st.image(img, use_column_width=True)
    
    titre, img = items[-1]
    with st.expander(titre):
                st.image(img, use_column_width=True)
    # for i in range(0, len(items)):
        # titre, img = items[i]
        # with st.expander(titre):
                # st.image(img, use_column_width=True)
    

# -----------------------------
# 5. Représentation du problème
# -----------------------------
elif page == "Représentation du problème":
    set_full_width()
    show_header()
    st.title("Représentation du problème")
    
    st.markdown("### Représentation")
    # st.markdown("""
        # Pour toute configuration `(q = (profil, palge de puissance sosucrite), r=région)` :
    st.latex(r"""\text{Pour toute configuration } q  = \text{ (profil - plage de puissance) dans une région } r """)
    col1, col2 = st.columns(2)
    with col1:

        st.markdown("#### La variable cible est ")

        st.markdown("- La série temporelle qui représente **consommation d’électricité moyenne (en Wh)**  par point de soutirage")
        st.latex(r"""\left(\overline{Y}_{t}^{(r,q)}\right)_{t \in \mathbb{T}} = \left(\frac{Y_{t}^{(r,q)}}{N_{t}^{(r,q)}}\right)_{t \in \mathbb{T}}""")

        
        st.markdown("-  La **consommation d’électricité (en Wh)** est une série temporelle de.")
        st.latex(r"""\left(Y_{t}^{(r,q)}\right)_{t \in \mathbb{T}}""")
        
        st.markdown("- Série temporelle représentant le **nombre de points de soutirage**")
        st.latex(r"""\left(N_{t}^{(r,q)}\right)_{t \in \mathbb{T}}""")
        
    with col2:
        st.markdown("#### Les variables exogènes sont les facteurs météorologiques")
        st.markdown(" - **Température moyenne (°C)** dans la région \\(r\\).")
        st.latex(r"""\left(T_{t}^{(r)}\right)_{t \in \mathbb{T}}""")
        
        st.markdown(" - **Humidité moyenne (%)** dans la région \\(r\\).")
        st.latex(r"""\left(U_{t}^{(r)}\right)_{t \in \mathbb{T}}""")
        
        st.markdown(" - **Rayonnement solaire global (W/m2)** dans la région \\(r\\).")
        st.latex(r"""\left(R_{t}^{(r)}\right)_{t \in \mathbb{T}}""")
        
        
          
    
        
    st.markdown("L’ensemble des **instants d’observation** disponibles dans notre base qui couvre la période du **01/01/2023** au **31/12/2024** est : ")
    st.latex(r"""\mathbb{T} = \left\{ t = kT_s,\; k \in \left\{ 0,\ldots,L \right\}, T_s = 1800 s, L = 35088 \right\}""") 
    with st.container():
        st.markdown("#### Formalisation du problème")
        st.latex(r"""
       
                \text{ Nous cherchons un modèle } \mathcal{M}^{(q,r)} 
                \text{ qui permet d’estimer les valeurs futures } 
                \left( Y_k \right)_{\tau \leq t \leq \tau + h} 
                \text{ pour un horizon } h, \\
                \text{ en fonction de l’ensemble d’informations disponible sur les valeurs passées de la série cible et les valeurs des variables exogènes}
                """)

        st.latex(r"""
        \left(\widehat{Y}_{\tau+1},\ \widehat{Y}_{\tau+2},\ ..., \ \widehat{Y}_{\tau+h}\right) 
        = \mathcal{M}^{(q,r)}\left( 
        \left(Y_t^{(r,q)}\right)_{t \leq \tau },\ 
        \left(T_t^{(r)}\right)_{t \leq \tau + h},\ 
        \left(U_t^{(r)}\right)_{t \leq \tau + h  },\ 
        \left(R_t^{(r)}\right)_{t \leq \tau + h } 
         \right)
        """)
       
    
    
     

  
# -----------------------------
# 6. Analyse des séries temporelles
# -----------------------------
elif page == "Analyse des séries temporelles":
    set_full_width()
    show_header()
    st.title("🔎 Analyse temporelle et spectrale")

    st.markdown("Cette section expose les  propriétés étudiées de nos séries temporelles avant d'établir la modélisation proposée.")

    ANALYSES = {
    
       "📉 **Tests de stationnarité (ADF, KPSS)**": {
            "commentaire": """
            Les tests de stationnarité permet de vérfier si les propriétés statistiques de nos séries sont constantes dans le temps afin:
            
                - d'adapter une méthode d'analyse spectrale appropriée
                - et paramètrer correctement les modèles  
                
     
            **Exemple**:
            
            - **ADF (Augmented Dickey-Fuller)** : H0 = non stationnaire (présence de racine unitaire) 
            - **KPSS** : H0 = stationnaire  
            Une p-value < 0.05 permet de rejeter l’hypothèse nulle.
            """,
            "fonction": "Tests de stationnarité"  
        },
        
        "🎵 **Analyse spectrale**": {
            
            "commentaire": """
            L’analyse spectrale a été utilisé pour  
            - mettre en évidence les **périodes dominantes** dans la série. 
            - extraire les **composantes saisonnières** en utilisant ces périodes.
            """,
            "fonction": "spectrogramme"
        },
        "📈 **Analyse ACF / PACF**": {
            "fonction": "ACF / PACF",  
            "commentaire": """
            Les fonctions ACF (auto-corrélation) et PACF (auto-corrélation partielle) aident à identifier l’ordre des modèles AR et MA.  
            - ACF montre les corrélations à différents retards  
            - PACF montre les corrélations après retrait des effets intermédiaires
            """
        },
        "🔍 **Décomposition des séries temporelles**": {
            "fonction": "Décomposition",
            "commentaire": """
            La série est décomposée en  :  
            - **Tendance**
            - **Saisonnalités**
            - **Résidu**  
            Cela permet de mieux modéliser chaque composante séparément.
            """
        },
        "📌 **Conclusions de cette analyse**": {
            "commentaire": """
       
            - les séries temporelles de la consommation d'électricité sont des séries multi-saisonnières, 
            - l'analyse temps-fréquence permet de déterminer les saisonnalités dominantes de celles-ci, 
            - les séries temporelles de la consommation d'électricité sont non stationnaires et ceci est dû à leurs tendances, 
            - les composantes saisonnières et résiduelles sont stationnaires,
            
            \u2794  **Nouvelle approche basée sur** 
            - une décomposition en cascade utilisant les périodes détectées par analyse spectrale
            - une analyse de la **corrélation entre les composantes de la consommation** et les **variables météorologiques** 
            - un modèle de prévision adapté pour chaque composante 
           
            """
        }
    }
    # Préparation des données : filtrer + normalisation + décomposition
    df_fusion_filtred= load_and_filter_df_fusion(FOLDERS_Fusion)
    df_fusion_filtred = force_datetime_index(df_fusion_filtred)
    df_fusion_filtred = imputer_series(df_fusion_filtred, method='ffill', window=3) 
    df_fusion_filtred["Total énergie soutirée (Wh)"] = df_fusion_filtred["Total énergie soutirée (Wh)"] / df_fusion_filtred["Nb points soutirage"]
    
    #start = pd.Timestamp("2023-01-01 00:00")
    #end = pd.Timestamp("2023-12-31 00:00")
    serie = df_fusion_filtred[['Total énergie soutirée (Wh)']]#.loc[start:end]

    
    if df_fusion_filtred is not None:
        st.dataframe(df_fusion_filtred.head())
        
    for titre, bloc in ANALYSES.items():
        with st.expander(titre):
            if bloc.get("fonction") == "Tests de stationnarité"  :
                st.markdown(bloc["commentaire"])
                
                col1, col2 = st.columns(2)
                with col1:
                    test_type = st.selectbox("🔍 Choix du test", ["ADF", "KPSS"])
                with col2:
                    col_name = st.selectbox("📈 Choisir une variable", ["Total énergie soutirée (Wh)", "T_moyenne","U_moyenne", "Rayonnement solaire global (W/m2)"])

                if st.button("🧪 Lancer le test"):
                    stationnarity_test(df_fusion_filtred[col_name], test_type)
            
            # Spectrogramme
            elif bloc.get("fonction") == "spectrogramme":
                st.markdown(bloc["commentaire"])
              
                if st.button("🎵 Lancer l’analyse spectrale"):
                    spectral_analysis_streamlit(serie)
            # ACF / PACF
            elif bloc.get("fonction") == "ACF / PACF":
                st.markdown(bloc["commentaire"])
                
                if st.button("🎵 Lancer l'analyse ACF / PACF"):
                    acf_pacf_streamlit(serie)
            
            # Decomposition        
            elif bloc.get("fonction") == "Décomposition":
                st.markdown(bloc["commentaire"])
                
                if st.button("🎵 Lancer la décomposition"):
                    decomposition_streamlit(serie)
            # Corrélation          
            else:
                st.markdown(bloc["commentaire"])


# -----------------------------
# 7. Analyse de corrélation
# -----------------------------
elif page == "Analyse des corrélations":
    set_full_width()
    show_header()
    st.title("🔎 Analyse des corrélations")

    st.markdown("""Cette section présente l'analyse de corrélation entre
    
- les composantes de la consommation d'électricité 
- et les composantes des variables météorologiques
                    """)

    ANALYSES = {
    
       "🎵 Analyse de la composante saisonnière": {
            "commentaire": """Notre analyse a permis d'établir que:
            
- les composantes saisonnières sont stationnaires,
- elles dépendent uniquement de la configuration profil & plage de puissance souscrite, 
- elles sont indépendantes des variables exogènes 

\u2794  des modèles SARIMA sont bien adaptés pour la prévision de ces composantes.
                """,
            "fonction": "Décomposition"  
        },
        
        "📉 Analyse de la tendance ": {
            
            "commentaire": """
            Notre analyse a permis d'établir: 
            
            - l'existence d'une relation non linéaire entre la tendance de la consommation d'électricité et les tendances des variables exogènes,
            - la corrélation entre la tendance de la consommation d'électricité  et les séries obtenues par translation et inversion des tendances des variables exogènes.
            """,
            
            "options_tendance": 
            {
            "🌡️ Température": "tendances_T",
            "💧 Humidité": "tendances_U",
            "☀️ Rayonnement": "tendances_R"
            }
        },
        "📉 Analyse du résidu ": {
            
            "commentaire": """
            Notre analyse a permis d'établir: 
            
            - l'existence d'une faible corrélation entre le résidu de la consommation d'électricité  et les résidus des variables exogènes.
            """,
            
            "options_résidus": 
            {
            "🌡️ Température": "résidu_T",
            "💧 Humidité": "résidu_U",
            "☀️ Rayonnement": "résidu_R"
            }
        },
       
        "📌 **Conclusions de cette analyse**": {
            "commentaire": """
    
            - Relation non linéaire entre la tendance/le résidu de la consommation d'électricité et les tendances/résidus des variables exogènes,
            - Effet mémoire : adaptation aux changementx après un délai    
            - Transformation nécessaire pour augmenter la corrélation
            
            """
        }
    }
    # Préparation des données : filtrer + normalisation + décomposition
    df_fusion_filtred= load_and_filter_df_fusion(FOLDERS_Fusion)
    df_fusion_filtred = force_datetime_index(df_fusion_filtred)
    df_fusion_filtred = imputer_series(df_fusion_filtred, method='ffill', window=3) 
    df_fusion_filtred["Total énergie soutirée (Wh)"] = df_fusion_filtred["Total énergie soutirée (Wh)"] / df_fusion_filtred["Nb points soutirage"]
    # Choisir une partie 
    start = pd.Timestamp("2023-01-01 00:00")
    end = pd.Timestamp("2023-01-31 00:00")
    serie = df_fusion_filtred[['Total énergie soutirée (Wh)']]
    serie_T = df_fusion_filtred[['T_moyenne']]
    serie_U = df_fusion_filtred[['U_moyenne']]
    serie_R = df_fusion_filtred[['Rayonnement solaire global (W/m2)']]
    if df_fusion_filtred is not None:
        st.dataframe(df_fusion_filtred.head())
        
    for titre, bloc in ANALYSES.items():
        with st.expander(titre):
            
            st.markdown(bloc.get("commentaire", ""))  # commentaire systématiquement affiché

            # 1. Décomposition
            if bloc.get("fonction") == "Décomposition":
                if st.button("🎵 Lancer la décomposition"):
                    decomposition_streamlit(serie.loc[start:end])

            # 2. Tendance
            elif "options_tendance" in bloc:
                choix = st.selectbox("🔎 Choisir la variable météo à étudier :", list(bloc["options_tendance"].keys()), key=f"{titre}_tendance")
                nom_fonction = bloc["options_tendance"][choix]

                if nom_fonction == "tendances_T":
                    plot_correlation_tendances(serie, serie_T, 'T')
                elif nom_fonction == "tendances_U":
                    plot_correlation_tendances(serie, serie_U, 'U')
                elif nom_fonction == "tendances_R":
                    plot_correlation_tendances(serie, serie_R, 'R')

            # 3. Résidus
            elif "options_résidus" in bloc:
                choix = st.selectbox("🔎 Choisir la variable météo à étudier :", list(bloc["options_résidus"].keys()), key=f"{titre}_residu")
                nom_fonction = bloc["options_résidus"][choix]

                if nom_fonction == "résidu_T":
                    plot_correlation_residu(serie.loc[start:end], serie_T.loc[start:end], 'T')
                elif nom_fonction == "résidu_U":
                    plot_correlation_residu(serie.loc[start:end], serie_U.loc[start:end], 'U')
                elif nom_fonction == "résidu_R":
                    plot_correlation_residu(serie.loc[start:end], serie_R.loc[start:end], 'R')

            # 4. Bloc avec image (par défaut)
            elif "image" in bloc:
                st.image(bloc["image"], use_column_width=True)
                
# -----------------------------
# 8. Approche proposée
# -----------------------------
elif page == "Approche proposée":
    set_full_width()
    show_header()
    st.title("💡 Approche proposée")
    
    col1, col2 = st.columns(2)

    # with col1:
        # st.markdown("""
        # ### Idée de l’approche :

        # - Diviser pour mieux régner: 
            # - Extraire les composantes saisonnières 
            # - Extraire la tendance 
            # - Extraire le résidus
        # - Prévoir chaque composante par un modèle approprié
        # - Concevoir une architecture générique et adaptable
         # """)
    with col1:
        st.markdown("""
        ### Les grandes étapes :

        1. **Analyse spectrale de la série de consommation**
        2. **Décomposition des  séries temporelles (cible et variables exogènes)** :
            - Extraction des composantes : tendance, saisonnalités, résidu.
        
        2. **Modélisation des composantes** 
            - Saisonnières : **SARIMAX**. 
            - Tendance :  modèles multi-couches pour capter la dynamique à long terme.
            - Résidu : modèles multi-couches pour capter les corrélations temporelles fines, les non linéairités et les dépendances à court terme.

        3. **Recomposition finale** :
            - Produit des prédictions des composantes pour obtenir la prévision globale.""")
    with col2:
        st.markdown("""
        ### Modèle multi-couches :
        """)
        st.image("Chap3/Archi_lstm_multicouche_VF.png", use_column_width=True, caption="Architecture du modèle multi-couches ")
        
    
    st.markdown("""
        ### Architecture du modèle global :
        """)
    st.image("Chap3/Archi.png", caption="Architecture générale de la solution proposée")
    
    


# -----------------------------
# 9. Réalisation – Implémentation
# -----------------------------
if page == "Réalisation – Implémentation":
    set_full_width()
    show_header()
    st.title("Quelques détails de la réalisation – implémentation")
    
    st.markdown("""### **Architecture modulaire en pipelines** """)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - **Analyse spectrale** : encapsulé dans une classe `SpectrogramAnalysis` compatible `sklearn.pipeline`
        - **Décomposition**  : encapsulé dans une classe `DecompositionSerieTemporelle` compatible `sklearn.pipeline`
        - **SARIMAX** : 
        
            - encapsulé dans une classe `SARIMAModel` compatible `sklearn.pipeline`
            - encapsulé dans un pipeline avec analyse spectrale 
            - recherche par cross-validation du meilleur modèle
            """)
    with col2:
        st.markdown("""
        - **LSTM Tendance / Résidu** :
            - Classe `LSTMModel` compatible `sklearn.pipeline` 
            - Recherche par cross-validation de la meilleur structure
                - nombre de couches lstm et denses, nombre de neuronnes, fonction d'activation, taux d'apprentissage,taux de dropout, ...)  
            - Surveillance pendant l'entraînement et arrêt si nécessaire (`EarlyStopping` et `ReduceLROnPlateau`) 
            - Encapsulée dans un pipeline avec 
                - une classe de préparation des données
                - une classe de transformation (transaltion , inversion) des variables exogènes
                - une classe de **Normalisation** :
                    - **MinMaxScaler** pour les tendances 
                    - **StandardScaler** pour les résidus
            """)

    st.markdown("""
     ### Entraînement des modèles :
        - entraînés pour chaque configuration `(profil, puissance, région)`
        - entrainé sur  une année glissante
        - prévision  au pas de 30 minutes pour un horizon donné
        - cross-validation adaptées aux séries temporelles
        - sauvegarde des meilleurs modèles
       """)
# -----------------------------
# 10. Démonstration
# -----------------------------
elif page == "Démonstration":
    set_full_width()
    show_header()
    st.title("📊 Démonstration avec des modèles pré-entraînés")

    # Choix de la région, profil et puissance 
    df_fusion_filtred = load_and_filter_df_fusion(FOLDERS_Fusion)  #
    df_fusion_filtred = force_datetime_index(df_fusion_filtred)
    df_fusion_filtred = imputer_series(df_fusion_filtred, method='ffill', window=3) 
    st.subheader("Aperçu des données chargées")
    st.dataframe(df_fusion_filtred.head())
    
    # ------------------------------------------------------------------------------------------------------------------------------------------------
    # Constantes et Variables globales servant de paramètres par défaut pour les constructeurs 
    # ------------------------------------------------------------------------------------------------------------------------------------------------
    st.markdown("### ⏳ Horizon de prédiction")
    NOMBRE_JOUR_PREDICTION = st.slider(
    "Sélectionnez l'horizon de la prédiction en jours =  48 pas",
    min_value = 1,  # 1 jour
    max_value = 60, # 1 semaine
    value = 7,     # valeur par défaut
    step = 7
     )

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
    #NOMBRE_JOUR_PREDICTION = 15 #a évaluer
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


    

    if df_fusion_filtred is not None:
        run_demo_prediction(df_fusion_filtred)
    
    st.markdown("""
        **Nous constatons que** : 
        - **SARIMAX**  capte les **saisonnalités multi-périodiques** avec une précision remarquable (MAPE < 0.01%).
        - **LSTM** capte la **tendance** avec une très bonne précision (MAPE moyen ≈ 0.2%).
        - le modèle dédié aux **résidus**, a de bons résultas mais peine pour certaines configurations, la MAPE est variable selon la configuration.
            """)
   
# -----------------------------
# 11. Résultats
# -----------------------------
elif page == "Résultats":
    set_full_width()
    show_header()
    st.title("📊 Résultats globaux")
    afficher_resultats_globaux(FOLDER_RESULT)


    st.title("📊 Comparaison avec l'état de l'art")

    st.markdown("Comparaison des approches récentes (2019–2023) selon : MAPE, méthode, base de données, granularité et horizon.")

    # Tableau des résultats
    df = pd.DataFrame([
        {"Auteur": "Liu et al. (2023)", "Méthode": "CNN + LSTM + Attention", "Données": "Résidentiel agrégé + météo", "Pas": "30 min", "Horizon": "24h", "MAPE": ">7.49%"},
        {"Auteur": "Zhang et al. (2020)", "Méthode": "GBDT", "Données": "Foyers individuels", "Pas": "30 min", "Horizon": "6h", "MAPE": "5–6%"},
        {"Auteur": "Kong et al. (2019)", "Méthode": "LSTM", "Données": "Agrégats résidentiels", "Pas": "30 min", "Horizon": "24h", "MAPE": "2 à 8%"},
    ])

    st.dataframe(df, use_container_width=True)

    st.success("Notre modèle (MAPE < 4% sur un horizon > 7 jours, 30min) dépasse plusieurs références.")

# -----------------------------
# 12. Conclusion
# -----------------------------
elif page == "Conclusion":
    set_full_width()
    show_header()
    st.title("✅ Conclusion")
    col1, col2 = st.columns(2)
    with col1:
            st.markdown("#### Par rapport au projet")
            st.markdown(""" 
                            - Les objectifs fixés sont atteints
                            - La précision est globalement très satisfaisante,
                            - La modularité de l'approche permet 
                                - le raffinement/perfectionement du modèle pour les configurations problèmatiques
                                - l'intégration d’autres variables explicatives (comme les jours spéciaux par exemple),
                                - l'intégration des  spécification linéaire par morceaux des effects des variables explicatives (cf. le rapport)                                
                    """)
            st.markdown("#### Les perspectives liées au projets")
            st.markdown("""
                - Faire une publication scientifique
                - Perfectionner l'approche
                - Intégrer la détection d’anomalies basée sur l’écart modèle-observé.
                - Comparaison avec un éventuel modèle d'Enedis  
                    """)
    with col2:
            st.markdown("#### Personnellement")
            st.markdown(""" 
                            
                            - Le projet est très formateur, 
                            - Permet la mise en pratique des compétences developpées
                            - Une grande charge de travail --> induit un retard sur la formation
                            - Très bonne formation
                    """)
                
# -----------------------------
# Footer
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.info("Prévision de la consommation électrique - YS")

