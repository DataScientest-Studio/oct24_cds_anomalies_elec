import streamlit as st
import pandas as pd
import numpy as np

import os
import sys
import io

import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import contextlib

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

from sklearn.preprocessing import MinMaxScaler


# Ajouter le dossier où se trouve le module analyse_spectrale.py
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# ➤ Maintenant tu peux importer ton module depuis src/models/
import importlib
import models.analyse_spectrale

importlib.reload(models.analyse_spectrale)

from models.analyse_spectrale import SpectrogramAnalysis

# --- Entête ----
def show_header():
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image("logos/dst-logo.svg", width=100)
    with col2:
        st.markdown("""
            # 🔌 Prévision de la Consommation d'électricité en France
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

    #st.success(f"✅ {len(df_fusion_filtred)} lignes sélectionnées pour {profil_selectionne} / {puissance_selectionnee}")

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
    P = int(TT.iloc[0,0])
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
        moyenne_mobile_T = serie_temp.rolling(window=48*7).mean()
    else : 
        moyenne_mobile_T = 1 / (serie_temp.rolling(window=48*7).mean() + 1 - np.min(serie_temp))

    moyenne_mobile_normalise = scaler.fit_transform(moyenne_mobile.values.reshape(-1,1))
    moyenne_mobile_T_normalisee = scaler.fit_transform(moyenne_mobile_T.values.reshape(-1,1))
    # Corrélation de Pearson
    #coeff = np.corrcoef(moyenne_mobile_normalise, moyenne_mobile_T_normalisee)[0, 1]
    
    
    fig = plt.figure(figsize=(12, 6))
    plt.plot(moyenne_mobile_normalise, 
             label='Tendance normalisée de la consommation', linestyle='--', color='black')
    plt.plot(moyenne_mobile_T_normalisee, 
             label='Tendance transformée et normalisée de la variable météo cible', linestyle='--', color='red')

    
    
    plt.title("Corrélation des tendances")
    plt.xlabel("Temps – pas = 30 minutes")
    plt.ylabel("Valeur (échelle normalisée)")
    plt.legend()
    plt.grid(True)
    
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
    "Analyse de corrélation",
    "Approche proposée",
    "Modèle et prévisions",
    "Résultats",
    "Démonstration"
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
        -  Désengagement progressif de **Guillaume ROTH**
        """)

    with col2:
        st.markdown("""
        ### 🎯 Objectif du projet
        Suite à :  
        - 🤝 Des échanges avec **Enedis**  
        - 📚 Une **étude de l’état de l’art**  
        - 👨‍🏫 Une concertation avec le **tuteur du projet**

        ---
        ➔ _Établir un modèle de prévision de la consommation d’électricité à court terme pour les utilisateurs du réseau Enedis en France._
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
        -  la prévision de la charge électrique :
            - imortante pour les fournisseurs et opérateurs d’électricité  
            - sert à équilibrer l’offre et la demande 
            - largement étudiée
        -  **La prévision de la consommation d'électricité** :
            - importante pour les consommateurs et pour les producteurs
            - sert à pérdire la consommation réelle 
            - moins abordée dans les études publiées
        
        Type selon l'horizon:
        - **Prévision à court terme** < 1 semaine
        - Prévision à moyen terme > 1 semaine et < 1 an
        - Prévision à long terme > 1 an
        """)
    
    with col2:
        st.markdown("""     
        **Notre objectif:**  
        Construire un modèle fiable, robuste et précis pour prévoir à court terme la consommation des utilisateurs du réseau Enedis, 
        en fonction des variables explicatives.
        
        
        **Les étapes de réalisation du projet:**  
        - Recherche des données de consommation à utiliser ➔ base de données Enedis , data.gouv,  échanges avec Enedis  
        - Détermination des facteurs influants sur la consommation ➔  état de l'art
        - Recherche des bases de données pour inclure ces variables 
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
        - Données restituant l'énergie totale soutirée au pas de 30 minutes d'agrégats de points de soutirage 
        - Plage de puissance ≤ 36 kVA
        - Agrégées par région, Profil et Plage de puissance souscrite 
        - Période choisie : 2023-2024
            
        2. **Données de Météo-Franceo-France**
            - Température (°C), 
            - Humidité (%), 
            - Vitesse du vent (m/s), 
            
        **Caractéristiques :**
        - Granularité temporelle (échantillonnage temporel) : un pas d'une heure 
        - Données de toutes les stations météo d'un département 
        - Période choisie : 2023-2024
    """)
    with col2:
        st.markdown(""" 
        3. **Données de Météo-Franceo-France**
            - nébulosité remplacée par le rayonnement global (W/m2)
        
        **Caractéristiques :**
        - Echantillonnage temporel : tri-horaire 
        - Données régionnales 
        - Période choisie : 2023-2024
      
        4. **Données calendaires (jours spéciaux)**
    """)    
    
    #st.image("figures/schema_donnees.png", caption="Schéma des données fusionnées (exemple)")

    st.title("📁 Visualisation rapide des CSV par répertoire")

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
        1. **Probème d'échantillonnage temporelle**:
            - Période d'échantillonnage choisie : 30 minutes
            - sur-échantillonnage par interpolation linéaire des données météorologiques 
            

        """)
        st.title("📁 Visualisation rapide des CSV par répertoire")
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
    
        - Les données sont utilisées pour générer les figures issues de la région Auvergne-Rhône-Alpes
        - la consommation a été divisée par le nombre de points de soutirage        
    
    """)

    FACTEURS = {
        "🕒 Saisonnalité intra-journalière": "Chap2/conso_par_heure.png",
        "📅 Saisonnalité annuelle": "Chap2/SaisonnaliteAnnuelle.png",
        "🧍 Influence du profil": "Chap2/conso_par_profile.png",
        "⚡ Influence de la puissance souscrite": "Chap2/ConsoPlagePuissance.png",
        "🌤️ Influence des facteurs météorologiques": "Chap2/conso_vs_facteurs_meteo.png",
        "📆 Influence des jours de semaine/week-end": "Chap2/effetjour.png"
    }

    # Convertir les éléments en liste pour itération par 2
    items = list(FACTEURS.items())

    # Affichage en deux colonnes
    for i in range(0, len(items)):
        titre, img = items[i]
        with st.expander(titre):
                st.image(img, use_column_width=True)
    

# -----------------------------
# 5. Représentation du problème
# -----------------------------
elif page == "Représentation du problème":
    set_full_width()
    show_header()
    st.title("Représentation du problème")
    
    st.markdown("### Représentation")
    st.markdown("""
        Pour toute configuration profil et palge de puissance sosucrites dans une région :
        - la consommation est une série temporelle
        - les facteurs météorologiques  sont des série temporelle""")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Les facteurs météorologiques")
        st.markdown(" - **Température moyenne (°C)** dans la région \\(r\\).")
        st.latex(r"""\left(T_{k}^{(r)}\right)_{kT_{s} \in \mathbb{T}}""")
        
        st.markdown(" - **Humidité moyenne (%)** dans la région \\(r\\).")
        st.latex(r"""\left(U_{k}^{(r)}\right)_{kT_{s} \in \mathbb{T}}""")
        
        st.markdown(" - **Rayonnement solaire global (W/m2)** dans la région \\(r\\).")
        st.latex(r"""\left(R_{k}^{(r)}\right)_{kT_{s} \in \mathbb{T}}""")
        
        
          
    with col2:

        st.markdown("### La variable cible")
        st.markdown("Série temporelle de la **consommation moyenne** par configuration(profil - palge de puissance sosucrites fixés).")
        st.latex(r"""\left(\overline{Y}_{k}^{(r,q)}\right)_{kT_{s} \in \mathbb{T}} = \left(\frac{Y_{k}^{(r,q)}}{N_{k}^{(r,q)}}\right)_{kT_{s} \in \mathbb{T}}""")
        
        st.markdown("Série temporelle de la **consommation d’électricité (en Wh)** pour une configuration q, dans une région r.")
        st.latex(r"""\left(Y_{k}^{(r,q)}\right)_{kT_{s} \in \mathbb{T}}""")
        
        st.markdown("Série temporelle représentant le **nombre de points de soutirage**")
        st.latex(r"""\left(N_{k}^{(r,q)}\right)_{kT_{s} \in \mathbb{T}}""")
        
        
    st.markdown("L’ensemble des **instants d’observation** disponibles dans notre base couvre la période du **01/01/2023** au **31/12/2024**")
    st.latex(r"""\mathbb{T} = \left\{ kT_{s},\; k \in \left\{ 0,\ldots,L \right\} \right\}""") 
    st.markdown("Le pas temporel d’échantillonnage est:")
    st.latex(r"""T_s= 30 \ minutes""")
 
     

  
# -----------------------------
# 6. Analyse des séries temporelles
# -----------------------------
elif page == "Analyse des séries temporelles":
    set_full_width()
    show_header()
    st.title("🔎 Analyse temporelle et spectrale des séries de consommation")

    st.markdown("Cette section explore différentes propriétés étudiées de nos séries temporelles avant de présenter la modélisation proposée.")

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
            - mettre en évidence les **périodes dominantes** dans la série (fréquences). 
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
            La série est décomposée en trois composantes :  
            - **Tendance**
            - **Saisonnalité**
            - **Résidu**  
            Cela permet de mieux modéliser chaque aspect séparément (ex : LSTM pour la tendance, SARIMAX pour la saisonnalité).
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
                    col_name = st.selectbox("📈 Choisir une variable", df_fusion_filtred.columns)

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
elif page == "Analyse de corrélation":
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
    st.title("🧠 Approche proposée – Architecture générale")

    st.markdown("""
    ### Objectif de l’approche :

    - Réaliser une **prédiction à court terme** de la consommation électrique (pas de 30 min, 1 semaine).
    - Exploiter la **structure multi-saisonnière** de la série temporelle.
    - Décomposer le signal pour le modéliser plus efficacement.

    ### Étapes clés de l'architecture :

    1. **Décomposition de la série temporelle** :
        - Extraction des composantes : tendance, saisonnalité, résidu.
    
    2. **Modélisation des composantes** :
        - Saison : **SARIMAX**, car très efficace pour capter la saisonnalité.
        - Tendance : **LSTM**, pour capter la dynamique à long terme.
        - Résidu : possibilité d’un traitement résiduel (non modélisé ici).

    3. **Recomposition finale** :
        - Somme des prédictions des composantes pour obtenir la prévision globale.

    ### Diagramme d’architecture :
    """)
    st.image("Chap3/Archi.png", caption="Architecture générale de la solution hybride SARIMAX + LSTM", use_column_width=True)

# -----------------------------
# 6. Modèle et prévisions
# -----------------------------
elif page == "Modèle et prévisions":
    set_full_width()
    show_header()
    st.title("🤖 Modèle et prévisions")
    st.markdown("""
    **Modèle SARIMA :**
    - Paramètres choisis après ACF/PACF
    - Différentiation pour stationnarité

    **Modèle LSTM :**
    - Architecture adaptée aux séquences
    - Entraîné sur les composantes décomposées

    **Prévisions :**
    - Résultats sur données test
    - Graphiques prévision vs vérité terrain
    """)

    #st.image("figures/prevision_sarima.png", caption="Exemple de prévision SARIMA")
    #st.image("figures/prevision_lstm.png", caption="Exemple de prévision LSTM")

# -----------------------------
# 7. Résultats
# -----------------------------
elif page == "Résultats":
    set_full_width()
    show_header()
    st.title("📊 Résultats et évaluation")
    st.markdown("""
    - MAPE global
    - MAE et RMSE par profil et plage de puissance
    - Distribution des erreurs
    """)

    #st.image("figures/mape_distribution.png", caption="Distribution du MAPE")
    #st.image("figures/mae_rmse.png", caption="MAE / RMSE par profil")

# -----------------------------
# 8. Téléchargement
# -----------------------------
elif page == "Téléchargement":
    set_full_width()
    show_header()
    st.title("📥 Télécharger les résultats")
    st.markdown("Téléchargez ici les prévisions générées au format CSV.")

    csv_file = st.file_uploader("Uploader le CSV des prévisions", type=["csv"])
    if csv_file:
        st.download_button(
            label="📥 Télécharger le fichier",
            data=csv_file.getvalue(),
            file_name="previsions.csv",
            mime="text/csv"
        )

# -----------------------------
# Footer
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.info("Projet DataScientest - Prévision de la consommation électrique")

