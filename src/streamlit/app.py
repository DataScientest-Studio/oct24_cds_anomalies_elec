import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import io
import contextlib


def show_header():
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image("logos/dst-logo.svg", width=100)
    with col2:
        st.markdown("""
            # 🔌 Prévision de la Consommation d'électricité en France
            **Projet DataScientest | Youssef SERRESTOU**
        """)
        

# -----------------------------
# Pour les visualisations
# -----------------------------
# --- Dossiers de données ---
BASE_DIR = Path(r"D:\MesDocuments\Formation\DataScientist_PSL\Projet\BD")
FOLDERS = {
    "🌦️ Météo (CSV brut)": BASE_DIR / "Meteo" / "CSV",
    "⚡ Consommation par région": BASE_DIR / "conso-inf36-region",
    "🌞 Rayonnement par région": BASE_DIR / "Meteo" / "rayonnement"
    #"📍 Météo par région": BASE_DIR / "Meteo" / "region",
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
    return pd.read_csv(file_path, sep=';', nrows=nrows, low_memory=False)

def show_file_info(file: Path, sep: str):
    size_mb = get_file_size(file)
    st.markdown(f"**Nom :** `{file.name}` — **Taille :** {size_mb:.2f} Mo — **Séparateur :** `{sep}`")

    if st.button("📥 Charger un aperçu de 20 lignes", key=file.name):
        try:
            df = load_csv_preview(file, sep=sep)
            st.dataframe(df)
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement : {e}")

SEPARATORS = {
    "🌦️ Météo (CSV brut)": ';',
    "⚡ Consommation par région": ';',
    "🧪 Rayonnement par région": ','  
}
        
# -----------------------------
# Sidebar navigation
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", [
    "Préambule",
    "Contexte et problématique",
    "Formalisation du problème",
    "Données utilisées",
    "Analyse exploratoire",
    "Méthodologie",
    "Modèle et prévisions",
    "Résultats",
    "Démonstration"
    "Conclusion"
])

# -----------------------------
# Préambule
# -----------------------------
if page == "Préambule":
    show_header()
    st.markdown("""
    **Historique et contexte initial :**  
    - Projet proposé initialement par Guillaume ROTH 
    - Équipe initiale : Fei YANG, Guillaume ROTH, et Youssef SERRESTOU
    - Objectif du projet dans sa formulation initiale : détection d'anomalie de la consoammation d'électricité dans des batiments résidentiels 
   
    **Changements importants :**
    - Absence de données à l'échelle d'un habitat
    - Départ de la collègue Fei YANG pour un autre projet
    - Desintéressement de la part de Guillaume ROTH 
                
    **Nouveau Contexte :**
    - Suite à : 
        - Des échanges avec Enedis
        - Une étude de l'état de l'art
        - Et en concertation avec le tuteur de projet
    
    - **➔ L'objectif fixé est  :**  
    l'établissment d'un modèle de prévision de la consommation d'électricité, à court terme, des utilisateurs du réseau Enedis en France.
    """)
# -----------------------------
# 1. Contexte et problématique
# -----------------------------
if page == "Contexte et problématique":
    show_header()
    st.markdown(""" 
    **Contexte :**  
    Les deux prévisions à distinguer :
    -  la prévision de la charge électrique :
        - fournisseurs et opérateurs d’électricité  
        - Equilibre entre l’offre et la demande    
    -  La prévision de la consommation d'électricité :
        - estimation de la demande  
        - suivi et gestion de la consommation
    
    Type selon l'horizon:
    - Prévision à court terme < 1 semaine
    - Prévision à moyen terme < 1 an
    - Prévision à long terme > 1 an
    
    **Problématique :**  
    Construire un modèle robuste pour prévoir à court terme la consommation des utilisateurs du réseau Enedis, 
    en fonction de leur profil, de la puissance souscrite et d'autres variables explicatives.
     """)



# -----------------------------
# 2. Formalisation du problème
# -----------------------------
elif page == "Formalisation du problème":
    show_header()
    st.header("🧭 Formalisation du problème")
    st.markdown("""
    Dans cette section, nous présentons la formalisation mathématique de notre problème de prévision.
    Nous explicitons les notations et les hypothèses retenues.
  

    #st.image("figures/schema_donnees.png", caption="Schéma des données fusionnées (exemple)")

    **Objectifs :**
    - Fusionner données Enedis et météo
    - Analyser les corrélations
    - Construire un pipeline SARIMA + LSTM
    - Évaluer les prévisions (MAPE, MAE, RMSE)
   """)
# -----------------------------
# 3. Données utilisées
# -----------------------------
elif page == "Données utilisées":
    show_header()
    st.title("📈 Données utilisées")
    st.markdown("""
    **Sources :**
    - Données Enedis (consommation au pas 30 minutes, profils, régions)
    - Données météorologiques (température, humidité, vent, rayonnement)
    
    **Caractéristiques :**
    - Granularité : 30 min
    - Plage de puissance ≤ 36 kVA
    - Variables exogènes : météo sur Auvergne-Rhône-Alpes
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
# 3. Analyse exploratoire
# -----------------------------
elif page == "Analyse exploratoire":
    show_header()
    st.title("🧹 Analyse exploratoire")
    st.markdown("""
    - Visualisation des séries temporelles
    - Tests de stationnarité (ADF, KPSS)
    - Analyse ACF / PACF
    - Corrélations météo-consommation
    """)

    #st.image("figures/acf_pacf.png", caption="Exemple d'ACF / PACF sur la série différenciée")
    #st.image("figures/correlation_meteo.png", caption="Corrélations météo / consommation")
    
    st.latex(r"""
        \begin{cases}
        Y_t = T_t + S_t + R_t \\
        log(Y_t) = \log(T_t) + \log(S_t) + \log(R_t)
        \end{cases}
        """)
    st.markdown("""
    - Visualisation des séries temporelles
    - Tests de stationnarité (ADF, KPSS)
    - Analyse ACF / PACF
    - Corrélations météo-consommation
    """)
# -----------------------------
# 4. Méthodologie
# -----------------------------
elif page == "Méthodologie":
    show_header()
    st.title("🔬 Méthodologie")
    st.markdown("""
    - Décomposition additive / multiplicative
    - Analyse temps-fréquence pour détecter multi-saisonnalités
    - Pipeline :
        - SARIMA pour les composantes saisonnières
        - LSTM pour tendance et résidu
    """)

    #st.image("figures/pipeline_general.png", caption="Pipeline général")
    #st.image("figures/spectrogramme.png", caption="Analyse temps-fréquence")

# -----------------------------
# 5. Modèle et prévisions
# -----------------------------
elif page == "Modèle et prévisions":
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
# 6. Résultats
# -----------------------------
elif page == "Résultats":
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
# 7. Téléchargement
# -----------------------------
elif page == "Téléchargement":
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

