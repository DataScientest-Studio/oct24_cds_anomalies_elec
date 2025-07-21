import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import io
import contextlib
import seaborn as sns

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
    "Formalisation du problème",
    "Analyse des séries temporelles",
    "Méthodologie",
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
        ➔ **Nouvel objectif :**  
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
        - Recherche des données de consommation à utiliser : base de données Enedis , data.gouv,  échanges avec Enedis  
        - Détermination des facteurs influants (variables explicatives/exogènes) sur la consommation :  état de l'art
        - Recherche des bases de données pour inclure ces variables : 
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
    
        - Les données sont celle de la région Auvergne-Rhône-Alpes
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
    
    # for i in range(0, len(items), 2):
        # col1, col2 = st.columns(2)

        # with col1:
            # titre1, img1 = items[i]
            # with st.expander(titre1):
                # st.image(img1, use_column_width=True)

        # if i + 1 < len(items):
            # with col2:
                # titre2, img2 = items[i + 1]
                # with st.expander(titre2):
                    # st.image(img2, use_column_width=True)
        
        
    # st.title("🔎 Analyse exploratoire de la consommation")
    # folder_label = st.selectbox("📂 Choisissez un répertoire :", list(FOLDERS_Fusion.keys()))
    # folder_path = FOLDERS_Fusion[folder_label]
    # sep = ','
    


    # if not folder_path.exists():
        # st.error(f"Le dossier `{folder_path}` n’existe pas.")
      

    # csv_files = list_csv_files(folder_path)
    # if not csv_files:
        # st.warning("Aucun fichier CSV trouvé dans ce dossier.")
    
   
    # selected_file = st.selectbox("📄 Choisissez un fichier CSV :", csv_files)
    # df_fusion = load_csv_entier(selected_file)  # ou load complet si besoin
    # show_exploratory_analysis(df_fusion)

# -----------------------------
# 5. Formalisation du problème
# -----------------------------
elif page == "Formalisation du problème":
    set_full_width()
    show_header()
    st.title("Formalisation du problème")
    
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
    st.title("🧹 Analyse des séries temporelles")
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
# 5. Méthodologie
# -----------------------------
elif page == "Méthodologie":
    set_full_width()
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

