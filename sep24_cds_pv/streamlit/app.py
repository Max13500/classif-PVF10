import streamlit as st
from streamlit_option_menu import option_menu

# Extracteurs personnalisés pour pipelines ML classiques (nécessaire pour le chargement des modèles avec joblib)
from extractors import BaseStatsExtractor,HOGExtractor, GLCMExtractor, EntropyExtractor, EdgeDensityExtractor, PixelsBrutsExtractor

# Fonctions de chargement des données / modèles
from app_setup import modeles,load_dataset,calculate_stats,encode_labels,load_modele,predict_on_test

# Fonctions d'affichage des différentes pages
from app_views import load_image,show_presentation,show_dataviz,show_method,show_results,show_demo,show_bilan

# Configuration générale de la page
st.set_page_config(
    page_title="Classification PVF-10",
    page_icon="resources/logo.png"
)
# Sidebar anthracite
st.markdown(
    """
    <style>
        [data-testid="stSidebar"] {
            background-color: #2C3E50;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Au lancement : chargement des données en cache avec barre de progression
if 'initialised' not in st.session_state:
  # Initialisation barre de progression
  progress = st.progress(0)
  status = st.empty()

  # Chargement du dafaframe (en cache)
  status.text("Chargement des données...")
  df_pvf10,X_train,X_test,y_train,y_test = load_dataset()
  progress.progress(10)

  # Calcul des statistiques (en cache)
  status.text("Calcul des statistiques...")
  statistisques = calculate_stats(df_pvf10)
  progress.progress(20)

  # Encodage des labels (en cache)
  status.text("Encodage des labels...")
  encoder,y_train_enc,y_test_enc = encode_labels(y_train,y_test)
  progress.progress(25)

  # Chargement des modèles entraînés (en cache)
  for i,modele_name in enumerate(modeles) :
    status.text(f"Chargement du modèle {modele_name}...")
    modeles[modele_name]["trained_model"] = load_modele(modele_name)
    progress.progress(int(25 + (i+1) * (50-25) / len(modeles)))

  # Prédictions sur les données de test (en cache)
  for i,modele_name in enumerate(modeles) :
    status.text(f"Prédictions sur l'ensemble de test pour {modele_name}...")
    modeles[modele_name]["predicted_data_test"] = predict_on_test(modele_name,X_test,tuple(encoder.classes_))
    progress.progress(int(50 + (i+1) * (100-50) / len(modeles)))

  # Fin du chargement : supprimer la barre de progression
  progress.empty()
  status.empty()
  st.toast("Application prête !",icon="✅")
  st.session_state.initialised = True

# Par la suite : rechargement des données depuis le cache
else:
   # Chargement du dataframe
  df_pvf10,X_train,X_test,y_train,y_test = load_dataset()

  # Calcul des statistiques 
  statistisques = calculate_stats(df_pvf10)

  # Encodage des labels
  encoder,y_train_enc,y_test_enc = encode_labels(y_train,y_test)

  # Chargement des modèles entraînés
  for modele_name in modeles :
    modeles[modele_name]["trained_model"] = load_modele(modele_name)

  # Prédictions sur les données de test
  for modele_name in modeles :
    modeles[modele_name]["predicted_data_test"] = predict_on_test(modele_name,X_test,tuple(encoder.classes_))

# Navigation sur 6 pages
pages = ["Présentation", "DataViz", "Méthode", "Résultats", "Démo","Bilan"]
# Sidebar pour la navigation
with st.sidebar:
    # Image drone
    st.image(load_image("resources/img_sommaire.png"))
    # Menu
    page = option_menu(
        menu_title="Sommaire",
        options=pages,
        icons=["house", "bar-chart", "cpu", "graph-up-arrow", "image", "check-circle"],  # icônes Bootstrap
        menu_icon="cast",
        default_index=0,
        styles={
          "container": {
                "background-color": "#2C3E50",
                "border-radius": "12px",
                "border": "2px solid white"
            },
            "icon": {
                "color": "white"
            },
            "nav-link": {
                "color": "white",
                "--hover-color": "#34495E"
            },
            "nav-link-selected": {
                "background-color": "#1ABC9C",
                "color": "white"
            },
            "menu-title": { 
                "color": "white"
            },
            "menu-icon": {
                "color": "white"
            }
          }
    )
    # Contact
    st.html(
    '''
    <p style="font-size:20px; font-weight:bold;">
      Equipe Projet :
    </p>
    <p style="font-size:18px;">
        Maxime Benoit
        <a href="https://fr.linkedin.com/in/maxime-benoit-92004a329" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" 
                 width="20" style="vertical-align:middle; margin-left:8px;">
        </a>
    <br>
        Sylvain Cordier
        <a href="https://fr.linkedin.com/in/sylvain-cordier" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" 
                 width="20" style="vertical-align:middle; margin-left:8px;">
        </a>
    <br>
        Philippe Marechal
        <a href="https://fr.linkedin.com/in/philippe-marechal-74a24a4" target="_blank">
            <img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" 
                 width="20" style="vertical-align:middle; margin-left:8px;">
        </a>
    </p>
    ''')
    # Datascientest
    st.html(
    '''
    <p style="font-size:20px; font-weight:bold;">
      Formation continue Data Scientist
      <a href="https://datascientest.com" target="_blank">
          <img src="https://datascientest.com/wp-content/uploads/2020/08/new-logo.png" 
                width="30" style="vertical-align:middle; margin-right:8px;">
      </a>
    <br>
      <span style = "font-weight:normal">Nov 2024 - Sep 2025</span>
    </p>
    ''')

# Page Présentation
if page == pages[0] : 
  show_presentation(df_pvf10)

# Page DataViz
if page == pages[1] : 
  show_dataviz(df_pvf10,statistisques)
  
# Page Méthode
if page == pages[2] :
  show_method()

# Page Résultats
if page == pages[3] :
  show_results(modeles,y_test)

# Page Démo
if page == pages[4] : 
  show_demo(modeles,X_test,y_test)

# Page Bilan
if page == pages[5] :
  show_bilan(modeles,y_test)
