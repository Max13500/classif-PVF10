import streamlit as st
from streamlit_option_menu import option_menu

# Extracteurs personnalisés pour pipelines ML classiques (nécessaire pour le chargement des modèles avec joblib)
from extractors import BaseStatsExtractor,HOGExtractor, GLCMExtractor, EntropyExtractor, EdgeDensityExtractor, PixelsBrutsExtractor

# Fonctions de chargement des données / modèles
from app_setup import modeles,load_dataset,encode_labels,load_modele,predict_on_test

# Fonctions d'affichage des différentes pages
from app_views import show_presentation,show_dataviz,show_modelisation,show_demo,show_bilan

# Au lancement : chargement des données en cache avec barre de progression
if 'initialised' not in st.session_state:
  # Initialisation barre de progression
  progress = st.progress(0)
  status = st.empty()

  # Chargement du dafaframe (en cache)
  status.text("Chargement des données...")
  df_pvf10,X_train,X_test,y_train,y_test = load_dataset()
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
  st.success("Application prête !")
  st.session_state.initialised = True

# Par la suite : rechargement des données depuis le cache 
else:
   # Chargement du dataframe
  df_pvf10,X_train,X_test,y_train,y_test = load_dataset()

  # Encodage des labels
  encoder,y_train_enc,y_test_enc = encode_labels(y_train,y_test)

  # Chargement des modèles entraînés
  for modele_name in modeles :
    modeles[modele_name]["trained_model"] = load_modele(modele_name)

  # Prédictions sur les données de test
  for modele_name in modeles :
    modeles[modele_name]["predicted_data_test"] = predict_on_test(modele_name,X_test,tuple(encoder.classes_))


# Titre et navigation sur 5 pages
st.title("Classification des défauts sur des panneaux photovoltaïques")
pages = ["Présentation", "DataViz", "Modélisation","Démo","Bilan"]
with st.sidebar:
    page = option_menu(
        menu_title="Sommaire",
        options=pages,
        icons=["house", "bar-chart", "cpu", "image", "check-circle"],  # icônes Bootstrap
        menu_icon="cast",
        default_index=0,
    )

# Page Présentation
if page == pages[0] : 
  show_presentation(df_pvf10)

# Page DataViz
if page == pages[1] : 
  show_dataviz(df_pvf10)
  
# Page Modélisation
if page == pages[2] :
  show_modelisation(modeles,y_test)

# Page Démo
if page == pages[3] : 
  show_demo(modeles,X_test,y_test)

# Page bilan
if page == pages[4] :
  show_bilan()
