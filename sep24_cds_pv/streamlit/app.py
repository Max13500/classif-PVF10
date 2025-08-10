import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix

# Charger les extracteurs personnalisés pour pipelines ML classiques (nécessaire pour le chargement des modèles avec joblib)
from extractors import BaseStatsExtractor,HOGExtractor, GLCMExtractor, EntropyExtractor, EdgeDensityExtractor, PixelsBrutsExtractor

# Charger les fonctions de l'application
from utils import modeles,load_dataset,encode_labels,preprocess_on_test,load_modele,predict_on_test

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

  # Preprocessing des données de test (en cache)
  for modele in modeles:
    status.text(f"Preprocessing de l'ensemble de test pour {modele}...")
    modeles[modele]["preprocessed_data_test"] = preprocess_on_test(modele,X_test)
  progress.progress(50)

  # Chargement des modèles entraînés (en cache)
  for modele in modeles :
    status.text(f"Chargement du modèle {modele}...")
    modeles[modele]["trained_model"] = load_modele(modele)
  progress.progress(75)

  # Prédictions sur les données de test (en cache)
  for modele in modeles :
    status.text(f"Prédictions sur l'ensemble de test pour {modele}...")
    modeles[modele]["predicted_data_test"] = predict_on_test(modele,tuple(encoder.classes_))
  progress.progress(100)

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

  # Preprocessing des données de test
  for modele in modeles:
    modeles[modele]["preprocessed_data_test"] = preprocess_on_test(modele,X_test)

  # Chargement des modèles entraînés
  for modele in modeles :
    modeles[modele]["trained_model"] = load_modele(modele)

  # Prédictions sur les données de test
  for modele in modeles :
    modeles[modele]["predicted_data_test"] = predict_on_test(modele,tuple(encoder.classes_))


# Titre et navigation sur 3 pages
st.title("Classification des défauts sur des panneaux photovoltaïques")
pages = ["Exploration", "DataVizualization", "Modélisation"]
with st.sidebar:
    page = option_menu(
        menu_title="Sommaire",
        options=pages,
        icons=["search", "bar-chart", "cpu"],  # icônes Bootstrap
        menu_icon="cast",
        default_index=0,
    )

# Page Exploration
if page == pages[0] : 
  st.write("### Introduction")

  st.dataframe(df_pvf10.head(10))
  st.write(df_pvf10.shape)

# Page DataViz
if page == pages[1] : 
  st.write("### DataVizualization")

# Page modélisation
if page == pages[2] : 
   st.write("### Modélisation")
   option = st.selectbox('Choix du modèle', list(modeles.keys()))
   st.write(option)
   st.write(modeles[option]["preprocessed_data_test"].shape)
   st.metric("Accuracy", round(accuracy_score(y_test,modeles[option]["predicted_data_test"]), 3))
   
# Fonction de prédiction sur une image unique (PAS de cache)
#def predict_single_image(model, image_path, preprocess_fn):
#    img = preprocess_fn(image_path)  # resize...
#    img_batch = np.expand_dims(img, axis=0)  # pour batch=1
#    return model.predict(img_batch)
# Page "Démo"
#img_path = random.choice(X_test_df['Chemin'])
#pred = predict_single_image(model, img_path, preprocess_fn)