import streamlit as st
from PIL import Image
import cv2
from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
import joblib

# Chemins Données et modèles
PVF10_RAW_DATA_DIR = "../../data/raw/"
PVF10_CSV_FILE = "../../data/processed/structure_pvf_10.csv"
TRAINED_MODELS_DIR = "../../models/"

# Modèles
modeles =  {
   'SVM' : {
      'methodo_name':'Machine Learning avec features localisés',
      'file':'final_svm.joblib',
      'trained_model':None,
      'predicted_data_test':None
   },
   'XGBoost' : {
      'methodo_name':'Machine Learning avec features non localisés',
      'file':'final_xgboost.joblib',
      'trained_model':None,
      'predicted_data_test':None
   },
   'CNN Perso' : {
      'methodo_name':'Deep Learning',
      'file':'final_cnn.keras',
      'trained_model':None,
      'predicted_data_test':None
   },
   'MobileNet' : {
      'methodo_name':'Transfer Learning',
      'file':'final_mobilenet.keras',
      'trained_model':None,
      'predicted_data_test':None
   }
}

# Fonction de chargement du dataset (avec mise en cache)
@st.cache_data
def load_dataset():
    # Chargement du dataframe
    df_pvf10 = pd.read_csv(PVF10_CSV_FILE)
    # Filtre sur le format 110x60
    df_pvf10 = df_pvf10.loc[(df_pvf10['Format'] == "110x60")]
    # Adaptation du chemin des images 
    df_pvf10['Chemin'] = df_pvf10['Chemin'].str.replace("../data/raw/",PVF10_RAW_DATA_DIR)
    df_pvf10['Chemin'] = df_pvf10['Chemin'].str.replace("\\","/")

    # Séparation train / test conforme au dataset d'origine
    df_train = df_pvf10.loc[(df_pvf10['Train_Test'] == 'train')] 
    df_test = df_pvf10.loc[(df_pvf10['Train_Test'] == 'test')] 

    # Séparation features / target
    X_train = df_train.drop('Classe',axis=1)
    y_train = df_train['Classe']
    X_test = df_test.drop('Classe',axis=1)
    y_test = df_test['Classe']
    return df_pvf10,X_train,X_test,y_train,y_test

# Fonction de calculs de statistiques sur les différentes classes (avec mise en cache)
@st.cache_data
def calculate_stats(df,calculate_glcm=False):
   # Dicos de statistiques par classe
   rgb_moy = {}
   intensites_min = {}
   intensites_max = {}
   intensites_median = {}
   intensites_moy = {}
   intensites_std = {}
   entropies={}
   edge_densities={}
   contrastes = {}
   homogeneites = {}
   energies = {}
   correlations = {}
   # Pour chaque classe
   for classe in df["Classe"].unique():
      # Initialisation des statistiques
      rgb_moy[classe] = []
      intensites_min[classe] = []
      intensites_max[classe] = []
      intensites_median[classe] = []
      intensites_moy[classe] = []
      intensites_std[classe] = []
      entropies[classe] = []
      edge_densities[classe] = []
      contrastes[classe] = []
      homogeneites[classe] = []
      energies[classe] = []
      correlations[classe] = []
      # Pour chaque image de la classe
      for chemin in df.loc[(df["Classe"]==classe)]["Chemin"]:
         # Récupération de l'image en couleurs
         img_rgb = Image.open(chemin).convert("RGB")
         img_rgb_array = np.array(img_rgb)
         # Calcul et stockage des indicateurs statistiques sur les canaux RGB
         rgb_moy[classe].append(np.mean(img_rgb_array, axis=(0, 1)))
         # Récupération de l'image en NdG
         img_gray = Image.open(chemin).convert("L") 
         img_gray_array = np.array(img_gray)
         # Calcul et stockage des indicateurs statistiques sur le NdG
         intensites_min[classe].append(np.min(img_gray_array))
         intensites_max[classe].append(np.max(img_gray_array))
         intensites_median[classe].append(np.median(img_gray_array))
         intensites_moy[classe].append(np.mean(img_gray_array))
         intensites_std[classe].append(np.std(img_gray_array))
         # Calcul et stockage de l'entropie
         entropies[classe].append(shannon_entropy(img_gray_array))
         # Calcul et stockage de la densité de contours
         edges = cv2.Canny(img_gray_array[2:-2,2:-2], 70, 140) # Filtre de Canny sur l'image rognée
         edge_density =  np.sum(edges > 0) / edges.size # On en déduit la densité de contours
         edge_densities[classe].append(edge_density)
         # Calcul et stockage des propriétés GLCM
         if (calculate_glcm):
            glcm = graycomatrix(img_gray_array, distances=[8], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256) # Calcul de la matrice GLCM
            contrastes[classe].append(graycoprops(glcm, 'contrast').mean())
            homogeneites[classe].append(graycoprops(glcm, 'homogeneity').mean())
            energies[classe].append(graycoprops(glcm, 'energy').mean())
            correlations[classe].append(graycoprops(glcm, 'correlation').mean())

   # On renvoie un dictionnaire global pour ces statistiques
   return {
      "Moyenne des canaux RGB":rgb_moy,
      "Min des Niveaux de gris":intensites_min,
      "Max des Niveaux de gris":intensites_max,
      "Mediane des Niveaux de gris":intensites_median,
      "Moyenne des Niveaux de gris":intensites_moy,
      "Ecart-type des Niveaux de gris":intensites_std,
      "Entropie":entropies,
      "Densite de contours":edge_densities,
      "Contraste":contrastes,
      "Homogeneite":homogeneites,
      "Energie":energies,
      "Correlation":correlations
   }

# Fonction d'encodage des labels
@st.cache_data
def encode_labels(y_train,y_test):
  encoder = LabelEncoder()
  y_train_enc = encoder.fit_transform(y_train)
  y_test_enc = encoder.transform(y_test)
  return encoder,y_train_enc,y_test_enc

# Fonction de chargement d'un modèle (avec mise en cache)
@st.cache_resource
def load_modele(modele_name):
    # CNN : chargement keras
    if modele_name=="CNN Perso":
      return load_model(f"{TRAINED_MODELS_DIR}{modeles[modele_name]['file']}")
    # MobileNet : chargement keras avec preprocess_input
    elif modele_name=="MobileNet":
       return load_model(f"{TRAINED_MODELS_DIR}{modeles[modele_name]['file']}",custom_objects={"preprocess_input": preprocess_input})
    # Pipelines ML : chargement joblib
    else:
       return joblib.load(f"{TRAINED_MODELS_DIR}{modeles[modele_name]['file']}")

# Chargement d'une image pour un modèle de deep learning
def load_image_DL(path,modele_name):
   # Lecture du fichier
   image = tf.io.read_file(path) 
   # Chargement de l'image 
   image = tf.image.decode_png(image, channels=3) 
   # Conversion en niveaux de gris
   image = tf.image.rgb_to_grayscale(image) 
   # Le modèle CNN Perso attend en entrée un format 110x60
   if modele_name=="CNN Perso":
      image = tf.image.resize(image, [110, 60])
   # Le modèle MobileNet attend en entrée un format 128x128x3
   elif modele_name=="MobileNet":
      image = tf.image.grayscale_to_rgb(image)
      image = tf.image.resize(image, [128, 128])
   return image

# Fonction de prédiction des données de test d'un modèle (avec mise en cache) 
@st.cache_data
def predict_on_test(modele_name,X_test,classes):
   # Les modèles DL attendent des images d'un format précis en entrée et renvoient une probabilité par classe
   if modele_name in ("CNN Perso","MobileNet"):
      # Chargement des images de test au format attendu par le modèle
      X_test_img = np.array([load_image_DL(p,modele_name) for p in X_test['Chemin']])
      # Prédictions (=> 10 probas)
      preds = modeles[modele_name]["trained_model"].predict(X_test_img)
      # Récupération de la meilleure classe de probabilité (=> numero de classe)
      y_pred_num = np.argmax(preds, axis=-1)
      # Obtention du label de la classe
      y_pred = np.array(classes)[y_pred_num]
   # Le modèle basé sur XGBoost attend un chemin en entrée et renvoie le numero de la classe prédite
   elif modele_name == 'XGBoost' :
      # Prédiction (=> numero de classe)
      preds = modeles[modele_name]["trained_model"].predict(X_test[['Chemin']])
      # Obtention du label de la classe
      y_pred = np.array(classes)[preds]
   # Les autres modèles ML attendent un chemin en entrée et renvoient directement le label de la classe prédite 
   else:
      # Prédiction (=> label de classe)
      preds = modeles[modele_name]["trained_model"].predict(X_test[['Chemin']])
      # Obtention du label de la classe
      y_pred = preds
   return y_pred