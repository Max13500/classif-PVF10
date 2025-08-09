import streamlit as st
from streamlit_option_menu import option_menu
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
      'file':'final_svm.joblib',
      'preprocessed_data_test':None,
      'trained_model':None,
      'predicted_data_test':None
   },
   'CNN Perso' : {
      'file':'final_cnn.keras',
      'preprocessed_data_test':None,
      'trained_model':None,
      'predicted_data_test':None
   },
   'MobileNet' : {
      'file':'final_mobilenet.keras',
      'preprocessed_data_test':None,
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

# Fonction d'encodage des labels
@st.cache_data
def encode_labels(y_train,y_test):
  encoder = LabelEncoder()
  y_train_enc = encoder.fit_transform(y_train)
  y_test_enc = encoder.transform(y_test)
  return encoder,y_train_enc,y_test_enc

# Preprocessing d'une image pour CNN Perso
def preprocess_path_CNN(path):
    image = tf.io.read_file(path) # Lecture du fichier
    image = tf.image.decode_png(image, channels=3) # Chargement de l'image 
    image = tf.image.rgb_to_grayscale(image) # Conversion en niveaux de gris
    image = tf.image.resize(image, [110, 60]) # Redimensionnement en 110x60
    return image

# Preprocessing d'une image pour MobileNet
def preprocess_path_MobileNet(path):
    image = tf.io.read_file(path) # Lecture du fichier
    image = tf.image.decode_png(image, channels=3) # Chargement de l'image 
    image = tf.image.rgb_to_grayscale(image) # Conversion en niveaux de gris
    image = tf.image.grayscale_to_rgb(image) # Passage en 3 canaux pour compatibilité MobileNet
    image = tf.image.resize(image, [128, 128]) # Redimensionnement en 128x128
    return image

# Fonction de preprocessing des données de test d'un modèle (avec mise en cache) 
@st.cache_data
def preprocess_on_test(modele,X_test):
   if modele=="CNN Perso":
      return np.array([preprocess_path_CNN(p) for p in X_test['Chemin']])
   elif modele=="MobileNet":
      return np.array([preprocess_path_MobileNet(p) for p in X_test['Chemin']])
   # Pipelines ML
   else:
      return X_test[['Chemin']]

# Fonction de chargement d'un modèle (avec mise en cache)
@st.cache_resource
def load_modele(modele):
    if modele=="CNN Perso":
      return load_model(f"{TRAINED_MODELS_DIR}{modeles[modele]['file']}")
    elif modele=="MobileNet":
       return load_model(f"{TRAINED_MODELS_DIR}{modeles[modele]['file']}",custom_objects={"preprocess_input": preprocess_input})
    # Pipelines ML
    else:
       return joblib.load(f"{TRAINED_MODELS_DIR}{modeles[modele]['file']}")

# Fonction de prédiction des données de test d'un modèle (avec mise en cache) 
@st.cache_data
def predict_on_test(modele,classes):
    preds = modeles[modele]["trained_model"].predict(modeles[modele]["preprocessed_data_test"])
    if modele in ("CNN Perso","MobileNet"):
      y_pred_num = np.argmax(preds, axis=-1)
      y_pred = np.array(classes)[y_pred_num]
    else:
       y_pred = preds
    return y_pred