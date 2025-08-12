import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,classification_report, confusion_matrix

def show_presentation(df):
    st.header("Présentation")
    st.write("TODO : Présentation du contexte, des enjeux, et du dataset")


def show_dataviz(df):
    st.header("DataViz")
    st.write("TODO : Analyse des données avec figures")

    # Répartition des classes
    fig = plt.figure()
    sns.countplot(y = df['Classe'],hue = df['Classe'],legend=False)
    plt.title("Répartition des classes de défauts",fontsize=14, fontweight='bold')
    plt.xlabel("Nombre d'images")
    plt.ylabel("Classe de défaut")
    sns.despine()
    st.pyplot(fig)


def show_modelisation(modeles,y_test):
    st.header("Modélisation")
    st.write("TODO : Présentation des modèles (dont preprocessing) et de leurs résultats")

    # Choix du modèle
    modele_name = st.selectbox('Choix du modèle', list(modeles.keys()))

    st.subheader(f"Résultats du modèle {modele_name}")
    # Récupération des prédictions du modèle sur le jeu de test
    y_pred = modeles[modele_name]["predicted_data_test"]

    # Affichage côte à côte des métriques principales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", round(accuracy_score(y_test,y_pred), 3))
    with col2:
        st.metric("F1 macro", round(f1_score(y_test,y_pred,average="macro"), 3))
    with col3:
        st.metric("Précision Healthy", round(precision_score(y_test, y_pred, labels=["healthy panel"],average=None)[0], 3))
    with col4:
        st.metric("Rappel Healthy", round(recall_score(y_test, y_pred, labels=["healthy panel"],average=None)[0], 3))

    # Choix entre rapport de classification et matrice de confusion
    display = st.radio('Que souhaitez-vous afficher ?', ('Rapport de classification', 'Matrice de confusion'))

    # Affichage rapport de classification
    if display == 'Rapport de classification':
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        # Métriques par classe
        st.table(report_df.iloc[:-3,:].style.format(precision=2))
        # Métriques globales
        st.table(report_df.iloc[-2:,:].style.format(precision=2))

    # Affichage matrice de confusion
    elif display == 'Matrice de confusion':
        cm = confusion_matrix(y_test, y_pred)
        fig = plt.figure()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
        plt.xlabel("Prédictions")
        plt.ylabel("Vraies classes")
        st.pyplot(fig)


def show_demo(modeles,X_test,y_test):
    st.header("Démo")
    st.write("TODO : Checkbox Toutes les classes. Si décoché :  sélecteur Nb d'images entre 1 et 10 + sélecteur Classe 1 / ... / Classe 10 / Aléatoire." \
    "On affichera l'image, la classe réelle, et pour chaque modèle la classe prédite (rouge si ko, vert si ok) ")

# Fonction de prédiction sur une image unique (PAS de cache)
#def predict_single_image(model, image_path, preprocess_fn):
#    img = preprocess_fn(image_path)  # resize...
#    img_batch = np.expand_dims(img, axis=0)  # pour batch=1
#    return model.predict(img_batch)
# Page "Démo"
#img_path = random.choice(X_test_df['Chemin'])
#pred = predict_single_image(model, img_path, preprocess_fn)

def show_bilan():
    st.header("Bilan")
    st.write("TODO : conclusion sur meilleur modèle, conclusion métier, critique, perspectives")

