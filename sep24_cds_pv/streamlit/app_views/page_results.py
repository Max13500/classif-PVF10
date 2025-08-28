import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

from .utils import load_image


def show_results(modeles,y_test):
    st.header("Résultats",divider="gray")

    st.markdown("""
    Nous présentons les meilleurs modèles obtenus après optimisation, pour les 4 approches mentionnées.
    """)

    # Choix du modèle par l'utilisateur  
    modele_name = st.selectbox(
        "Choix du modèle",
        list(modeles.keys()),
        format_func=lambda name: f"{modeles[name]['methodo_name']} : modèle {name}"
    )

    st.subheader(f"Architecture du modèle {modele_name}")

    # Description de l'architecture du modèle
    if modele_name == "SVM":
        st.markdown("""
                    Les meilleurs résultats de l'approche ML avec features localisés ont été obtenus avec le classifieur SVM (noyau rbf) après ces étapes de preprocessing :
                    - extraction de features : vecteur de pixels bruts + descripteur HOG + statistiques d'entropie
                    - normalisation Min-Max
                    - réduction de dimensions par PCA conservant 90% de la variance
                    """)
    if modele_name == "XGBoost":
        st.markdown("""
                    Les meilleurs résultats de l'approche ML avec features non localisés ont été obtenus avec le classifieur XGBoost après extraction des features,
                    sans étape de preprocessing supplémentaire : statistiques sur les intensités + propriétés extraites de GLCM + statistiques d'entropie + densité de contours
                    """)
    if modele_name == "CNN Perso":
        st.markdown("""
                    Les meilleurs résultats de l'approche Deep Learning ont été obtenus avec ce réseau de neurones :
                    - couches d'augmentation de données actives : `RandomFlip`, `RandomBrightness`, `RandomContrast` et `GaussianNoise`
                    - normalisation des niveaux de gris par `Rescaling`
                    - 4 blocs convolutionels pour l'extraction de features composés chacun de : `Conv2D` avec activation ReLU, puis `MaxPooling2D` pour réduire la taille, et un `Dropout` afin de régulariser 
                    - passage en 1D : simple `Flatten`
                    - pour la classification : 2 couches `Dense` 
                    """)
    if modele_name == "MobileNet":
        st.markdown("""
                    Les meilleurs résultats de l'approche Transfer Learning ont été obtenus avec un fine-tuning du modèle pré-entraîné MobileNetV2 :
                    - couches d'augmentation de données actives : `RandomFlip`, `RandomBrightness`, `RandomContrast` et `GaussianNoise`
                    - couches d'extraction de features : backbone MobileNet, avec un dégel des poids à partir du 5ème bloc
                    - passage en 1D par `GlobalAveragePooling2D`
                    - pour la classification : une couche `Dense` puis une régularisation `Dropout`, et une dernière couche `Dense`
                    """)

    st.subheader(f"Performances du modèle {modele_name}")

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

    # Choix par l'utilisateur entre rapport de classification et matrice de confusion
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
    
    st.subheader(f"Interprétabilité du modèle {modele_name}")

    # Description de l'interprétabilité du modèle
    if modele_name == "SVM":
        st.markdown("""
                    Le classifieur SVM, qui repose sur un noyau non linéaire et précédé d'une PCA, n'est pas facilement interprétable.
                    Nous avons utilisé **LIME** qui reste une méthode approximative et locale.

                    Dans cet exemple, on note que le modèle a bien fait le focus sur la bande de salissure située en haut pour classifier l'image Bottom Dirt. 
                    """)
        st.image(load_image("resources/interpretabilite_svm.png"),caption="Interprétabilité LIME sur une image Bottom Dirt")
    if modele_name == "XGBoost":
        st.markdown("""
                    Nous pouvons faire ici une interprétabilité assez directe, à la fois globale et locale, à l'aide de :
                    - la simplicité relative des features et de la pipeline
                    - l'**importance des features** fournie intrinsèquement par XGBoost
                    - l'utilisation de **SHAP**

                    Par exemple avec SHAP, nous pouvons voir les caractéristiques les plus influentes pour la classification :
                    - la valeur max des pixels de l'image. Notamment pour les classes healthy et break (qui doit être peu élevée pour la première, très élevée pour la seconde)
                    - la densité de contours
                    - le degré de dissymétrie de la distribution des niveaux de gris
                    - les propriétés de texture en général
                    """)
        st.image(load_image("resources/interpretabilite_xgboost.png"),caption="Interprétabilité SHAP globale")
    if modele_name == "CNN Perso":
        st.markdown("""
                    Nous avons appliqué la technique de **Grad-CAM** sur les couches de convolution de ce CNN pour visualiser les zones des images les plus déterminantes dans la décision.
                    D'après les exemples d'images étudiés, le CNN a bien appris à repérer les zones chaudes, les patterns de salissure en bas, etc., concordant avec l'expertise métier.
                    Nous avons également utilisé **SHAP** en complément, qui a confirmé la cohérence des principales zones observées par le modèle pour la prédiction des défauts.

                    Dans l'exemple ci-dessous, une image prédite Short circuit panel, la Grad-CAM met en évidence les cellules avec des grosses variations de température dès la seconde couche de convolution.
                    """)
        st.image(load_image("resources/interpretabilite_cnn.png"),caption="Grad-CAM appliqué aux 4 couches de convolution sur une image Short circuit panel")
    if modele_name == "MobileNet":
        st.markdown("""
                    Nous avons appliqué **SHAP** sur les images des différentes catégories pour comprendre sur quelles régions notre modèle se focalise pour telle ou telle prédiction.
                    Sur quelques images test, SHAP a pu mettre en évidence les zones chaudes ou anormales du panneau comme ayant les valeurs SHAP les plus élevées pour prédire la classe correspondante.
                    En d'autres termes, le modèle MobileNet fine-tuné utilise bien les hot spots ou motifs de panne attendus.

                    Par exemple sur l'image Hot Cell ci-dessous, on constate que les cellules avec une forte variation locale de température à droite ont bien été repérées.
                    """)
        st.image(load_image("resources/interpretabilite_mobilenet.png"),caption="Interprétabilité SHAP sur une image Hot Cell")
