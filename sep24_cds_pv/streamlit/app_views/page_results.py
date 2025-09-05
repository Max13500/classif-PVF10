import streamlit as st
import pandas as pd
import numpy as np
import io
import contextlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

from .utils import load_image

# Fonction pour affichage du model.summary()
def get_model_summary(model):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        model.summary()
    summary_str = stream.getvalue()
    return summary_str

def show_results(modeles,y_test):
    st.header("Résultats",divider="gray")

    st.markdown("""
    Voici les **:red[meilleurs modèles]** obtenus après optimisation, pour les 4 approches mentionnées :
    - **Machine Learning avec features localisés** : classifieur **SVM**
    - **Machine Learning avec features non localisés** : classifieur **XGBoost**
    - **Deep Learning *from scratch*** : notre **CNN perso**
    - **Transfer Learning** : modèle **MobileNetV2** fine-tuné                
    """)

    # Choix du modèle par l'utilisateur  
    modele_name = st.selectbox(
        "Sélectionnez une approche / un modèle",
        list(modeles.keys()),
        format_func=lambda name: f"{modeles[name]['methodo_name']} : modèle {name}"
    )

    st.subheader(f"Architecture du modèle {modele_name}")

    # Description de l'architecture du modèle
    if modele_name == "SVM":
        with st.columns([0.1,0.8,0.1])[1]:
            st.image(load_image("resources/resultats/architecture_svm.png"))
    if modele_name == "XGBoost":
        with st.columns([0.3,0.4,0.3])[1]:
            st.image(load_image("resources/resultats/architecture_xgboost.png"))
    if modele_name == "CNN Perso":
        with st.columns([0.1,0.8,0.1])[1]:
            st.image(load_image("resources/resultats/architecture_cnn.png"))
        with st.expander("Pour visualiser l'architecture détaillée du réseau..."):
            # Récupérer et afficher le résumé dans Streamlit
            summary_str = get_model_summary(modeles[modele_name]["trained_model"])
            st.code(summary_str, language="text")
    if modele_name == "MobileNet":
        with st.columns([0.05,0.9,0.05])[1]:
            st.image(load_image("resources/resultats/architecture_mobileNet.png"))
        with st.expander("Pour visualiser l'architecture détaillée du réseau..."):
            # Récupérer et afficher le résumé dans Streamlit
            summary_str = get_model_summary(modeles[modele_name]["trained_model"])
            st.code(summary_str, language="text")

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
    tab1, tab2 = st.tabs(["Rapport de classification", "Matrice de confusion"])

    # Affichage rapport de classification
    with tab1:
        with st.columns([0.25,0.5,0.25])[1]:
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()
            # Métriques par classe
            st.table(report_df.iloc[:-3,:].style.format(precision=2))
            # Métriques globales
            st.table(report_df.iloc[-2:,:].style.format(precision=2))

    # Affichage matrice de confusion
    with tab2:
        with st.columns([0.25,0.5,0.25])[1]:
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
                    **Interprétabilité avec :red[LIME]** 
                    - Interprétation **cohérente** sur les quelques exemples étudiés
                    - **Complexité** de la pipeline => difficultés d'interprétabilité
                    - Méthode **approximative et locale**.
                    """)
        with st.columns([0.1,0.8,0.1])[1]:
            st.image(load_image("resources/resultats/interpretabilite_svm.png"),caption="Interprétabilité LIME sur une image Bottom Dirt")
    if modele_name == "XGBoost":
        st.markdown("""
                    **Interprétabilité avec :red[l'importance des features] et :red[SHAP]** 
                    - Interprétabilité **globale** et **locale**. Résultats cohérents.
                    - **Simplicité** de la pipeline => explication du modèle assez directe
                    """)
        with st.columns([0.2,0.6,0.2])[1]:
            st.image(load_image("resources/resultats/interpretabilite_xgboost.png"),caption="Interprétabilité SHAP globale")
    if modele_name == "CNN Perso":
        st.markdown("""
                    **Interprétabilité avec :red[Grad-CAM] et :red[SHAP]** 
                    - Interprétabilité directement sur les **images**
                    - Repérage des zones en défaut **conforme** à l'expertise métier
                    """)
        with st.columns([0.3,0.4,0.3])[1]:
            with st.container(border=True):
                st.image(load_image("resources/resultats/interpretabilite_cnn.png"),caption="Interprétabilité Grad-CAM sur une image Short circuit panel")
            with st.container(border=True):
                st.image(load_image("resources/resultats/interpretabilite_cnn_2.png"),caption="Interprétabilité Grad-CAM sur une image Shadow")
    if modele_name == "MobileNet":
        st.markdown("""
                    **Interprétabilité avec :red[SHAP]** 
                    - Interprétabilité directement sur les **images**
                    - Repérage des zones en défaut **conforme** à l'expertise métier
                    """)
        with st.columns([0.2,0.6,0.2])[1]:
            with st.container(border=True):
                st.image(load_image("resources/resultats/interpretabilite_mobilenet.png"),caption="Interprétabilité SHAP sur une image Hot Cell")
            with st.container(border=True):
                st.image(load_image("resources/resultats/interpretabilite_mobilenet_2.png"),caption="Interprétabilité SHAP sur une image Substring open circuit")
