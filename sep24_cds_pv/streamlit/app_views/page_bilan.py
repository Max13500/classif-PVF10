import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def show_bilan(modeles,y_test):
    st.header("Bilan",divider="gray")

    st.subheader(f"Le Transfer Learning en tête")
    st.markdown("***TODO***")

    # L'utilisateur choisit un modèle de référence
    modele_ref_name = st.selectbox("Référence de comparaison",list(modeles.keys()))

    # Récupération des prédictions du modèle de référence sur le jeu de test
    y_pred_ref = modeles[modele_ref_name]["predicted_data_test"]
    accu_ref = accuracy_score(y_test,y_pred_ref)
    f1_ref = f1_score(y_test,y_pred_ref,average="macro")
    prec_healthy_ref = precision_score(y_test, y_pred_ref, labels=["healthy panel"],average=None)[0]
    recall_healthy_ref = recall_score(y_test, y_pred_ref, labels=["healthy panel"],average=None)[0]

    # Pour chaque modèle
    for modele_name in modeles:
        # Récupération des prédictions du modèle sur le jeu de test
        y_pred = modeles[modele_name]["predicted_data_test"]
        # Affichage côte à côte des métriques principales et de leur différence avec les métriques de référence
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f"**{modele_name} :**")
        with col2:
            accu = accuracy_score(y_test,y_pred)
            st.metric("Accuracy", f"{accu*100:.1f} %", f"{(accu-accu_ref)*100:.1f} %" if modele_name!=modele_ref_name else None)
        with col3:
            f1 = f1_score(y_test,y_pred,average="macro")
            st.metric("F1 macro", f"{f1*100:.1f} %", f"{(f1-f1_ref)*100:.1f} %" if modele_name!=modele_ref_name else None)
        with col4:
            prec_healthy = precision_score(y_test, y_pred, labels=["healthy panel"],average=None)[0]
            st.metric("Précision Healthy", f"{prec_healthy*100:.1f} %", f"{(prec_healthy - prec_healthy_ref)*100:.1f} %" if modele_name!=modele_ref_name else None)
        with col5:
            recall_healthy = recall_score(y_test, y_pred, labels=["healthy panel"],average=None)[0]
            st.metric("Rappel Healthy", f"{recall_healthy*100:.1f} %", f"{(recall_healthy - recall_healthy_ref)*100:.1f} %" if modele_name!=modele_ref_name else None)
    
    st.subheader("Conclusion")
    st.markdown("***TODO : conclusion métier, regard critique***")

    st.subheader("Perspectives")
    st.markdown("***TODO***")
