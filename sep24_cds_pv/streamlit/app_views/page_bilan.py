import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def show_bilan(modeles,y_test):

    st.header("Bilan",divider="gray")

    # Création des tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Meilleur modèle", "Conclusions", "Critiques", "Perspectives"])
    
    # Tab 1 : Meilleur modèle
    with tab1:

        st.subheader(":red[MobileNetV2] (CNN pré-entraîné) :red[+ Transfer Learning]")
        st.markdown("""
            - **Meilleures performances** pour les méthodes **:red[Deep Learning]** basées sur des réseaux de neurones convolutionnels (CNN - Convolutional Neural Network)
            - **:red[MobileNetV2]** fine-tuné par **:red[Transfer Learning]** atteint un **F1-macro de :red[91.3%]**
        """)

        # L'utilisateur choisit un modèle de référence
        modele_ref_name = st.selectbox("Référence de comparaison", list(modeles.keys()), index=3)

        # Récupération des prédictions du modèle de référence sur le jeu de test
        y_pred_ref = modeles[modele_ref_name]["predicted_data_test"]
        accu_ref = accuracy_score(y_test,y_pred_ref)
        f1_ref = f1_score(y_test,y_pred_ref,average="macro")
        prec_healthy_ref = precision_score(y_test, y_pred_ref, labels=["healthy panel"],average=None)[0]
        recall_healthy_ref = recall_score(y_test, y_pred_ref, labels=["healthy panel"],average=None)[0]

        # Nombre de colonnes
        nb_cols = 7

        # Ligne d'en-têtes
        metrics = [
            "Accuracy", 
            "F1 macro", 
            "Précision Healthy", 
            "Rappel Healthy",
        ]
        definitions = [
            "Pourcentage de bonnes prédictions, toutes classes confondues",
            "Moyenne arithmétique simple des F1-scores de toutes les classes",
            "Pourcentage de prédictions 'healthy' correctes parmi toutes les prédictions 'healthy' \n\n👉 Précision 'healthy' faible -> on manque des défauts",
            "Pourcentage de vrais 'healthy' détectés parmi tous les individus réellement 'healthy' \n\n👉 Rappel 'healthy' faible -> on détecte des défauts à tort",
        ]
        cols = st.columns(nb_cols)
        for idx, col in enumerate(cols[2:-1]):
            with col:
                st.markdown(f":gray[{metrics[idx]}]", help=definitions[idx])

        # Pour chaque modèle
        metric_height = "stretch"
        for modele_name in modeles:
            # Récupération des prédictions du modèle sur le jeu de test
            y_pred = modeles[modele_name]["predicted_data_test"]
            # Affichage côte à côte des métriques principales et de leur différence avec les métriques de référence
            _, col1, col2, col3, col4, col5, _ = st.columns(nb_cols)
            with col1:
                st.markdown(f"**{modele_name}**")
            with col2:
                accu = accuracy_score(y_test,y_pred)
                st.metric("Accuracy", 
                          f"{accu*100:.1f} %", 
                          f"{(accu-accu_ref)*100:.1f} %" if modele_name!=modele_ref_name else None, 
                          height=metric_height,
                          label_visibility="collapsed")
            with col3:
                f1 = f1_score(y_test,y_pred,average="macro")
                st.metric("F1 macro", 
                          f"{f1*100:.1f} %", 
                          f"{(f1-f1_ref)*100:.1f} %" if modele_name!=modele_ref_name else None,
                          height=metric_height,
                          label_visibility="collapsed")
            with col4:
                prec_healthy = precision_score(y_test, y_pred, labels=["healthy panel"],average=None)[0]
                st.metric("Précision Healthy", 
                          f"{prec_healthy*100:.1f} %", 
                          f"{(prec_healthy - prec_healthy_ref)*100:.1f} %" if modele_name!=modele_ref_name else None, 
                          height=metric_height,
                          label_visibility="collapsed")
            with col5:
                recall_healthy = recall_score(y_test, y_pred, labels=["healthy panel"],average=None)[0]
                st.metric("Rappel Healthy", 
                          f"{recall_healthy*100:.1f} %", 
                          f"{(recall_healthy - recall_healthy_ref)*100:.1f} %" if modele_name!=modele_ref_name else None, 
                          height=metric_height,
                          label_visibility="collapsed")
    
        with st.expander("Rappel de la définition des différentes métriques"):

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Accuracy")
                st.markdown("""
                    Pourcentage de bonnes prédictions faites par le modèle, toutes classes confondues.
                """)
                st.latex(r"Accuracy = \frac{VP + VN}{VP + VN + FP + FN}")

            with col2:
                with st.columns([1,2,1])[1]:
                    st.text("")
                    st.text("")
                    st.markdown("""
                        - **VP** : Vrais positifs
                        - **VN** : Vrais Négatifs
                        - **FP** : Faux positifs
                        - **FN** : Faux négatifs
                    """)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Précision")
                st.markdown("""
                    Pourcentage de prédictions positives correctes parmi toutes les prédictions positives faites par le modèle.
                """)
                st.latex(r"Précision = \frac{VP}{VP + FP}")
                st.markdown("👉 Parmi tous les individus identifiés comme positifs, combien sont vraiment positifs ?")

            with col2:
                st.subheader("Rappel")
                st.markdown("""
                    Pourcentage de vrais positifs détectés parmi tous les individus réellement positifs.
                """)
                st.latex(r"Rappel = \frac{VP}{VP + FN}")
                st.markdown("👉 Parmi tous les individus réellement positifs, quelle proportion le modèle a-t-il correctement détectés ?")


    # Tab 2 : Conclusions
    with tab2:

        st.subheader("Conclusions sur le projet")
        st.markdown("""
            - Confirmation : **Deep Learning > Machine Learning** : CNN plus efficaces ✅
            - **Objectif initial :red[dépassé]** : équipe PVF-10 :red[battue] ! 🎯
                - **Equipe PVF-10** -> CoatNet (20.2M paramètres) : Accuracy **93.3%** - F1 macro **88.7%**
                - **Equipe Datascientest** -> MobileNetV2 (3.5M paramètres) : Accuracy **:red[94.2%]** - F1 macro **:red[91.3%]**
            - **Avantage DL**
                - 🔎 **:red[Auto-sélection des features]** -> **le modèle "choisit" lui-même** les **features les plus pertinents :red[pendant l'apprentissage]**
            - **Avantage ML** : 
                - 🧩 **:red[Meilleure interprétabilité]** (dépend en réalité du **modèle utilisé**)
                - 🪶 **:red[Modèles plus légers]** -> ⏱️ **gain de temps** (apprentissage comme prédiction - **mais l'extraction des features peut être long**)
            - Choix **ML vs DL** : **compromis** entre
                - 📈 **:red[Performance de prédiction]** (accuracy, précision, rappel)
                - 🖥️ **:red[Contraintes de déploiement]** (moyens de calcul limités - temps réel embarqué par exemple)
        """)

#         """)

    # Tab 3 : Critiques
    with tab3:

        st.subheader("Regard critique sur notre travail")
        st.markdown("""
            - **Chaîne d'acquisition :red[non maîtrisée]**
                - 🔐 Conversion images infrarouge -> niveau de gris :red[non documentée]
                - 🚫 :red[Difficulté de réutilisation] de notre modèle sur **autres jeux de données** ou **images brutes**
                - ✅ **Méthodologie appliquée :red[reste pertinente] !**
            - **Déséquilibre entre classes -> :red[impact sur les performances]** (notamment des modèles ML)
                - 📉 **Gain limité** des approches testées de :red[sur-échantillonnage] ou d':red[augmentation de données]
                - 🚀 **Potentiel d'amélioration** possible par exploration d'approches plus élaborées
            - **Marge de progression des modèles ML** -> :red[exploration de features complémentaires]
                - 🎯 **Ciblage** de certaines zones des images (Caractéristiques [GLCM](https://en.wikipedia.org/wiki/Co-occurrence_matrix) ou **entropie** 
                ou **indicateurs statistiques :red[localisés]**)
                - 🌀 Autres descripteurs de **texture** ([Local Binary Patterns](https://en.wikipedia.org/wiki/Local_binary_patterns) par exemple) 
                ou **features de forme** des hot spots de l'image
                - **:red[Mais]** rapport **effort / gain en performance** :red[défavorable] pour les modèles **ML vs DL** ⚠️
                    - Tout **nouveau feature** doit être testé, tuné, validé, ... -> :red[opérations chronophages]
                    - **Chaque ajout de feature** :red[complexifie] **le pipeline** de calcul et l'apprentissage
                    - Rappel : les **réseaux de neurones** :red[apprennent par eux-mêmes] **les features pertinents** pour le problème soumis !
        """)

    # Tab 4 : Perspectives
    with tab4:
        st.subheader("Perspectives - comment passer à l'industrialisation...")
        st.markdown("""
            - **Valider la :red[généralisation du modèle]**
                - 🧪 Tester d'**autres jeux de données**
            - **Compléter l':red[intégration opérationelle]**
                - ⚙️ **Maîtriser le pré-processing** (images thermiques → niveaux de gris)
                - 🎯 Ajouter un **modèle de détection** en amont (type [YOLO](https://en.wikipedia.org/wiki/You_Only_Look_Once)) -> :red[segmentation] des panneaux PV dans une image complète
            - **Maintenir une :red[veille technologique]** sur nouveaux modèles ou architectures DL
                - 🧠 **Architectures plus récentes** : Transformers
                - 🔀 **Hybrides** : Tranformers + CNN ou CNN + classifieurs ML
                - 🌡️ **Spécialisés** : modèles pré-entraînés sur bases d'images thermiques
            - **Améliorer l':red[interprétabilité]** des modèles DL mis en oeuvre -> développement d'un module d'explication
                - 🔎 Analyse **Grad-CAM** + logique basée sur **extraction de features** (type hot spots)
                - **Objectif** : 🤝 :red[Performance DL] + :red[explication métier] compréhensible pour un opérateur terrain
            - Certainement beaucoup d'autres choses...
        """)
