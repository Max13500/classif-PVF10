import streamlit as st

from .utils import load_image


def show_method():
    st.header("Méthode",divider="gray")

    st.subheader(f"Machine Learning et Deep Learning")

    # Description des approches
    st.markdown("""
    **:red[Quatre approches]** ont été expérimentées : 
    """)

    col1, col2, col3 = st.columns([0.45,0.1,0.45])

    with col1:
        st.markdown("**1. Extraction de features localisés + Machine Learning**")
        st.image(
            "resources/methode/ML_localise.png",
        )
        st.markdown("**3. Deep Learning : CNN *from scratch***")
        st.image(
            "resources/methode/DL_perso.png",
        )
        
    with col3:
        st.markdown("**2. Extraction de features non localisés + Machine Learning**")
        st.image(
            "resources/methode/ML_non_localise.png",
        )
        st.markdown("**4. Deep Learning : Transfer Learning**")
        st.image(
            "resources/methode/DL_transfer_learning.png",
        )
        
    # Description des features localisés
    with st.expander("Pour en savoir plus sur les **features localisés**"):
        st.markdown("""
                    Ils conservent une **:red[information de position ou de structure]** dans l'image :
                    - Le **vecteur de pixels bruts** : après redimensionnement de l'image, les pixels sont linéarisés sous forme de vecteur 1D.
                    - Le **descripteur HOG** ([Histogram of Oriented Gradients](https://towardsdatascience.com/histogram-of-oriented-gradients-hog-in-computer-vision-a2ec66f6e671/?source=rss----7f60cf5620c9---4)) :
                    on découpe l'image en cellules et on y calcule des histogrammes d'orientations de gradient, puis on normalise ces histogrammes => Le vecteur HOG capture les formes et structures présentes dans l'image.
                    """)
        with st.columns([0.3,0.4,0.3])[1]:
            st.image(load_image("resources/methode/features_hog.png"),caption = "Exemples de descripteurs HOG (représentation 2D) pour quelques images")

    # Description des features non localisés
    with st.expander("Pour en savoir plus sur les **features non localisés**"):
        st.markdown("""
                    Ils sont **:red[calculés sur l'image entière]**, sans considération explicite de la position spatiale :
                    - **statistiques sur les intensités** : moyenne, médiane, minimum, maximum, écart-type, quantiles et histogramme des niveaux de gris.
                    - **propriétés extraites de la matrice GLCM** : contraste, énergie, homogénéité, corrélation.
                    - **statistiques sur la carte d'entropie** : moyenne, écart-type, histogramme d'entropie, etc. pour caractériser la complexité de l'image.
                    - **densité de contours** détectés dans l'image : pourcentage calculé après application d'un filtre de Canny.
                    - **statistiques sur les “hot spots”** : détection et extraction de caractéristiques de régions anormalement chaudes dans le panneau.
                    """)
        with st.columns([0.3,0.4,0.3])[1]:
            st.image(load_image("resources/methode/features_entropy_canny.png"),caption = "Exemples de cartes d'entropie et contours détectés pour quelques images")
    
    st.subheader(f"A la recherche du meilleur modèle...")

    # Un onglet Méthodologie ML / Un onglet Méthodologie DL
    tab1, tab2 = st.tabs(["Approches Machine Learning", "Approches Deep Learning"])
    with tab1:
        col1,col2,col3 = st.columns([0.6,0.05,0.35])
        with col1:
            # Description validation croisée
            st.markdown("""                       
            Mise en oeuvre d'une **:red[validation croisée]** avec grille de paramètres pour **optimiser notre pipeline** :
            - détermination des **meilleures étapes de prétraitement** :
                - extraction de features
                - mise à l'échelle
                - réduction de dimensions
                - rééchantillonnage
            - optimisation des **paramètres internes de ces étapes** (ex : nb de composantes de la PCA)
            - détermination des **meilleurs classifieurs**
            - optimisation des **hyperparamètres des classifieurs** (ex : paramètre de régularisation `C` pour SVM)  
            """)
            # Description des métriques
            st.markdown("""
            Principales **:red[métriques]** observées : 
            - le **F1-score macro** (moyenne sur les 10 classes) => bon compromis dans un jeu déséquilibré
            - l'**accuracy**
            - la **précision et le rappel de la classe *healthy*** => capacité à distinguer les panneaux sains vs défectueux 
            - les **temps de calcul** : entraînement et prédiction
            """)
        # Image des classifieurs
        with col3:
            st.image(load_image("resources/methode/classifieurs_ML.png"))
    with tab2:
        col1,col2,col3 = st.columns([0.65,0.05,0.3])
        with col1:
            # Description validation fixe
            st.markdown("""
            Mise en oeuvre d'une **:red[validation fixe]** avec grille de paramètres pour **optimiser notre réseau** :
            - activation ou non de **couches d'augmentation de données** (ex : `RandomFlip`, `RandomContrast`)  
            - optimisation des **couches d'extraction de features**
                - Pour le CNN from scratch : configuration des couches de convolution (ex : nb de filtres de convolution)
                - Pour le Transfer Learning : optimisation de la profondeur du dégel lors du fine-tuning
            - optimisation des **couches de classification** (ex : nb de couches `Dense`)   
            """)
            # Description des métriques
            st.markdown("""
            Principales **:red[métriques]** observées : 
            - le **F1-score macro** (moyenne sur les 10 classes) => bon compromis dans un jeu déséquilibré
            - l'**accuracy**
            - la **précision et le rappel de la classe *healthy*** => capacité à distinguer les panneaux sains vs défectueux 
            - les **temps de calcul** : entraînement et prédiction
            """)
        # Image des modèles
        with col3:
            st.image(load_image("resources/methode/modeles_DL.png"))
    
    
