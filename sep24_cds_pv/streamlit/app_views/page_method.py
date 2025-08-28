import streamlit as st

from .utils import load_image


def show_method():
    st.header("Méthode",divider="gray")

    st.subheader(f"Machine Learning et Deep Learning")

    # Description des approches
    st.markdown("""
    Quatre approches ont été expérimentées :

    - **Approche Machine Learning classique** exploitant des **features *localisés*** (ex : pixels bruts...)  
    - **Approche Machine Learning classique** exploitant uniquement des **features *non localisés*** (ex : contraste...)  
    - **Approche Deep Learning** exploitant les images avec des réseaux de neurones convolutionnels *maison*  
    - **Approche Transfer Learning** exploitant les images avec des modèles pré-entraînés  
    """)

    # Description des features localisés
    with st.expander("Pour en savoir plus sur les features localisés"):
        st.markdown("""
                    Ces descripteurs conservent une information de position ou de structure dans l'image. Nous avons extrait pour chaque image :
                    - Le **vecteur de pixels bruts** : les images sont d'abord redimensionnées (de 60x110 à 30x55 pixels par exemple), puis leurs pixels sont linéarisés sous forme de vecteur 1D. Chaque pixel, codant une valeur de température (niveau de gris), est alors une feature.
                    - Le **descripteur HOG** ([Histogram of Oriented Gradients](https://towardsdatascience.com/histogram-of-oriented-gradients-hog-in-computer-vision-a2ec66f6e671/?source=rss----7f60cf5620c9---4)) :
                    on découpe l'image en cellules et on y calcule des histogrammes d'orientations de gradient, puis on normalise ces histogrammes. Le vecteur HOG résultant capture les formes et structures présentes dans l'image.
                    """)
        st.image(load_image("resources/features_hog.png"),caption = "Exemples de descripteurs HOG (représentation 2D) pour quelques images")

    # Description des features non localisés
    with st.expander("Pour en savoir plus sur les features non localisés"):
        st.markdown("""
                    Ces descripteurs sont calculés sur l'image entière, sans considération explicite de la position spatiale. Nous avons extrait pour chaque image :
                    - des **statistiques sur les intensités** (niveaux de gris de l'image) : moyenne, médiane, minimum, maximum, écart-type, quantiles (p5, p10, …, p95) et histogramme sur 256 bins (0 à 255).
                    - des **propriétés extraites de la matrice GLCM** : contraste, énergie, homogénéité, corrélation
                    - des **statistiques sur la carte d'entropie** : moyenne, écart-type, histogramme d'entropie, etc. pour caractériser la complexité de l'image.
                    - la **densité de contours** détectés dans l'image : pourcentage calculé après application d'un filtre de Canny.
                    - des **statistiques sur les “hot spots”** : ce sont des régions anormalement chaudes dans le panneau. Nous utilisons un seuillage adaptatif pour les détecter et nous en extrayons des statistiques...
                    """)
        st.image(load_image("resources/features_entropy_canny.png"),caption = "Exemples de cartes d'entropie et contours détectés pour quelques images")
    
    st.subheader(f"A la recherche du meilleur modèle...")

    # Un onglet Méthodologie ML / Un onglet Méthodologie DL
    tab1, tab2 = st.tabs(["Approches Machine Learning", "Approches Deep Learning"])
    with tab1:
        st.markdown("""
        Nous avons entraîné plusieurs **classifieurs** :
        - basés sur les distances (SVM...)
        - basés sur les arbres de décision (Random Forest, XGBoost, LightGBM...)
        - également un réseau de neurones dense
                    
        Nous avons mis en oeuvre une **validation croisée** avec grille de paramètres pour optimiser notre pipeline :
        - détermination des **meilleures étapes de prétraitement** : extraction de features, mise à l'échelle, réduction de dimensions, rééchantillonnage…  
        - optimisation des **paramètres internes des étapes** (ex. extraction de features : taille des cellules HOG)  
        - optimisation des **hyperparamètres des classifieurs** (ex. SVM : paramètre de régularisation `C`)  
        """)
    with tab2:
        st.markdown("""
        Nous avons testé un réseau de neurones convolutif *maison*, et 2 modèles préentraînés sur ImageNet : MobileNetV2 et EfficientNetV2B2.
        Pour l'**entraînement**, nous avons utilisé :
        - la fonction de perte `sparse_categorical_crossentropy` adaptée aux problèmes multi-classes
        - l'optimiseur Adam pour la descente de gradient.
        - des callbacks pour éviter le sur-apprentissage (`EarlyStopping` et `ReduceLROnPlateau` en cas de plafonnement des performances).
                    
        Nous avons mis en oeuvre une **validation fixe** avec grille de paramètres pour optimiser notre réseau :
        - activation ou non de **couches d'augmentation de données** (ex : `RandomFlip`, `RandomContrast`)  
        - recherche d'une configuration optimale pour les **couches d'extraction de features** (ex : nb de filtres de convolution par couche)  
        - recherche d'une configuration optimale pour les **couches de classification** (ex : nb de couches `Dense`)  
        - optimisation de la **profondeur du dégel** dans le cas du Transfer Learning  
        """)
    
    # Description des métriques
    st.markdown("""
    Voici les principales **métriques** observées : 
    - le F1-score macro (moyenne sur les 10 classes) : bon compromis pour évaluer les performances globales dans un jeu déséquilibré.
    - l'accuracy
    - les temps d'entraînement et de prédiction
    - la précision et le rappel de la classe *healthy* : capacité à distinguer les panneaux sains des panneaux défectueux  
    """)
