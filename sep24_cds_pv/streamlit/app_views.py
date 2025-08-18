import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,classification_report, confusion_matrix

@st.cache_data
def load_image(path):
    return Image.open(path)

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

def show_method():
    st.header("Méthodologie")

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
                    - Le **descripteur HOG** (Histogram of Oriented Gradients) : on découpe l'image en cellules et on y calcule des histogrammes d'orientations de gradient, puis on normalise ces histogrammes. Le vecteur HOG résultant capture les formes et structures présentes dans l'image.
                    """)
        st.image(load_image("resources/features_hog.png"),caption = "Exemples de descripteurs HOG (représentation 2D) pour quelques images")

    # Description des features non localisés
    with st.expander("Pour en savoir plus sur les features non localisés"):
        st.markdown("""
                    Ces descripteurs sont calculés sur l'image entière, sans considération explicite de la position spatiale. Nous avons extrait pour chaque image :
                    - des **statistiques sur les intensités** (niveaux de gris de l'image) : moyenne, médiane, minimum, maximum, écart-type, quantiles (p5, p10, …, p95) et histogramme sur 256 bins (0 à 255).
                    - des **propriétés extraites de la matrice GLCM** : la Grey Level Co-occurrence Matrix mesure la fréquence de co-occurrence de paires de niveaux de gris à une certaine distance et orientation.
                    On en extrait des propriétés qui quantifient la texture globale de l'image : contraste, énergie... 
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

def show_results(modeles,y_test):
    st.header("Résultats")

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


def show_demo(modeles,X_test,y_test):
    st.header("Démo")

    st.markdown("""
                Choisissez des images de test : vous pouvez afficher tous les types de défauts, ou vous concentrer sur un défaut en particulier.

                Remarque : ces images n'ont pas été utilisées lors de l'entraînement des modèles.
    """)

    with st.container(border=True):
        # Choix par l'utilisateur d'afficher une image de chaque classe, ou une classe en particulier
        all_classes = st.toggle("Tester tous les types de défauts",True)

        # 1er cas : on récupère une image de chaque classe au hasard (=> 10 en tout)
        if all_classes:
            nb_img = y_test.nunique()
            index_sel = y_test.groupby(y_test).apply(lambda x: x.sample(1)).index.get_level_values(1)
        # 2ème cas : l'utilisateur choisit la classe et le nb d'images de cette classe à afficher
        else:
            sel_cols = st.columns(2)
            with sel_cols[0]:
                nom_classe = st.selectbox("Choix du défaut",y_test.unique())
            with sel_cols[1]:
                nb_img = st.slider("Nombre d'images à afficher",1,10,5)
            # Sécurité : cas où nb d'éléments de la classe < nb demandé
            nb_img = min(nb_img,len(y_test[y_test==nom_classe]))
            # On récupère de manière aléatoire les images demandées
            index_sel = y_test.groupby(y_test[y_test==nom_classe]).apply(lambda x: x.sample(nb_img, replace=False)).index.get_level_values(1)
        
        # Bouton pour rafraichir => grâce au tirage aléatoire, on affichera d'autres images avec les paramètres sélectionnés
        if st.button("🔄 Changer d'images"):
            pass

    # On récupère les chemins et les labels des images sélectionnées
    sel_path = X_test.loc[index_sel,"Chemin"]
    sel_y_test = y_test[index_sel]
    
    st.markdown("""
    Pour chaque image, comparez les prédictions de nos modèles : les erreurs apparaissent en rouge.
    """)

    # Entêtes de la grille de comparaison
    with st.container(border=True):
        cols = st.columns(len(modeles)+2)
        headers = ["Image", "Défaut"] + [f"Prédiction {modele_name}" for modele_name in modeles]
        for c, h in zip(cols, headers):
            with c:
                st.html(f"<div style='text-align:center; font-weight:bold'>{h}</div>")

    # 1 ligne par image à prédire
    for i in range(nb_img):
        with st.container(border=True):
            cols = st.columns(len(modeles)+2) 
        
            # Colonne 0 : affiche de l'image
            # Artifice pour centrer horizontalement
            with cols[0]:
                left, mid, right = st.columns([1, 3, 1])
                with mid:
                    st.image(sel_path.iloc[i])
            
            # Col 1 : affichage du défaut réel
            with cols[1]:
                st.html(f"<div style='text-align:center; border:1px solid #eee; padding:2px';><b>{sel_y_test.iloc[i]}</b></div>")

            # Col 2...n : affichage du défaut prédit pour chaque modèle
            for j, model_name in enumerate(modeles):
                pred = modeles[model_name]["predicted_data_test"][index_sel][i]
                color = "green" if pred == sel_y_test.iloc[i] else "red"
                with cols[j+2]:
                    st.html(f"<div style='text-align:center; border:1px solid #eee; padding:2px; color:{color}'><b>{pred}</b></div>")

def show_bilan():
    st.header("Bilan")
    st.write("TODO : conclusion sur meilleur modèle, conclusion métier, critique, perspectives")

