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
    # Titre
    st.html(
        """
        <div style="
            border: 2px solid #1ABC9C;   /* bordure turquoise */
            border-radius: 12px;         /* angles arrondis */
            padding: 20px;               /* espace autour du texte */
            text-align: center;          /* texte centré */
            box-shadow: 2px 2px 12px rgba(0,0,0,0.1);  /* légère ombre */
        ">
            <h1 style="margin:0;font-size:48px">Classification de défauts dans les panneaux photovoltaïques</h1>
        </div>
        """)
    
    st.subheader("Le contexte")
    st.markdown("***TODO***")

    st.subheader("Les objectifs")
    st.markdown("***TODO***")

    st.subheader("Les données")
    st.markdown("***TODO***")
    
    # On récupère au hasard une ligne du dataframe par classe
    df_sel = df.groupby("Classe").apply(lambda x: x.sample(1))

    st.markdown("""
    Voici un **premier aperçu** des images du dataset (une image par classe) :
    """)

    # Affichage de 10 images : 5 images sur 2 lignes, avec leur classe en titre
    with st.container(border=True):
        for i in range(0, 10, 5):
            cols = st.columns(5)
            for j, col in enumerate(cols):
                if i + j < len(df_sel):
                    col.image(df_sel["Chemin"].iloc[i + j], caption = df_sel["Classe"].iloc[i + j],use_container_width=True)
    
    # Bouton pour rafraichir => grâce au tirage aléatoire, on affichera d'autres images avec les paramètres sélectionnés
    if st.button("🔄 Changer d'images"):
        pass

def show_dataviz(df,statistiques):
    st.header("DataViz",divider="gray")

    st.subheader("Analyse générale du dataset")

    # Description du dataset et nettoyage
    st.markdown("""
    Notre étude porte sur **5579 images de type PNG au format 110x60 réparties selon 10 classes de défauts**.
    Nous avons procédé à une analyse globale et un **nettoyage** de ce jeu de données :
    - suppression de 7 doublons d'images    
    - annotation de 4% des images qui ont une dimension réelle différente de 110x60 (images d'origine carrées => potentiellement mauvais découpage ou resizing)
    - renommage des 10 classes pour plus de lisibilité
    """)

    # Affichage du dataframe des métadonnées
    with st.expander("Pour visualiser le dataframe final contenant les métadonnées..."):
        st.dataframe(df)
    
    st.subheader("Equilibre des classes")

    # Description de l'équilibre des classes
    st.markdown("""
    Notre jeu de données est réparti suivant dix catégories : neuf types de défauts différents et une classe représentant les panneaux sains (*healthy panel*).
    La répartition des classes est **légèrement déséquilibrée** :
    - la classe des panneaux sains représente un peu plus d'un quart des observations
    - les neuf classes de défauts se partagent le reste de manière relativement équitable
    - deux catégories sont toutefois en retrait : Break et String short circuit.
    """)

    # Diagramme de répartition des classes
    col1, col2, col3 = st.columns([0.1, 0.8, 0.1]) 
    with col2:     
        fig = plt.figure()
        sns.countplot(y = df['Classe'],hue = df['Classe'],legend=False)
        plt.title("Répartition des classes de défauts",fontsize=12, fontweight='bold')
        plt.xlabel("Nombre d'images")
        plt.ylabel("Classe de défaut")
        sns.despine()
        st.pyplot(fig)

    st.subheader("Les pseudo-couleurs")

    # Description des canaux RGB
    st.markdown("""
    L'analyse des canaux RGB a montré une **composante rouge très élevée**, et une composante bleue faible, quel que soit le type de défaut observé.
                
    Pour vous en rendre compte, visualisez la répartition des intensités dans les 3 canaux Rouge, Vert et Bleu pour l'ensemble des classes :
    """)    

    # Distribution des intensités moyennes des canaux RGB
    # Afficher 2 classes en parallèle
    rgb_cols = st.columns(2)
    for i,c in enumerate(rgb_cols):
        with c:
            fig = plt.figure()
            # L'utilisateur choisit la classe
            nom_classe = st.selectbox("Classe de défaut :" if i==0 else "Comparer avec :",df["Classe"].unique(),i,key=f"classe_rgb_{i}")
            # Récupération des intensités moyennes sur les 3 canaux R/G/B
            mean_colors = statistiques["Moyenne des canaux RGB"][nom_classe]
            # Création du violinplot correspondant
            parts = plt.violinplot(np.array(mean_colors),showmedians=True)
            plt.ylim([0,255])
            plt.title(f"Distribution des canaux RVB",fontsize=14, fontweight='bold')
            plt.xticks([1, 2, 3],labels=["Rouge","Vert","Bleu"],fontsize=14)
            plt.ylabel("Intensités moy (0-255)",fontsize=14)
            # Changer la couleur de chaque violon
            colors = ["red","green","blue"]
            for j, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[j])
                pc.set_edgecolor('black')
            st.pyplot(fig)
    
    # Fin description des canaux RGB
    st.markdown("""            
    Les images thermiques infra-rouges sont en **fausses couleurs** (ou pseudo-couleurs) :
    chaque pixel encode en réalité une valeur de température, et une palette de couleur adaptée (du type “inferno”) est utilisée pour améliorer la perception à l'oeil humain des variations de température.
                
    Nous avons donc fait le choix de travailler sur les **images converties en niveaux de gris**.
    """)

    st.subheader("Les niveaux de gris")

    # Description des niveaux de gris
    st.markdown("""
    L'analyse de **la distribution des niveaux de gris a montré des spécificités** selon le type de défauts.

    Vous pouvez observer pour chaque classe les histogrammes de 5 **indicateurs statistiques** des niveaux de gris :
    """)    

    # L'utilisateur choisit l'indicateur statistique
    indicateur = st.selectbox("Indicateur statistique",list(statistiques.keys())[1:6]) 
    # Afficher 2 classes en parallèle
    ndg_cols = st.columns(2)
    for i,c in enumerate(ndg_cols):
        with c:
            fig = plt.figure()
            # L'utilisateur choisit la classe
            nom_classe = st.selectbox("Classe de défaut :" if i==0 else "Comparer avec :",df["Classe"].unique(),i,key=f"classe_ndg_{i}")
            # Récupération de l'indicateur statistique demandé sur les NDG
            statistique = statistiques[indicateur][nom_classe]
            # Afficher l'histogramme et la densité de probabilité de l'indicateur
            sns.histplot(statistique,bins=20,stat="density",kde=True,alpha=0.6)
            plt.xlabel(indicateur,fontsize=14)
            plt.ylabel("Densité de probabilité",fontsize=14)
            plt.title(f"Histogramme (avec densité KDE)",fontsize=14, fontweight='bold')
            st.pyplot(fig)

    # Fin description des niveaux de gris
    st.markdown("""
    Les différences sont en général plus marquées sur les indicateurs Max et Ecart-type.
    
    Nous avons complété cette visualisation par des **tests statistiques** (Kruskal-Wallis + test post-hoc de Dunn-Bonferroni) qui ont montré que des classes sont significativement différentes l'une de l'autre selon les indicateurs observés.
    """)    

    st.subheader("Les textures")

    # Description entropie et densité de contours
    st.markdown("""
    L'analyse de caractéristiques avancées extraites des images a permis d'approfondir cette étude :
    - la **densité de contours** : proportion de contours dans l'image après application du filtre de Canny, indiquant des transitions abruptes.
    - l'**entropie** : quantifie la diversité ou le désordre des niveaux de gris. Une entropie élevée traduit une texture complexe.

    Observez pour chaque classe les histogrammes de ces propriétés :
    """)

    # Propriétés de texture
    # L'utilisateur choisit la propriété de texture
    propriete = st.selectbox("Propriété texturale",list(statistiques.keys())[6:8]) 
    # Afficher 2 classes en parallèle
    prop_cols = st.columns(2)
    for i,c in enumerate(prop_cols):
        with c:
            fig = plt.figure()
            # L'utilisateur choisit la classe
            nom_classe = st.selectbox("Classe de défaut :" if i==0 else "Comparer avec :",df["Classe"].unique(),i,key=f"classe_prop_{i}")
            # Récupération de la propriété demandée
            statistique = statistiques[propriete][nom_classe]
            # Afficher l'histogramme et la densité de probabilité de la propriété
            sns.histplot(statistique,bins=20,stat="density",kde=True,alpha=0.6)
            plt.xlabel(propriete,fontsize=14)
            plt.ylabel("Densité de probabilité",fontsize=14)
            plt.title(f"Histogramme (avec densité KDE)",fontsize=14, fontweight='bold')
            st.pyplot(fig)
    
    # Description propriétés GLCM
    st.markdown("""
    Nous avons également calculé la matrice [GLCM](https://en.wikipedia.org/wiki/Co-occurrence_matrix) de chaque image.
    La Gray Level Co-occurrence Matrix mesure la fréquence de co-occurrence de paires de niveaux de gris à une certaine distance et orientation.
    Nous pouvons en extraire les propriétés suivantes :
    - le **contraste** : mesure l'intensité des variations locales. Un contraste élevé indique une texture avec de fortes différences de niveaux de gris.
    - l'**énergie** : plus l'énergie est grande, plus la texture est uniforme et répétitive.
    - l'**homogénéité** : reflète la similarité entre pixels voisins. Une forte homogénéité indique une texture lisse.
    - la **corrélation** : mesure la dépendance linéaire entre pixels voisins. Une forte corrélation indique une structure régulière.

    Voici les valeurs moyennes de ces propriétés observées pour chaque classe :
    """)
    
    # Histogrammes GLCM
    propriete_glcm = st.selectbox("Propriété GLCM",["Contraste","Correlation","Energie","Homogeneite"]) 
    col1, col2, col3 = st.columns([0.1, 0.8, 0.1]) 
    with col2:  
        st.image(
            load_image(f"resources/histo_{propriete_glcm.lower()}.png"),
            use_container_width=True
        )
    
    # Fin description textures
    st.markdown("""
    Concernant les textures, les propriétés de Contraste et de Densité de contours semblent avoir un pouvoir discriminant plus marqué en général.
    
    Nous avons complété cette visualisation par des **tests statistiques** (Kruskal-Wallis + test post-hoc de Dunn-Bonferroni).
    Ils nous ont montré que des classes sont significativement différentes l'une de l'autre selon les propriétés texturales observées.
    """)  

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


def show_demo(modeles,X_test,y_test):
    st.header("Démo",divider="gray")

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
    Pour chaque image, **comparez les prédictions de nos modèles** : les erreurs apparaissent en rouge.
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


