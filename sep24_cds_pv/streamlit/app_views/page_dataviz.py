import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .utils import load_image


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
            width="stretch",
        )
    
    # Fin description textures
    st.markdown("""
    Concernant les textures, les propriétés de Contraste et de Densité de contours semblent avoir un pouvoir discriminant plus marqué en général.
    
    Nous avons complété cette visualisation par des **tests statistiques** (Kruskal-Wallis + test post-hoc de Dunn-Bonferroni).
    Ils nous ont montré que des classes sont significativement différentes l'une de l'autre selon les propriétés texturales observées.
    """)  
