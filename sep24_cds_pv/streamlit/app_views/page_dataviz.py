import streamlit as st
from PIL import Image
import cv2
from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix, graycoprops
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .utils import load_image


def show_dataviz(df,statistiques):
    st.header("DataViz",divider="gray")

    # Création des tabs
    tab1, tab2, tab3 = st.tabs(["Analyse générale", "Couleurs / Niveaux de gris", "Textures"])

    # Tab 1 : Analyse générale
    with tab1:
        st.subheader("Structure du dataset")

        # Description du dataset et nettoyage
        st.markdown("""
        Notre étude porte sur **:red[5579 images de type PNG au format 110x60]**.
                    
        **Nettoyage** de ce jeu de données :
        - suppression de 7 doublons d'images    
        - annotation de 4% des images qui ont une dimension réelle différente de 110x60 (=> potentiellement mauvais découpage ou resizing)
        - renommage des 10 classes pour plus de lisibilité
        """)

        # Affichage du dataframe des métadonnées
        with st.expander("Pour visualiser le dataframe final contenant les **métadonnées**..."):
            st.dataframe(df)
        
        st.subheader("Equilibre des classes")

        # Description de l'équilibre des classes
        st.markdown("""
        **Répartition en 10 classes :red[légèrement déséquilibrée]** :
        - la classe des panneaux sains représente un peu plus d'un quart des observations
        - les neuf classes de défauts se partagent le reste de manière relativement équitable
        - deux catégories sont toutefois **en retrait : Break et String short circuit**.
        """)

        # Diagramme de répartition des classes
        with st.columns([0.2, 0.6, 0.2])[1] :     
            fig = plt.figure()
            sns.countplot(y = df['Classe'],hue = df['Classe'],legend=False)
            plt.title("Répartition des classes de défauts",fontsize=12, fontweight='bold')
            plt.xlabel("Nombre d'images")
            plt.ylabel("Classe de défaut")
            sns.despine()
            st.pyplot(fig)

    # Tab 2 : Couleurs et niveaux de gris
    with tab2:
        st.subheader("Les pseudo-couleurs")

        # Description des canaux RGB
        st.markdown("""
        Visualisez la répartition des **intensités des canaux Rouge Vert Bleu** pour l'ensemble des classes :
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
        
        st.markdown("""
        - Les images thermiques ont une **composante rouge très élevée** et une composante bleue faible => ce sont des **:red[pseudo-couleurs]**
        - Pour la suite de l'étude : **images converties en niveaux de gris**
        """) 

        st.subheader("Les niveaux de gris")

        # Description des niveaux de gris
        st.markdown("""
        Visualisez pour chaque classe les histogrammes de 5 **indicateurs statistiques** des niveaux de gris :
        """)    

        # L'utilisateur choisit l'indicateur statistique
        indicateur = st.selectbox("Indicateur statistique",list(statistiques.keys())[1:6],1) 
        # Afficher 2 classes en parallèle
        ndg_cols = st.columns(2)
        for i,c in enumerate(ndg_cols):
            with c:
                fig = plt.figure()
                # L'utilisateur choisit la classe
                nom_classe = st.selectbox("Classe de défaut :" if i==0 else "Comparer avec :",
                                        df["Classe"].unique(),
                                        6 if i==0 else 9,
                                        key=f"classe_ndg_{i}")
                # Récupération de l'indicateur statistique demandé sur les NDG
                statistique = statistiques[indicateur][nom_classe]
                # Afficher l'histogramme et la densité de probabilité de l'indicateur
                sns.histplot(statistique,bins=20,stat="density",kde=True,alpha=0.6)
                plt.xlabel(indicateur,fontsize=14)
                plt.ylabel("Densité de probabilité",fontsize=14)
                plt.title(f"Histogramme (avec densité KDE)",fontsize=14, fontweight='bold')
                st.pyplot(fig)

        st.subheader("A retenir")

        # Bilan
        st.markdown("""
        - **La distribution des niveaux de gris montre des :red[spécificités selon le type de défauts]**.
        - Les différences sont en général plus marquées sur les indicateurs **Max** et **Ecart-type**.
        - Des **tests statistiques** ont confirmé la différence significative entre plusieurs classes d'après ces indicateurs.
        """)

        # On récupère 2 exemples : break et healthy panel
        st.markdown("""
        **Exemples d'indicateurs de niveaux de gris** calculés sur des images Break et Healthy Panel :
                    """)
        chemin_ex_break = df.loc[(df["Nom"]=="DJI_20230223114728_0989_T_000001")]["Chemin"].iloc[0]
        chemin_ex_healthy = df.loc[(df["Nom"]=="DJI_20230222130345_0306_T_000024")]["Chemin"].iloc[0]

        # Pour chaque exemple :
        for chemin in [chemin_ex_break,chemin_ex_healthy]:
            # Récupération de l'image en NdG
            img_gray = Image.open(chemin).convert("L") 
            img_gray_array = np.array(img_gray)
            # Calcul des indicateurs statistiques sur le NdG
            intensite_min = np.min(img_gray_array)
            intensite_max = np.max(img_gray_array)
            intensite_median = np.median(img_gray_array)
            intensite_moy = np.mean(img_gray_array)
            intensite_std = np.std(img_gray_array)
            # Affichage des images et des indicateur statistiques
            with st.container(border=True):
                col1,col2,col3,col4,col5,col6 = st.columns(6)
                with col1:
                    st.image(img_gray,caption="Break" if chemin==chemin_ex_break else "Healthy")
                with col2:
                    st.metric("NdG min", intensite_min)
                with col3:
                    st.metric("NdG max", intensite_max)
                with col4:
                    st.metric("NdG moyen", np.round(intensite_moy,2))
                with col5:
                    st.metric("NdG médian", intensite_median)
                with col6:
                    st.metric("Ecart-type NdG", np.round(intensite_std,2))

    # Tab 3 : Textures
    with tab3:
        st.subheader("Densité de contours et entropie")

        # Description des propriétés de texture
        st.markdown("""
        Visualisez pour chaque classe les histogrammes de **propriétés texturales** extraites des images :
        """) 
        
        # L'utilisateur choisit la propriété de texture
        propriete = st.selectbox("Propriété texturale",list(statistiques.keys())[6:8],1) 
        # Afficher 2 classes en parallèle
        prop_cols = st.columns(2)
        for i,c in enumerate(prop_cols):
            with c:
                fig = plt.figure()
                # L'utilisateur choisit la classe
                nom_classe = st.selectbox("Classe de défaut :" if i==0 else "Comparer avec :",
                                        df["Classe"].unique(),
                                        6 if i==0 else 9,
                                        key=f"classe_prop_{i}")
                # Récupération de la propriété demandée
                statistique = statistiques[propriete][nom_classe]
                # Afficher l'histogramme et la densité de probabilité de la propriété
                sns.histplot(statistique,bins=20,stat="density",kde=True,alpha=0.6)
                plt.xlabel(propriete,fontsize=14)
                plt.ylabel("Densité de probabilité",fontsize=14)
                plt.title(f"Histogramme (avec densité KDE)",fontsize=14, fontweight='bold')
                st.pyplot(fig)
        
        st.subheader("Proriétés GLCM")
        st.markdown("""
        Voici d'autres **propriétés texturales, extraites de la matrice [GLCM](https://en.wikipedia.org/wiki/Co-occurrence_matrix)** des images :
        """)

        # Histogrammes GLCM
        propriete_glcm = st.selectbox("Propriété texturale GLCM",["Contraste","Correlation","Energie","Homogeneite"]) 
        col1, col2, col3 = st.columns([0.25, 0.5, 0.25]) 
        with col2:  
            st.image(
                load_image(f"resources/dataviz/histo_{propriete_glcm.lower()}.png"),
                width="stretch",
            )

        st.subheader("A retenir sur les textures")

        # Bilan des propriétés texturales
        st.markdown("""
        - **La distribution des propriétés calculées montre des :red[spécificités selon le type de défauts]**.
        - Les différences sont en général plus marquées sur le **contraste** et la **densité de contours**.
        - Des **tests statistiques** ont confirmé la différence significative entre plusieurs classes d'après ces propriétés.
        """)

        # On récupère 2 exemples : break et healthy panel
        st.markdown("""
        **Exemples de propriétés texturales** calculées sur des images Break et Healthy Panel :
                    """)
        chemin_ex_break = df.loc[(df["Nom"]=="DJI_20230223114728_0989_T_000001")]["Chemin"].iloc[0]
        chemin_ex_healthy = df.loc[(df["Nom"]=="DJI_20230222130345_0306_T_000024")]["Chemin"].iloc[0]
        # Pour chaque exemple
        for chemin in [chemin_ex_break,chemin_ex_healthy]:
            # Récupération de l'image en NdG
            img_gray = Image.open(chemin).convert("L") 
            img_gray_array = np.array(img_gray)
            # Calcul de l'entropie
            entropie = shannon_entropy(img_gray_array)
            # Calcul de la densité de contours
            edges = cv2.Canny(img_gray_array[2:-2,2:-2], 70, 140) # Filtre de Canny sur l'image rognée
            edge_density =  np.sum(edges > 0) / edges.size # On en déduit la densité de contours
            # Calcul des propriétés GLCM
            glcm = graycomatrix(img_gray_array, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)
            contraste = graycoprops(glcm, 'contrast').mean()
            homogeneite = graycoprops(glcm, 'homogeneity').mean()
            energie = graycoprops(glcm, 'energy').mean()
            correlation = graycoprops(glcm, 'correlation').mean()
            # Affichage des images et des propriétés texturales
            with st.container(border=True):
                col1,col2,col3,col4,col5,col6,col7 = st.columns(7)
                with col1:
                    st.image(img_gray,caption="Break" if chemin==chemin_ex_break else "Healthy")
                with col2:
                    st.metric("Entropie", np.round(entropie,2))
                with col3:
                    st.metric("Densité de contours", f"{np.round(edge_density*100,1)} %")
                with col4:
                    st.metric("Contraste", np.round(contraste,2))
                with col5:
                    st.metric("Homogénéité", np.round(homogeneite,2))
                with col6:
                    st.metric("Energie", np.round(energie,2))
                with col7:
                    st.metric("Corrélation", np.round(correlation,2))

        # Définition des propriétés texturales
        with st.expander("Pour en savoir plus sur les propriétés texturales..."):

            st.markdown("""
            - l'**entropie** : quantifie la diversité des niveaux de gris (entre 0 et log2(256), soit 8). Une entropie élevée traduit une texture complexe.
            - la **densité de contours** : proportion de contours dans l'image après application du filtre de Canny, indiquant des transitions abruptes.
            - la **matrice [GLCM](https://en.wikipedia.org/wiki/Co-occurrence_matrix)** : décrit la fréquence à laquelle des paires de niveaux de gris (i,j) apparaissent dans une image, selon une certaine distance et direction entre pixels.
            On en déduit les propriétés suivantes : 
                - le **contraste** (entre 0 et 255²): mesure l'écart d'intensité entre pixels voisins. Un contraste élevé indique de fortes variations de niveaux de gris.
                - la **corrélation** (entre -1 et 1): mesure la dépendance linéaire entre pixels voisins. Une forte corrélation suggère une structure régulière.
                - l'**énergie** (entre 0 et 1): reflète l'uniformité de la distribution. Une énergie basse indique une image totalement aléatoire (toutes transitions équiprobables).
                - l'**homogénéité** (entre 0 et 1) : reflète la similarité entre pixels voisins. Une forte homogénéité indique une texture lisse.
            """)
