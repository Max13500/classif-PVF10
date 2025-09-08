import streamlit as st

from app_views import load_image


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
    
    # Création des tabs
    tab1, tab2, tab3 = st.tabs(["Contexte", "Objectifs", "Données"])
    
    # Tab 1 : Contexte
    with tab1:

        # Section 1
        st.subheader("La nécessaire transition énergétique")

        with st.columns([1,2,1])[1]:
            st.image(
                "resources/presentation/transition.png",
                # width=200,
                )

        st.markdown("""
            Dans le contexte actuel de **réchauffement climatique et d'épuisement des ressources fossiles**, 
            **:red[l'usage des énergies renouvelables augmente]**.
        """)

        # Section 2
        st.subheader("L'énergie photovoltaïque")

        st.markdown("""
            Parmi les énergies renouvelables, **:red[l'énergie photovoltaïque] voit sa capacité de production installée 
            augmenter de façon exponentielle** depuis le début des années 2000. 
        """)

        with st.columns([1,2,1])[1]:
            st.image("resources/presentation/installed-solar-pv-capacity.svg", width=800)

            st.image(
                "resources/presentation/Vue-aérienne-de-la-centrale-photovoltaïque-de-Cestas.jpg",
                caption="Centrale photvoltaïque de Cestas - la plus grande de France (en 2025) - 300 MWc - 260 hectares",
                width=800,
            )
        
        # Section 3
        st.subheader("L'enjeu : maintenir le niveau de production")

        st.markdown("""
            - **Au cours de la vie d'une centrale** de production électrique, **:red[de nombreux défauts différents peuvent apparaître]** sur les panneaux photovoltaïques.
            - Ces **défauts** ont un **:red[impact sur l'énergie produite]**, et sur les **:red[revenus financiers] de l'industriel exploitant**.
            - Sur des centrales de grande taille, **l'inspection manuelle :red[n'est pas possible]**.
        """)

        # Section 4
        st.subheader("Une solution : l'inspection par imagerie thermique")

        st.markdown("""
            L'utilisation de **drones équipés de caméras thermiques** permet de **parcourir l'ensemble d'une centrale en :red[quelques heures]**.
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                "resources/presentation/Termal-Muayene.jpg",
                caption="Drone inspectant une centrale PV",
            )

        with col2:
            st.text('')
            st.text('')
            st.image(
                "resources/presentation/Image IR drone.jpg",
                caption="Exemple de défauts vus par caméra thermique",
            )

    # Tab 2 : Objectifs du projet
    with tab2:

        # Section 1
        st.subheader("Automatiser la détection et l'identification des défauts")

        with st.columns([1,8,1])[1]:
            st.image(
                "resources/presentation/solar_panels_defects.png",
                caption="Exemples de signatures thermiques de différents défauts",
            )

        st.markdown("""
            - **Les défauts** qui apparaissent sur les panneaux photovoltaïques présentent des **:red[signatures thermiques caractéristiques]**.
            - L'analyse de **plusieurs dizaines ou centaines de milliers d'images** n'est **:red[pas possible] pour un opérateur humain**.
            - Mais c'est précisément **le domaine d'action des algorithmes de :red[Machine Learning et Deep Learning]**.
        """)

        # Section 2
        st.subheader("Notre jeu de données : PVF-10")

        st.markdown("""
            - Issu d'une **publication scientifique datée d':red[octobre 2024]**.
            - **Dataset :red[entièrement annoté] mis à disposition** de la communauté scientifique.
            - **:red[5579 images] différentes** déclinées en **3 formats**.
            - **:red[9] types de défauts** différents **+ :red[1] type "sain"**.
        """)

        with st.columns([1,1,1])[1]:
            st.image("resources/presentation/elsevier.png")

        with st.expander("Les 10 classes répertoriées dans PVF-10"):
            st.markdown("""
                - **:red[bottom dirt]** : Accumulation de salissures (poussière, boue, sable) sur le bas du panneau. **Impact** : diminution locale de l'irradiance, légère surchauffe des zones propres.
                - **:red[break]** : Fissure ou rupture visible d'une cellule ou d'un module. **Impact** : forte surchauffe locale, danger de points chauds. Défaut critique.
                - **:red[debris cover]** : Présence d'un objet étranger sur la surface (feuilles, plastiques, etc.). **Impact** : ombrage irrégulier entraînant une élévation thermique hétérogène.
                - **:red[junction box heat]** : Surchauffe localisée au niveau de la boîte de jonction. **Impact** : défaut électrique potentiellement dangereux, perte d'efficacité.
                - **:red[hot cell]** : Cellule ou groupe de cellules présentant une température anormalement élevée. **Impact** : défaut thermique ponctuel souvent causé par une mauvaise connexion ou une cellule défectueuse.
                - **:red[shadow]** : Ombrage partiel dû à des éléments extérieurs (branches, câbles, etc.). **Impact** : baisse de rendement temporaire, souvent visible en bandes froides.
                - **:red[short circuit panel]** : Court-circuit généralisé affectant l'ensemble du panneau. **Impact** : très forte surchauffe homogène, risque de dégradation accélérée.
                - **:red[string short circuit]** : Court-circuit affectant une chaîne de cellules. **Impact** : surchauffe linéaire visible dans une zone continue du panneau.
                - **:red[substring open circuit]** : Ouverture du circuit dans une sous-chaîne de cellules. **Impact** : surchauffe isolée, comportement thermique anormal sur une ligne.
                - **:red[healthy panel]** : Panneau sans défaut thermique ou structurel. Référence de fonctionnement normal.
            """)

    # Tab 3 : Aperçu des données
    with tab3:

        st.subheader("Aperçu des données du dataset PVF-10")
        
        # On récupère au hasard une ligne du dataframe par classe
        df_sel = df.groupby("Classe").sample(1)

        data_col1, data_col2 = st.columns([4,3])

        with data_col2:
            # Choix par l'utilisateur d'afficher les images en niveau de gris
            grayscale = st.toggle("Afficher les images en niveaux de gris", False)

            # Bouton pour rafraichir => grâce au tirage aléatoire, on affichera d'autres images avec les paramètres sélectionnés
            if st.button("🔄 Changer d'images"):
                pass

        with data_col1:
            # Affichage de 10 images : 5 images sur 2 lignes, avec leur classe en titre
            with st.container(border=True, width=700):
                for i in range(0, 10, 5):
                    cols = st.columns(5)
                    for j, col in enumerate(cols):
                        if i + j < len(df_sel):
                            image = load_image(df_sel["Chemin"].iloc[i + j])
                            if grayscale:
                                image = image.convert("L")
                            col.image(image, caption = df_sel["Classe"].iloc[i + j], width="stretch")
        
        st.markdown("""
            - **Les images thermographiques "brutes" sont :red[encodées sur un seul canal]** (la valeur de chaque pixel est fonction de la :red[température de l'objet]).
            - La représentation en couleurs est dûe à l'**application d'une palette RGB arbitraire** (de type "inferno" par exemple).
        """)
