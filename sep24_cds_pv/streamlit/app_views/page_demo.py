import streamlit as st


def show_demo(modeles,X_test,y_test):
    st.header("Démo",divider="gray")

    st.markdown("""
                Choisissez des images de test : vous pouvez afficher tous les types de défauts, ou vous concentrer sur un défaut en particulier.

                Remarque : ces images n'ont pas été utilisées lors de l'entraînement des modèles.
    """)

    with st.container(border=True):
        # Choix par l'utilisateur d'afficher une image de chaque classe, ou une classe en particulier
        all_classes = st.toggle("Tester tous les types de défauts",False)

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
                nb_img = st.slider("Nombre d'images à afficher",1,10,3)
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
