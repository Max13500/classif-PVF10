import streamlit as st


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
