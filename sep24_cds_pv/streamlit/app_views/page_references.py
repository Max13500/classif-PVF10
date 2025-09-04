import streamlit as st


def show_references():
    st.header("Références", divider="gray")

    st.subheader(f"GitHub du projet")
    st.markdown("""
        -  Notebooks, rapports et appli Streamlit  : [SEP24-CDS-PHOTOVOLTAIQUE](https://github.com/DataScientest-Studio/SEP24-CDS-PHOTOVOLTAIQUE)
    """)

    st.subheader(f"Images du jeu de données")
    st.markdown("""
        -  Dataset PVF-10 : [Télécharger le zip](https://drive.usercontent.google.com/download?id=1SQq0hETXi8I3Kdq9tDAEVyZgIsRCbOah&export=download&authuser=0)
    """)

    st.subheader(f"Publication scientifique des auteurs du dataset PVF-10")
    st.markdown("""
        - Long, H., Sun, J., Feng, Y., Qin, H., Zhang, J., & Li, H. (2024). 
        [PVF-10: A high-resolution unmanned aerial vehicle thermal infrared image dataset for fine-grained photovoltaic fault classification.](https://doi.org/10.1016/j.apenergy.2024.124187)
        Applied Energy, 362, 123972.
    """)

    st.subheader(f"Sources des images et illustrations")
    st.markdown("""
        - Section **Présentation**
            - Transition des énergies fossiles vers les énergies renouvelables : ChatGPT
            - Capacité solaire photovoltaïque installée : [Our World in Data](https://ourworldindata.org/grapher/installed-solar-pv-capacity)
            - Centrale photvoltaïque de Cestas : Photo Neoen - licence : [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.fr)
            - Exemples de défauts vus en thermographie infrarouge : FLIR - **A guide to inspecting solar fields with thermal imaging drones**
            - Image d'un drone d'inspection en survol d'une centrale PV : [MapperX](https://mapperx.com/fr/avantages-de-linspection-thermique/)
            - Tableau des signatures thermiques de différents types de défauts : TESTO - **Practical Guide - Solar Panel Thermography**
        - Section **Méthode**
            - Arbre de décision : [Insight IMI](https://insightimi.wordpress.com/2020/12/28/implementing-decision-tree-and-random-forest-on-python/)
            - Réseau de neurones convolutif : [ResearchGate](https://www.researchgate.net/figure/Schematic-structure-of-convolutional-neural-network-CNN_fig3_371870690)
            - Schéma du Transfer Learning : [Medium](https://medium.com/data-science/how-to-easily-draw-neural-network-architecture-diagrams-a6b6138ed875)
    """)


