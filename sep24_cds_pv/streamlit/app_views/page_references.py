import streamlit as st


def show_references():
    st.header("Références", divider="gray")

    st.subheader(f"Publication scientifique des auteurs du dataset PVF-10")
    st.markdown("""
        - Long, H., Sun, J., Feng, Y., Qin, H., Zhang, J., & Li, H. (2024). 
        [PVF-10: A high-resolution unmanned aerial vehicle thermal infrared image dataset for fine-grained photovoltaic fault classification.](https://doi.org/10.1016/j.apenergy.2024.124187)
        Applied Energy, 362, 123972.
    """)


    st.subheader(f"Sources des images et illustrations")
    st.markdown("""
        - Section **Présentation**
            - Transition des énergies fossiles vers les énergies renouvelables : GhatGPT
            - Capacité solaire photovoltaïque installée : [Our World in Data](https://ourworldindata.org/grapher/installed-solar-pv-capacity)
            - Centrale photvoltaïque de Cestas : Photo Neoen - licence : [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.fr)
            - Exemples de défauts vus en thermographie infrarouge : FLIR - **A guide to inspecting solar fields with thermal imaging drones**
            - Image d'un drone d'inspection en survol d'une centrale PV : [MapperX](https://mapperx.com/fr/avantages-de-linspection-thermique/)
            - Tableau des signatures thermiques de différents types de défauts : TESTO - **Practical Guide - Solar Panel Thermography**
            
    """)


