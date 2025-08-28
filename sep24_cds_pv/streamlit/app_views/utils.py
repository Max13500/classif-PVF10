import streamlit as st
from PIL import Image


@st.cache_data
def load_image(path):
    return Image.open(path)
