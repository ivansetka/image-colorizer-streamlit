import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

from pathlib import Path

import streamlit as st
from utils.infer import load_model, colorize_image
from PIL import Image


BASE_DIR = Path(__file__).parent
logo_path = str(BASE_DIR / "assets" / "logo.svg")

st.set_page_config(
    page_title='Image Colorizer',
    page_icon=logo_path,
    layout='wide',
    initial_sidebar_state='expanded',
)

st.sidebar.image(logo_path)
st.sidebar.markdown("---")

model_name = st.sidebar.radio(
    label="Choose colorization model:",
    options=["Convolutional model", "GAN model", "Diffusion-based model", "Transformer-based model"]
)

col1, _ = st.columns([1.82, 1])

with col1:
    st.markdown("""
        # Deep Learning Image Colorizer
    
        This application allows you to upload your own grayscale image and apply one of several deep learning models to 
        automatically colorize it. The models featured here are based on convolutional, adversarial, 
        diffusion, and transformer architectures.

        They were developed and trained as part of a thesis project focused on generative approaches to 
        image colorization. The goal was to evaluate their performance on both standard datasets and historical
        black-and-white photos. For more details, see the repository linked below ⬇️
        
        <br/>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("### -> &nbsp;**Upload a grayscale image**:", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        col11, col12 = st.columns([1, 1])

        image = Image.open(uploaded_file).convert('L')
        col11.image(image, caption="Grayscale Input", use_container_width=True)

        with col12:
            if model_name != "Diffusion-based model":
                model = load_model(model_name)
                result = colorize_image(image, model_name, model)
                st.image(result, caption=f"Colorized Output: {model_name}", use_container_width=True)
            else:
                st.markdown("""
                    *Unfortunately, the diffusion model is too slow for practical use in this demo, 
                    as it runs many steps sequentially on the CPU.* 😕
                """)

    st.markdown("""
        ---
        Check out the full project and code on my [GitHub repo](https://github.com/Luka958/eval-fusion/tree/main) 🚀
    """)
