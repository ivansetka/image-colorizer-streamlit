import streamlit as st
import torch
from PIL import Image
from huggingface_hub import hf_hub_download

from .config import model_map
from .postprocessing import postprocess
from .preprocessing import preprocess


@st.cache_resource
def load_model(model_name):
    filename, model_cls = model_map[model_name]
    weights_path = hf_hub_download(
        repo_id="ivansetka/image-colorizer",
        filename=filename,
        cache_dir="models_weights"
    )

    model = model_cls()
    model.load_weights(weights_path)

    return model


def colorize_image(image, model_name, model):
    original_image_size = image.size
    grayscale_input = preprocess(image, model_name)

    with torch.no_grad():
        generated_image = model.colorize(grayscale_input)

    colorized_output = postprocess(generated_image, model_name)
    colorized_output = Image.fromarray(colorized_output).resize(original_image_size)

    return colorized_output

