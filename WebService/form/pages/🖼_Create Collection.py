import streamlit as st
import io
import os
import yaml
import base64
import requests

from PIL import Image
from typing import List

st.set_page_config(page_title="Create collection", page_icon="🖼")


@st.experimental_memo
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


img = get_img_as_base64("./form/images/cool-background.png")
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

with open("./configs/streamlit.yaml", 'r') as stream:
    streamlit_config = yaml.safe_load(stream)

existen_models = streamlit_config["existing_model"]


def train_request(uploaded_files: List[str], asset_type: str):
    url = streamlit_config["backend_hostname"] + \
        streamlit_config["train_endpoint"]
    files = [
        ('images', file.getvalue())
        for file in uploaded_files
    ]
    result = requests.post(
        url,
        files=files,
        data={"asset_type": asset_type}
    )
    return result


asset_type = st.text_input(label="Colection name", value="The name will be used as an object name")
uploaded_files = st.file_uploader(
    "Please choose a file",
    accept_multiple_files=True,
    label_visibility="hidden"
)


if uploaded_files:
    n = st.slider(
        label="Select a number of image in raw",
        min_value=1,
        max_value=len(uploaded_files) + 1,
        value=(len(uploaded_files) + 1) // 2,
        label_visibility="hidden",
    )

    groups = []
    for i in range(0, len(uploaded_files), n):
        groups.append(uploaded_files[i: i + n])

    for group in groups:
        cols = st.columns(n)
        for i, image_file in enumerate(group):
            cols[i].image(image_file)


if uploaded_files and asset_type:
    if st.button("Train"):
        images_path = os.path.join(existen_models, asset_type, "images")
        os.makedirs(images_path)
        for i, file in enumerate(uploaded_files):
            image = Image.open(io.BytesIO(file.getvalue()))
            image.save(os.path.join(images_path, f"{i}.png"))
        train_response = train_request(uploaded_files, asset_type)
