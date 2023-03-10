import streamlit as st
import cv2
import os
import yaml
import numpy as np
import base64
import requests

from PIL import Image
from streamlit_image_select import image_select


st.set_page_config(page_title="Generate Image", page_icon="🚀")


@st.experimental_memo
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_img = get_img_as_base64("./form/images/cool-background.png")

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{bg_img}");
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

with open("./configs/streamlit.yaml", 'r') as stream:
    streamlit_config = yaml.safe_load(stream)

collections_path = streamlit_config["collections_path"]


def generate_request(asset_type: str):
    url = streamlit_config["backend_hostname"] + \
        streamlit_config["generate_endpoint"] + f"/{asset_type}/"
    return requests.post(url)


preview_images = []
captions = []
for model_name in os.listdir(collections_path):
    image_path = os.path.join(collections_path, model_name, "images", "1.png")
    preview_images.append(
        image_path
    )
    captions.append(model_name)


img_path = image_select(
    "Existing Collections 🖼",
    preview_images,
    captions=captions,
    use_container_width=False
)

asset_type = img_path.split("/")[2]

st.markdown(
    """
    <span style="font-weight:bold;">Your wallet</span>
    """,
    unsafe_allow_html=True
)
st.code('''0xd5c962029b823bf5''', language='bash')

st.markdown(
    """
    <span style="font-weight:bold;">Collection name</span>
    """,
    unsafe_allow_html=True
)
st.code(f'''{asset_type}''', language='bash')

st.markdown(
    """
    <span style="font-weight:bold;">Generation endpoint</span>
    """,
    unsafe_allow_html=True
)
code = '''http://127.0.0.1:8000/api/v1/generate/{asset_type}/'''
st.code(code, language='bash')


col1, col2, col3 = st.columns(3)
response = None
with col2:
    if st.button("Generate"):
        response = generate_request(asset_type)

col1, col2, col3 = st.columns(3)

with col1:
    if response and response.status_code == 201:
        nparr = np.frombuffer(base64.b64decode(
            response.json()["image"]
        ), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)[:, :, [2, 1, 0, 3]]
        pil_img = Image.fromarray(img)
        st.image(pil_img)

with col3:
    if response and response.status_code == 201:
        st.markdown(
            """
            <span style="font-weight:bold;">Item info</span>
            """,
            unsafe_allow_html=True
        )
