import base64
import streamlit as st

st.set_page_config(
    page_title="Genlock",
    page_icon="👋"
)

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

original_title = """
<p style='font-family: apple-system, sans-serif; font-weight: bold; text-align: center; color: black; font-size: 35px;'>
✨ Genlock - we can change the game ✨
</p>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)
for _ in range(20):
    st.write("")
st.markdown(original_title, unsafe_allow_html=True)
