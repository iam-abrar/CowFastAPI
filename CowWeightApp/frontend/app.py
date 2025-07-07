import streamlit as st
import requests
import io
import base64
from PIL import Image
import pandas as pd

st.set_page_config(page_title="Cow Weight Estimator", layout="wide")
st.title("🐄 Cow Weight Estimator Dashboard")

with st.sidebar:
    st.header("Input Parameters")
    height = st.number_input("Height (in inches)", min_value=1.0, max_value=100.0, step=0.5)
    teeth = st.number_input("Teeth Count", min_value=1, max_value=10)
    image_file = st.file_uploader("Upload a cow image", type=["jpg", "jpeg", "png"])

    if st.button("Estimate Weight"):
        if image_file is None:
            st.warning("Please upload an image.")
        else:
            with st.spinner("Processing..."):
                files = {"image_file": (image_file.name, image_file, "image/jpeg")}
                data = {"height_in_inch": height, "teeth": teeth}
                response = requests.post("http://localhost:8000/predict/", files=files, data=data)

                if response.status_code == 200:
                    result = response.json()
                    st.success(f"✅ Prediction Complete for {result['image_count']} image(s)")
                    st.subheader(f"📦 Estimated Cow Weight: **{result['cow_weight_pred']} kg**")

                    item = result["features"][0]
                    col1, col2, col3 = st.columns([1, 1, 2])
                    with col1:
                        st.image(Image.open(io.BytesIO(base64.b64decode(item["mask_b64"]))), caption="Segmentation")
                    with col2:
                        st.image(Image.open(io.BytesIO(base64.b64decode(item["depth_b64"]))), caption="Depth Map")
                    with col3:
                        st.markdown(f"**Image:** `{item['filename']}`")
                        feature_names = [
                            "area", "width", "height", "aspect_ratio",
                            "mean_depth", "std_depth", "height_in_inch", "teeth"
                        ]
                        st.dataframe(pd.DataFrame(
                            [item["features"]],
                            columns=feature_names
                        ), use_container_width=True)
                    st.markdown("---")
                else:
                    st.error(f"❌ Request failed with status code {response.status_code}")
