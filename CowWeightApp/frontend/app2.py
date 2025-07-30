import streamlit as st
import pandas as pd
import requests
import io, base64, os
from PIL import Image
from statistics import mean

# === Static Paths ===
CSV_PATH = "/home/abrar/Desktop/cow_detection/videos_detection_estimation/cleaned_cow_dataset_final.csv"
IMAGE_ROOT = "/mnt/78707D0F707CD57A/downloaded_videos/isolated/isolated_frames"

# === Setup Page ===
st.set_page_config(page_title="🐄 Cow Weight Estimator (Batch)", layout="wide")
st.title("📂 Batch Cow Weight Estimator (Folder Based)")

# === Load CSV Automatically ===
if not os.path.exists(CSV_PATH):
    st.error(f"❌ CSV file not found: {CSV_PATH}")
    st.stop()

df = pd.read_csv(CSV_PATH)
df.set_index("sku", inplace=True)  # for fast lookup
st.success(f"✅ Loaded metadata for {len(df)} cows from CSV")

# === Detect available folders ===
available_skus = sorted([
    f for f in os.listdir(IMAGE_ROOT)
    if os.path.isdir(os.path.join(IMAGE_ROOT, f))
])
st.info(f"📁 Found {len(available_skus)} folders under image root")

# === Estimate Button ===
if st.button("🚀 Estimate All from Folders"):
    results = []

    for sku in available_skus:
        if sku not in df.index:
            st.warning(f"⚠️ SKU {sku} not found in CSV, skipping...")
            continue

        # Get metadata from CSV
        try:
            height = float(df.loc[sku]["height_in_inch"])
            teeth = int(df.loc[sku]["teeth"])
            actual_weight = float(df.loc[sku]["weight_in_kg"])
        except Exception as e:
            st.warning(f"⚠️ Invalid metadata for {sku}: {e}")
            continue

        folder_path = os.path.join(IMAGE_ROOT, sku)
        predictions = []
        image_features = []

        for f in sorted(os.listdir(folder_path)):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_file = os.path.join(folder_path, f)

            with open(image_file, "rb") as img:
                files = {"image_file": (os.path.basename(image_file), img, "image/jpeg")}
                data = {"height_in_inch": height, "teeth": teeth}
                try:
                    response = requests.post("http://localhost:8000/predict/", files=files, data=data)
                except Exception as e:
                    st.error(f"❌ Error contacting backend for {sku} image {f}: {e}")
                    continue

            if response.status_code != 200:
                st.error(f"❌ Failed for SKU {sku}, Image {f}, Status: {response.status_code}")
                continue

            result = response.json()
            pred = result["cow_weight_pred"]
            item = result["features"][0]

            predictions.append(pred)
            image_features.append(item)

        if not predictions:
            st.warning(f"⚠️ No successful predictions for {sku}")
            continue

        avg_pred = round(mean(predictions), 2)
        abs_error = round(abs(avg_pred - actual_weight), 2)
        rep_feat = image_features[0]

        results.append({
            "sku": sku,
            "actual_weight": actual_weight,
            "predicted_weight_avg": avg_pred,
            "abs_error": abs_error,
            **dict(zip([
                "area", "width", "height", "aspect_ratio",
                "mean_depth", "std_depth", "height_in_inch", "teeth"
            ], rep_feat["features"]))
        })

        # === Visualize for this cow ===
        st.markdown(f"""
        ### 🐮 SKU: `{sku}`
        - 🏷️ **Actual Weight:** `{actual_weight} kg`
        - 📦 **Avg Predicted Weight (from {len(predictions)} images):** `{avg_pred} kg`
        - 📉 **Absolute Error:** `{abs_error} kg`
        - 📏 **Height:** `{height} inch`
        - 🦷 **Teeth Count:** `{teeth}`
        """)

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.image(Image.open(os.path.join(folder_path, os.listdir(folder_path)[0])), caption="Sample Original", use_container_width=True)
        with col2:
            st.image(Image.open(io.BytesIO(base64.b64decode(rep_feat["mask_b64"]))), caption="Segmentation", use_container_width=True)
        with col3:
            st.image(Image.open(io.BytesIO(base64.b64decode(rep_feat["depth_b64"]))), caption="Depth Map", use_container_width=True)

        st.markdown("---")

    # === Final Summary Table ===
    if results:
        st.subheader("📊 Summary Table: Avg Prediction vs Actual Weight + Metadata")
        result_df = pd.DataFrame(results)
        st.dataframe(result_df, use_container_width=True)
