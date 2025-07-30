# 🐄 Cow Weight Estimation Web App

This project is a complete web-based system for estimating cattle weight from a single image using deep learning models. It includes:

- ✅ A **FastAPI backend** for inference
- ✅ A **Streamlit frontend** for interactive use
- ✅ SAM2-based segmentation
- ✅ Depth estimation via **Depth Anything v2**
- ✅ A pretrained **PyTorch regression model** that combines RGB, depth, and tabular metadata

---

## 🧠 Features

- 🔍 **Automatic cow segmentation** using SAM2
- 🧮 **Depth estimation** from RGB via Depth Anything
- 📏 **Feature extraction** (area, aspect ratio, width, height, mean/std depth, height, teeth)
- 🧠 **Multi-modal model** for weight prediction: image + depth + metadata
- 🖼️ Visualizes segmentation mask and depth map
- 🌐 Deployable via **GitHub Codespaces** or local server

---

## 📂 Project Structure

```
CowWeightApp/
├── backend/
│   ├── main.py                   # FastAPI app
│   ├── model.py                  # PyTorch weight regressor
│   ├── sam2_loader.py            # SAM2 utilities
│   ├── depth_utils.py            # Depth Anything v2 utilities
│   ├── feature_utils.py          # Feature extraction functions
│   ├── checkpoints/              # SAM2 checkpoint
│   └── model/                    # Pretrained .pth files
├── frontend/
│   ├── app.py                    # Streamlit UI
│   └── app2.py                   # Alternate frontend version
├── cleaned_cow_dataset_final.csv
├── requirements.txt
├── .gitattributes                # Git LFS tracked files
└── .gitignore
```

---

## 🚀 Getting Started

### 🔧 Requirements

- Python 3.10
- Git LFS (for downloading `.pth` models from GitHub)

Install Git LFS:

```bash
sudo apt install git-lfs
git lfs install
```

---

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

---

### ▶️ Running the App

#### 1. Start the FastAPI Backend

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### 2. Start the Streamlit Frontend

```bash
cd frontend
streamlit run app.py
```

---

## 🧢 Model Info

### ✅ Model Types Supported

| Model Name       | Description                        |
| ---------------- | ---------------------------------- |
| `side_all_full`  | Full model with RGB + Depth + Meta |
| `side_all_small` | Smaller full model                 |
| `rgb_only_full`  | RGB-only model (no depth, no meta) |
| `rgb_only_small` | Lightweight RGB-only               |

> The models are stored in: `backend/model/`

---

### 📅 Inputs

- 📸 Cow image (side view)
- 📏 Cow height in inches
- 🧅 Teeth count

### 📤 Output

- 🐮 Predicted weight (kg)
- 📊 Features used
- 🖼️ Segmentation mask (base64)
- 🖼️ Depth map (base64)

---

## 🧠 Models Used

- **Segmentation**: [SAM2](https://github.com/facebookresearch/segment-anything) (Hierarchical Large)
- **Depth**: [Depth Anything v2](https://huggingface.co/isl-org/Depth-Anything)
- **Regression**: Custom CNN + FiLM layers

---

## 💡 Deployment with GitHub Codespaces

This project is Codespaces-ready.

Add this under `.devcontainer/` for GitHub Codespaces deployment:

```json
{
  "name": "CowWeightApp",
  "build": {
    "dockerfile": "Dockerfile"
  },
  "forwardPorts": [8000, 8501],
  "postCreateCommand": "pip install -r requirements.txt"
}
```

And a basic `Dockerfile`:

```Dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt
```

---

## 🛠️ Author & Credits

- 👤 **Abrar Al Sayem**\
  BRAC University | School of Data and Sciences\
  📧 [abrar.al.sayem@g.bracu.ac.bd](mailto\:abrar.al.sayem@g.bracu.ac.bd)

- 👤 **M A Moontasir Abtahee**\
  BRAC University | School of Data and Sciences\
  📧 [m.a.moontasir.abtahee@g.bracu.ac.bd](mailto\:m.a.moontasir.abtahee@g.bracu.ac.bd)
---

---

