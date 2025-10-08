🧠 Anomaly Detection API

This repository provides a FastAPI-based image anomaly detection service built on Anomalib.
It includes:

A REST API for inference with image uploads

Endpoints to retrieve anomaly visualization overlays

Jupyter notebooks demonstrating model training and analysis

Dockerfile for easy deployment



---

📁 Repository Structure

anomaly/
│
├── main.py               # FastAPI application (this API)
├── model/
│   └── model.ckpt        # Pretrained model checkpoint (required for inference)
│
├── notebooks/            # Jupyter notebooks for training / exploration
│   ├── train_patchcore.ipynb
│   ├── evaluate_model.ipynb
│   └── ...
│
├── pyproject.toml        # Poetry project dependencies
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container build configuration
└── README.md


---

🚀 Quick Start

1. Environment Setup

Requires Python ≥ 3.11.

git clone https://github.com/vkupershtein/anomaly.git
cd anomaly
python -m venv venv
source venv/bin/activate    # (Windows: venv\Scripts\activate)
pip install -U pip
pip install .

> You can also install dependencies directly:

pip install albumentationsx>=2.0.11 anomalib>=2.1.0 jupyterlab>=4.4.7



2. Run the FastAPI App

Start the API:

uvicorn main:app --reload

The API will be available at:
👉 http://127.0.0.1:8000

Interactive Swagger docs:
👉 http://127.0.0.1:8000/docs


---

🐳 Docker Deployment

A preconfigured Dockerfile is included for lightweight deployment using Python 3.12-slim.

FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN apt-get update && apt-get install -y python3-opencv
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}

Build & Run

docker build -t anomaly-api .
docker run -p 8000:8000 anomaly-api

Then visit http://localhost:8000/docs for interactive API docs.


---

🧩 API Endpoints

POST /predict

Description:
Upload an image to obtain anomaly detection results.

Parameters:

file: image file (PNG/JPEG)

threshold: optional float form parameter (default 0.3) to classify as anomaly/non-anomaly


Response JSON:

{
  "id": "7b32a8fc-6a9f-42a8-b8cb-ef30fcd30f3f",
  "score": 0.42,
  "label": 1
}

Example with cURL:

curl -X POST "http://127.0.0.1:8000/predict" \
  -F "file=@example.jpg" \
  -F "threshold=0.4"


---

GET /anomaly_map/{id}

Description:
Retrieve a visualization (PNG image) overlaying the detected anomaly map on top of the original image.

Example:

curl -o output.png http://127.0.0.1:8000/anomaly_map/7b32a8fc-6a9f-42a8-b8cb-ef30fcd30f3f

Returns:
An RGB PNG image with a heatmap overlay indicating anomalous regions.


---

🧠 Model & Inference

This API uses the PatchCore model from anomalib:

Backbone: resnet18

Layers used: layer2, layer3

Image size: 256×256

Preprocessing: central crop + resize

Thresholding: adjustable via threshold parameter


The model checkpoint is expected at:

model/model.ckpt


---

📓 Jupyter Notebooks

The notebooks/ directory includes scripts demonstrating:

Data preprocessing and augmentation (via AlbumentationsX)

Training a PatchCore model with Anomalib

Evaluating detection performance and visualizing results

Saving the trained model checkpoint


To open:

jupyter lab


---

⚙️ Dependencies

Core dependencies (see pyproject.toml):

Package	Purpose

anomalib>=2.1.0	Core anomaly detection framework
albumentationsx>=2.0.11	Image preprocessing & augmentation
jupyterlab>=4.4.7	Notebook interface
fastapi / uvicorn	REST API server
opencv-python, numpy, pillow, matplotlib	Image processing & visualization



---

🧪 Example Workflow

1. Train the model using notebooks in /notebooks


2. Save trained weights to model/model.ckpt


3. Run the API (locally or via Docker)


4. Upload images via /predict


5. Retrieve visual overlays via /anomaly_map/{id}




---

🧱 Project Notes

Each prediction is stored in memory (results_store) during runtime.

Anomaly heatmaps are normalized and colored using OpenCV’s COLORMAP_JET.

You can modify the center_crop_resize_bytes function for custom preprocessing.

The API is stateless across restarts — results are not persisted.



---

📄 License

MIT License © 2025 Vladimir Kupershtein


---

🔮 Future Improvements

Add persistent storage for inference results

Include endpoint for batch image uploads

Integrate automatic retraining or online learning

Publish prebuilt Docker image to Docker Hub



---

