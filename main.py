import tempfile
import io
import uuid
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from anomalib.engine import Engine
from anomalib.models import Patchcore
import uvicorn

# === Initialize FastAPI ===
app = FastAPI(title="Anomaly Detection API")

# === Загружаем модель один раз при старте ===
ckpt_path = "notebooks/results/Patchcore/hackathon/latest/weights/lightning/model.ckpt"

pre_processor = Patchcore.configure_pre_processor(image_size=(256, 256))

model = Patchcore(
    backbone="resnet18",
    layers=["layer2", "layer3"],
    pre_trained=True,
    num_neighbors=9,
    pre_processor=pre_processor
)

engine = Engine()
model.eval()

# === Result store by ID ===
results_store = {}

# === Схема ответа ===
class PredictResponse(BaseModel):
    id: str
    score: float
    label: int


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    threshold: float = Form(0.5)
):
    """Upload image, run the model, return score/label/id."""

    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    # inference
    result = engine.predict(model=model, ckpt_path=ckpt_path, data_path=tmp_path)

    if result is None:
        raise HTTPException(status_code=404, detail="Prediction failed")    

    anomaly_map = result[0].anomaly_map.cpu().numpy() # type: ignore
    score = float(result[0].pred_score.item()) # type: ignore
    label = int(result[0].pred_score.item() > threshold) # type: ignore    

    # сохраняем в store
    uid = str(uuid.uuid4())
    results_store[uid] = {
        "score": score,
        "label": label,
        "anomaly_map": anomaly_map
    }

    return PredictResponse(id=uid, score=score, label=label)


@app.get("/anomaly_map/{uid}")
async def get_anomaly_map(uid: str):
    """Return anomaly map by uid as PNG (без OpenCV)."""
    if uid not in results_store:
        raise HTTPException(status_code=404, detail="ID not found")

    anomaly_map = results_store[uid]["anomaly_map"]

    # Убираем лишние оси
    anomaly_map = np.squeeze(anomaly_map)

    # Нормализация [0,1] -> [0,255]
    anomaly_map_norm = (255 * (anomaly_map - anomaly_map.min()) /
                        (anomaly_map.max() - anomaly_map.min() + 1e-8))
    anomaly_map_uint8 = anomaly_map_norm.astype(np.uint8)

    # Применяем colormap через matplotlib
    cmap = plt.get_cmap("jet")
    anomaly_map_color = cmap(anomaly_map_uint8 / 255.0)  # RGBA в [0,1]
    anomaly_map_rgb = (anomaly_map_color[:, :, :3] * 255).astype(np.uint8)

    # В Pillow -> PNG в память
    img = Image.fromarray(anomaly_map_rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    return Response(content=buf.getvalue(), media_type="image/png")
