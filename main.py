import tempfile
import uuid
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

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
    """Return anomaly map by yid as PNG."""
    if uid not in results_store:
        raise HTTPException(status_code=404, detail="ID not found")

    anomaly_map = results_store[uid]["anomaly_map"]

    # нормализуем [0,1] -> [0,255]
    anomaly_map = np.squeeze(anomaly_map)  # убираем лишние оси (1,H,W) -> (H,W)

    # нормализация в диапазон [0,255]
    anomaly_map_norm = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX) # type: ignore

    # перевод в uint8
    anomaly_map_uint8 = anomaly_map_norm.astype(np.uint8)

    # применяем цветовую карту
    anomaly_map_color = cv2.applyColorMap(anomaly_map_uint8, cv2.COLORMAP_JET)

    _, img_bytes = cv2.imencode(".png", anomaly_map_color)
    return Response(content=img_bytes.tobytes(), media_type="image/png")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

