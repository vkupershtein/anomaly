import tempfile
import io
import uuid
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from anomalib.engine import Engine
from anomalib.models import Patchcore

# === Initialize FastAPI ===
app = FastAPI(title="Anomaly Detection API")

# === Load model ===
ckpt_path = "model/model.ckpt"

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

def center_crop_resize_bytes(image_bytes: bytes, size: int = 256) -> np.ndarray:
    # read from bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Does not work")

    h, w = image.shape[:2]
    min_dim = min(h, w)

    # centere coordinates
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    cropped = image[start_y:start_y + min_dim, start_x:start_x + min_dim]

    # resize в (size,size)
    resized = cv2.resize(cropped, (size, size))
    return resized


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    threshold: float = Form(0.3)
):
    """Upload image, run the model, return score/label/id."""

    contents = await file.read()
    img = center_crop_resize_bytes(contents)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
        cv2.imwrite(tmp_path, img)      

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
        "score": score * 1.67,
        "label": label,
        "anomaly_map": anomaly_map,
        "image": img
    }

    return PredictResponse(id=uid, score=score, label=label)


@app.get("/anomaly_map/{uid}")
async def get_anomaly_map(uid: str):
    """Return anomaly map overlay on image."""
    if uid not in results_store:
        raise HTTPException(status_code=404, detail="ID not found")

    anomaly_map = results_store[uid]["anomaly_map"]
    base_img = results_store[uid]["image"]

    # Normalized [0,255]
    anomaly_map = np.squeeze(anomaly_map)
    anomaly_map_norm = cv2.normalize(anomaly_map, None, 0, 255, cv2.NORM_MINMAX) # type: ignore
    anomaly_map_uint8 = anomaly_map_norm.astype(np.uint8)

    # Colored
    anomaly_map_color = cv2.applyColorMap(anomaly_map_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(base_img, 0.6, anomaly_map_color, 0.4, 0)

    _, img_bytes = cv2.imencode(".png", overlay)
    return Response(content=img_bytes.tobytes(), media_type="image/png")
