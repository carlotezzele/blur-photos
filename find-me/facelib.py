import numpy as np, cv2, os
import onnxruntime as ort
ort.preload_dlls()
from PIL import Image, ImageOps
import pillow_heif
pillow_heif.register_heif_opener()
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
                   providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))


def load(path, max_side=2000):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    im.thumbnail((max_side, max_side))
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def faces(img):
    out = []
    for f in app.get(img):
        e = f.normed_embedding
        x1, y1, x2, y2 = f.bbox
        out.append((e, f.bbox, min(x2 - x1, y2 - y1), f.det_score))
    return out
