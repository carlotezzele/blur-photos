"""Crea ref_emb.npy, l'impronta del tuo volto usata da app.py.

Uso:  python make_ref.py <cartella con foto in cui ci sei tu>

In ogni foto prende il volto piu' grande, poi tiene solo quelli coerenti tra loro
(cosi' le foto in cui il volto piu' grande non e' il tuo vengono scartate).
Bastano 10-20 foto; piu' sono varie (luce, eta', angolazione) meglio funziona.
"""
import os, sys, numpy as np, cv2
import onnxruntime as ort
if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()
from PIL import Image, ImageOps
import pillow_heif
pillow_heif.register_heif_opener()
from insightface.app import FaceAnalysis

src = sys.argv[1]
app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
                   providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

embs = []
for n in sorted(os.listdir(src)):
    if os.path.splitext(n)[1].lower() not in {".jpg", ".jpeg", ".png", ".heic"}:
        continue
    im = ImageOps.exif_transpose(Image.open(os.path.join(src, n))).convert("RGB")
    im.thumbnail((2000, 2000))
    fs = app.get(cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR))
    if fs:
        f = max(fs, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        embs.append(f.normed_embedding)
E = np.array(embs)
if len(E) < 3:
    sys.exit("servono almeno 3 foto con un volto riconoscibile")

keep = np.ones(len(E), bool)
for _ in range(5):
    m = E[keep].mean(0); m /= np.linalg.norm(m)
    keep = E @ m >= 0.4
np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref_emb.npy"), E[keep])
print(f"volti trovati: {len(E)}, tenuti come riferimento: {keep.sum()}")
