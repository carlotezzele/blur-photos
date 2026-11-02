"""Hides the people who aren't you.

Start:  python app.py "/path/to/the/folder/with/the/photo/subfolders"
Then open http://localhost:5055 (it opens by itself).
Originals are never modified: results go to "<folder>/Obscured photos".
"""
import io, os, sys, shutil, threading, subprocess, webbrowser
import numpy as np, cv2
import onnxruntime as ort
if hasattr(ort, "preload_dlls"):
    ort.preload_dlls()
from PIL import Image, ImageOps
import pillow_heif
pillow_heif.register_heif_opener()
from flask import Flask, jsonify, request, send_file, abort
from insightface.app import FaceAnalysis
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("OSCURA_BASE", "."))
OUT = os.environ.get("OSCURA_OUT", os.path.join(BASE, "Obscured photos"))
EXTS = {".jpg", ".jpeg", ".png", ".heic"}
WORK_SIDE = 2000          # working resolution for analysis and preview
PORT = 5055

print("loading models...", flush=True)
face_app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
                        providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))
seg = YOLO(os.path.join(HERE, "yolo11m-seg.pt"))     # downloaded automatically on first launch
if not os.path.exists(os.path.join(HERE, "ref_emb.npy")):
    sys.exit("ref_emb.npy is missing: create it first with  python make_ref.py <folder of photos of you>")
REF = np.load(os.path.join(HERE, "ref_emb.npy"))
model_lock = threading.Lock()
cache, cache_lock = {}, threading.Lock()


def folder_path(folder):
    p = os.path.realpath(os.path.join(BASE, folder))
    if os.path.dirname(p) != os.path.realpath(BASE) or not os.path.isdir(p):
        abort(400)
    return p


def list_images(folder):
    return sorted(n for n in os.listdir(folder_path(folder)) if os.path.splitext(n)[1].lower() in EXTS)


def out_name(name):
    stem, ext = os.path.splitext(name)
    return name if ext.lower() in {".jpg", ".jpeg", ".png"} else stem + ".jpg"


def open_full(path):
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def similarity(e):
    s = np.sort(REF @ e)[::-1]
    return float(s[:5].mean())


def contours(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        if cv2.contourArea(c) < 20:
            continue
        c = cv2.approxPolyDP(c, 1.2, True)
        out.append(c.reshape(-1).astype(int).tolist())
    return out


def analyze(folder, name):
    key = (folder, name)
    with cache_lock:
        if key in cache:
            return cache[key]
    with model_lock:
        with cache_lock:
            if key in cache:
                return cache[key]
        im = open_full(os.path.join(folder_path(folder), name))
        im.thumbnail((WORK_SIDE, WORK_SIDE))
        rgb = np.asarray(im)
        h, w = rgb.shape[:2]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        r = seg(bgr, classes=[0], imgsz=1024, retina_masks=True, conf=0.25, verbose=False, device="cpu")[0]
        masks = [] if r.masks is None else [m > 0.5 for m in r.masks.data.cpu().numpy()]
        masks = [m for m in masks if m.sum() > h * w * 0.0003]

        items = [{"polys": contours(m), "score": None} for m in masks]
        for f in face_app.get(bgr):
            sc = similarity(f.normed_embedding)
            x1, y1, x2, y2 = f.bbox
            cx, cy = int(np.clip((x1 + x2) / 2, 0, w - 1)), int(np.clip((y1 + y2) / 2, 0, h - 1))
            owner = [i for i, m in enumerate(masks) if m[cy, cx]]
            if owner:
                # if two people overlap, pick the smallest mask that contains the face
                i = min(owner, key=lambda i: masks[i].sum())
                if items[i]["score"] is None or sc > items[i]["score"]:
                    items[i]["score"] = sc
            else:
                # face with no detected body: ellipse around the head
                m = np.zeros((h, w), np.uint8)
                ax, ay = (x2 - x1) * 0.8, (y2 - y1) * 0.9
                cv2.ellipse(m, (cx, cy), (int(ax), int(ay)), 0, 0, 360, 1, -1)
                items.append({"polys": contours(m), "score": sc})
        for i, it in enumerate(items):
            it["id"] = i
            it["score"] = None if it["score"] is None else round(it["score"], 3)

        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90)
        res = {"w": w, "h": h, "items": items, "jpg": buf.getvalue()}
    with cache_lock:
        cache[key] = res
        if len(cache) > 30:
            cache.pop(next(iter(cache)))
    return res


app = Flask(__name__, static_folder=None)


@app.get("/")
def index():
    return send_file(os.path.join(HERE, "index.html"))


@app.get("/api/folders")
def folders():
    fs = sorted(d for d in os.listdir(BASE) if os.path.isdir(os.path.join(BASE, d)) and d != "Obscured photos")
    return jsonify(fs)


@app.get("/api/list")
def api_list():
    folder = request.args["folder"]
    done = set(os.listdir(OUT)) if os.path.isdir(OUT) else set()
    return jsonify([{"name": n, "done": out_name(n) in done} for n in list_images(folder)])


@app.get("/api/analyze")
def api_analyze():
    res = analyze(request.args["folder"], request.args["name"])
    return jsonify({k: v for k, v in res.items() if k != "jpg"})


@app.get("/api/prefetch")
def api_prefetch():
    threading.Thread(target=analyze, args=(request.args["folder"], request.args["name"]), daemon=True).start()
    return "", 204


@app.get("/img")
def img():
    res = analyze(request.args["folder"], request.args["name"])
    return send_file(io.BytesIO(res["jpg"]), mimetype="image/jpeg")


@app.post("/api/save")
def api_save():
    folder, name = request.form["folder"], request.form["name"]
    if name not in list_images(folder):
        abort(400)
    src = os.path.join(folder_path(folder), name)
    os.makedirs(OUT, exist_ok=True)
    dst = os.path.join(OUT, out_name(name))
    st = os.stat(src)

    alpha = np.asarray(Image.open(request.files["mask"].stream).convert("RGBA"))[:, :, 3]
    if alpha.max() == 0 and out_name(name) == name:
        shutil.copy2(src, dst)          # nothing to cover: identical copy
        return jsonify(ok=True, copied=True)

    orig = Image.open(src)
    exif = orig.getexif()
    full = ImageOps.exif_transpose(orig).convert("RGB")
    img = np.asarray(full).astype(np.float32)
    H, W = img.shape[:2]
    m = cv2.resize(alpha, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
    feather = max(1.0, max(W, H) / 1500)
    m = cv2.GaussianBlur(m, (0, 0), feather)[:, :, None]

    if request.form["mode"] == "blur":
        sigma = float(request.form["strength"]) * max(W, H) / 400
        f = 8
        small = cv2.resize(img, (max(1, W // f), max(1, H // f)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigma / f, borderType=cv2.BORDER_REFLECT)
        eff = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        c = request.form["color"].lstrip("#")
        eff = np.empty_like(img)
        eff[:] = [int(c[i:i + 2], 16) for i in (0, 2, 4)]
    out = np.clip(img * (1 - m) + eff * m, 0, 255).astype(np.uint8)

    res = Image.fromarray(out)
    if dst.lower().endswith(".png"):
        res.save(dst, "PNG")
    else:
        exif[0x0112] = 1        # the image is already upright
        res.save(dst, "JPEG", quality=95, exif=exif.tobytes())
    os.utime(dst, (st.st_atime, st.st_mtime))
    return jsonify(ok=True, copied=False)


def open_browser(url):
    if shutil.which("cmd.exe"):         # WSL: open the Windows browser
        subprocess.Popen(["cmd.exe", "/c", "start", "", url], cwd="/mnt/c",
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        webbrowser.open(url)


if __name__ == "__main__":
    url = f"http://localhost:{PORT}"
    print("folder:", BASE, "\nready:", url, flush=True)
    if not os.environ.get("OSCURA_NOBROWSER"):
        threading.Timer(1.5, open_browser, args=(url,)).start()
    app.run(host="127.0.0.1", port=PORT, threaded=True)
