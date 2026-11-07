"""Gallery of "All photos" with women blurred automatically, served on the local network.

Start:  python gallery/server.py "/path/From Google"
The analysis goes to gallery/cache and corrections to gallery/corrections.json.
At every start the page asks where exported photos go (changeable at any time from the header):
  - "folder":  originals untouched, blurred copies in "From Google/Blurred photos"
  - "replace": blurred photos overwrite the originals in "All photos"; a pristine copy of each
               original is kept in "From Google/Originals (backup)/All photos", so corrections and un-blurring still work.
"""
import io, os, json, secrets, shutil, threading, queue, time
import numpy as np, cv2
import onnxruntime as ort
ort.preload_dlls()
from PIL import Image, ImageOps
import pillow_heif
pillow_heif.register_heif_opener()
from flask import Flask, jsonify, request, send_file, abort, make_response
from insightface.app import FaceAnalysis
from ultralytics import YOLO
import torch
torch.set_num_threads(4)        # leave some CPU free for the web server

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
BASE = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PHOTOS_BASE", ".")   # folder containing "All photos"
SRC = os.path.join(BASE, "All photos")
TRASH = os.path.join(BASE, "To delete")
MINE = os.path.join(BASE, "My photos")
DESTS = {"del": TRASH, "mine": MINE}
OUT = os.environ.get("GALLERY_OUT", os.path.join(BASE, "Blurred photos"))
BACKUP = os.path.join(BASE, "Originals (backup)", os.path.basename(SRC))     # same layout as app.py: backups are shared
VIDEO_EXTS = {".mov", ".mp4"}         # Live Photo videos, moved together with their photo
CACHE = os.path.join(HERE, "cache")
EXTS = {".jpg", ".jpeg", ".png", ".heic"}
SIDE = 1600
PORT = 5056
os.makedirs(CACHE, exist_ok=True)

tok_file = os.path.join(HERE, "token.txt")
if not os.path.exists(tok_file):
    open(tok_file, "w").write(secrets.token_urlsafe(9))
TOKEN = open(tok_file).read().strip()

print("loading models...", flush=True)
gpu = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
face_app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"], providers=gpu)
face_app.prepare(ctx_id=0, det_size=(640, 640))
_yolo = os.path.join(HERE, "yolo11m-seg.pt")
seg = YOLO(_yolo if os.path.exists(_yolo) else "yolo11m-seg.pt")    # downloaded on first start
clip = ort.InferenceSession(os.path.join(HERE, "clip_vitl14.onnx"), providers=gpu)
CLIP_TEXT = np.load(os.path.join(HERE, "clip_text.npy"))      # [female, male]
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
UNSURE = (0.3, 0.7)     # P(female) inside this range = uncertain
model_lock = threading.Lock()
fix_lock = threading.Lock()
FIX_FILE = os.path.join(HERE, "corrections.json")     # name -> {person index: "F"/"M"}
EXP_FILE = os.environ.get("GALLERY_EXP", os.path.join(HERE, "exported.json"))        # name -> {"v", "u", "out"} of the last export


def load_json(path):
    try:
        return json.load(open(path))
    except Exception:
        return {}


def save_json(path, data):
    tmp = path + ".tmp"
    json.dump(data, open(tmp, "w"), indent=0)
    os.replace(tmp, path)


SETTINGS_FILE = os.path.join(HERE, "settings.json")
SETTINGS = load_json(SETTINGS_FILE)
ASKED = False           # the output choice is asked again at every start
FIXES = load_json(FIX_FILE)
EXPORTED = load_json(EXP_FILE)

NAMES = sorted(n for n in os.listdir(SRC) if os.path.splitext(n)[1].lower() in EXTS)
NAMESET = set(NAMES)


def original(name):
    """The untouched original: the backup if the photo was already replaced in place."""
    bk = os.path.join(BACKUP, name)
    return bk if os.path.exists(bk) else os.path.join(SRC, name)


def paths(name):
    b = os.path.join(CACHE, name)
    return b + ".jpg", b + ".lab.png", b + ".json"


INFO = {}      # name -> counts {"F","M","U"} for photos already analysed


def eff(p):
    """Effective sex: the user's correction wins over the estimate."""
    return p.get("fix", p["sex"])


def count(name):
    try:
        d = json.load(open(paths(name)[2]))
        p = d["people"]
    except Exception:
        return
    INFO[name] = {"F": sum(eff(x) == "F" for x in p), "M": sum(eff(x) == "M" for x in p),
                  "U": sum(eff(x) is None for x in p), "fix": sum("fix" in x for x in p), "v": d.get("v", 0),
                  "err": bool(d.get("error"))}


def is_done(name):
    return name in INFO


def female_prob(rgb, masks):
    """P(female) for each person: CLIP looks at the whole silhouette (hair, body, clothes) on a grey background."""
    if not masks:
        return []
    batch = []
    for m in masks:
        ys, xs = np.where(m)
        y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        c = rgb[y1:y2, x1:x2].copy()
        c[~m[y1:y2, x1:x2]] = 128
        side = max(c.shape[:2])
        sq = np.full((side, side, 3), 128, np.uint8)
        oy, ox = (side - c.shape[0]) // 2, (side - c.shape[1]) // 2
        sq[oy:oy + c.shape[0], ox:ox + c.shape[1]] = c
        sq = cv2.resize(sq, (224, 224), interpolation=cv2.INTER_AREA if side > 224 else cv2.INTER_CUBIC)
        batch.append(((sq.astype(np.float32) / 255 - CLIP_MEAN) / CLIP_STD).transpose(2, 0, 1))
    out = []
    for i in range(0, len(batch), 16):
        e = clip.run(None, {"x": np.stack(batch[i:i + 16])})[0]
        logit = 100 * e @ CLIP_TEXT.T
        logit -= logit.max(1, keepdims=True)
        p = np.exp(logit)
        out += (p[:, 0] / p.sum(1)).tolist()
    return out


def analyze(name):
    """People (masks) + sex estimated by CLIP. Saves preview, label map and json."""
    jpg_p, lab_p, js_p = paths(name)
    with model_lock:
        if is_done(name):
            return
        try:
            im = ImageOps.exif_transpose(Image.open(original(name))).convert("RGB")
        except Exception as e:
            json.dump({"error": str(e), "people": []}, open(js_p, "w"))
            count(name)
            return
        im.thumbnail((SIDE, SIDE))
        rgb = np.asarray(im)
        h, w = rgb.shape[:2]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        r = seg(bgr, classes=[0], imgsz=1024, retina_masks=True, conf=0.25, verbose=False, device="cpu")[0]
        masks = [] if r.masks is None else [m > 0.5 for m in r.masks.data.cpu().numpy()]
        masks = [m for m in masks if m.sum() > h * w * 0.0003]
        for f in face_app.get(bgr):
            x1, y1, x2, y2 = f.bbox
            cx, cy = int(np.clip((x1 + x2) / 2, 0, w - 1)), int(np.clip((y1 + y2) / 2, 0, h - 1))
            if not any(m[cy, cx] for m in masks):
                # face with no detected body: ellipse around the head
                m = np.zeros((h, w), np.uint8)
                cv2.ellipse(m, (cx, cy), (int((x2 - x1) * 0.8), int((y2 - y1) * 0.9)), 0, 0, 360, 1, -1)
                masks.append(m.astype(bool))
        pf = female_prob(rgb, masks)
        people = [{"pf": round(p, 3), "sex": "F" if p >= UNSURE[1] else "M" if p < UNSURE[0] else None} for p in pf]
        for i, sex in FIXES.get(name, {}).items():      # re-apply corrections if the cache was deleted
            if int(i) < len(people) and sex != people[int(i)]["sex"]:
                people[int(i)]["fix"] = sex

        # label map: 0 = background, i+1 = person i; smaller masks win where they overlap
        lab = np.zeros((h, w), np.uint8)
        for i in sorted(range(len(masks)), key=lambda i: -masks[i].sum())[:254]:
            lab[masks[i]] = i + 1
        im.save(jpg_p, "JPEG", quality=88)
        cv2.imwrite(lab_p, lab)
        json.dump({"w": w, "h": h, "people": people, "v": len(FIXES.get(name, {}))}, open(js_p, "w"))
    count(name)


# --- queue: analyse everything in the background, photos requested by the browser jump the queue ---
urgent = queue.LifoQueue()


def worker():
    rest = iter(NAMES)
    while True:
        try:
            name = urgent.get_nowait()
        except queue.Empty:
            name = next(rest, None)
            if name is None:
                time.sleep(1)
                continue
        if name in NAMESET and not is_done(name):
            try:
                analyze(name)
            except Exception as e:
                print("error", name, e, flush=True)


def selected(info, unknown):
    return [i + 1 for i, p in enumerate(info["people"]) if eff(p) == "F" or (unknown and eff(p) is None)]


def blur_people(img, lab, sel):
    """Blur the people whose label is in sel. lab is resized to the resolution of img."""
    H, W = img.shape[:2]
    if lab.shape[:2] != (H, W):
        lab = cv2.resize(lab, (W, H), interpolation=cv2.INTER_NEAREST)
    m = np.isin(lab, sel).astype(np.float32)
    m = cv2.GaussianBlur(m, (0, 0), max(1.0, max(W, H) / 1500))[:, :, None]
    sigma = 7 * max(W, H) / 400
    small = cv2.resize(img, (max(1, W // 8), max(1, H // 8)), interpolation=cv2.INTER_AREA).astype(np.float32)
    small = cv2.GaussianBlur(small, (0, 0), sigma / 8, borderType=cv2.BORDER_REFLECT)
    soft = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    return np.clip(img * (1 - m) + soft * m, 0, 255).astype(np.uint8)


def render(name, unknown, full, orig):
    jpg_p, lab_p, js_p = paths(name)
    thumb_p = f"{jpg_p[:-4]}.t{int(unknown)}{int(orig)}.jpg"
    if not full and os.path.exists(thumb_p):
        return open(thumb_p, "rb").read()
    info = json.load(open(js_p))
    if info.get("error"):
        abort(404)
    img = cv2.imread(jpg_p)
    sel = [] if orig else selected(info, unknown)
    lab = cv2.imread(lab_p, cv2.IMREAD_UNCHANGED) if sel else None
    if not full:
        # thumbnail: downscale before blurring, much cheaper
        s = 400 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if sel:
            lab = cv2.resize(lab, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    if sel:
        img = blur_people(img, lab, sel)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85 if full else 75])
    if not full:
        open(thumb_p, "wb").write(buf.tobytes())
    return buf.tobytes()


# --- full-resolution export: to "Blurred photos" or in place ---
LOWER = {n.lower() for n in NAMES}
export_lock = threading.Lock()
EXPORT = {"running": False, "done": 0, "todo": 0, "errors": 0}


def out_name(name):
    stem, ext = os.path.splitext(name)
    if ext.lower() in {".jpg", ".jpeg", ".png"}:
        return name
    # HEIC -> JPG; if a .jpg with the same name already exists, don't overwrite it
    return stem + ("-heic.jpg" if (stem + ".jpg").lower() in LOWER else ".jpg")


def export_one(name, unknown, mode):
    jpg_p, lab_p, js_p = paths(name)
    info = json.load(open(js_p))
    here, bk = os.path.join(SRC, name), os.path.join(BACKUP, name)
    if mode == "replace":
        if not os.path.exists(bk):          # keep the untouched original before the first overwrite
            os.makedirs(BACKUP, exist_ok=True)
            shutil.copy2(here, bk)
        src, dst, out = bk, here, name
    else:
        if os.path.exists(bk):              # switched back from "replace": put the original back
            shutil.copy2(bk, here)
            os.remove(bk)
        os.makedirs(OUT, exist_ok=True)
        src, dst, out = here, os.path.join(OUT, out_name(name)), out_name(name)
    st = os.stat(src)
    sel = selected(info, unknown)
    if not sel and out_name(name) == out:
        shutil.copy2(src, dst)              # nothing to blur: identical copy
    elif not sel and mode == "replace":
        shutil.copy2(src, dst)              # nothing (left) to blur: restore the original
    else:
        orig = Image.open(src)
        exif = orig.getexif()
        rgb = np.asarray(ImageOps.exif_transpose(orig).convert("RGB"))
        if sel:
            bgr = blur_people(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cv2.imread(lab_p, cv2.IMREAD_UNCHANGED), sel)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = Image.fromarray(rgb)
        exif[0x0112] = 1                    # the image is already upright
        ext, tmp = os.path.splitext(dst)[1].lower(), dst + ".tmp"
        if ext == ".png":
            res.save(tmp, "PNG")
        elif ext == ".heic":
            res.save(tmp, "HEIF", quality=95, exif=exif.tobytes())
        else:
            res.save(tmp, "JPEG", quality=95, exif=exif.tobytes())
        os.replace(tmp, dst)                # never leave a half-written photo
        os.utime(dst, (st.st_atime, st.st_mtime))
    EXPORTED[name] = {"v": info.get("v", 0), "u": int(unknown), "mode": mode, "out": out}


def mode_now():
    return SETTINGS.get("output") if ASKED else None


def up_to_date(name, unknown, mode):
    e = EXPORTED.get(name)
    return (e is not None and e["v"] == INFO[name]["v"] and e["u"] == int(unknown)
            and e.get("mode", "folder") == mode)


def present(name, mode):
    """The exported result is still where it should be."""
    if mode == "replace":
        return os.path.exists(os.path.join(BACKUP, name))
    return os.path.exists(os.path.join(OUT, out_name(name)))


def exportable():
    return [n for n in NAMES if n in NAMESET and n in INFO and not INFO[n]["err"]]


def export_job(unknown, mode):
    todo = [n for n in exportable() if not up_to_date(n, unknown, mode) or not present(n, mode)]
    EXPORT.update(done=0, todo=len(todo), errors=0)
    try:
        for i, n in enumerate(todo):
            if n not in NAMESET:
                continue
            try:
                export_one(n, unknown, mode)
            except Exception as e:
                EXPORT["errors"] += 1
                print("export error", n, e, flush=True)
            EXPORT["done"] = i + 1
            if i % 20 == 0:
                save_json(EXP_FILE, EXPORTED)
    finally:
        save_json(EXP_FILE, EXPORTED)
        EXPORT["running"] = False


def drop_export(name):
    """The photo was moved out of "All photos": remove its blurred copy too."""
    e = EXPORTED.pop(name, None)
    if e and e.get("mode", "folder") == "folder" and os.path.exists(os.path.join(OUT, e["out"])):
        os.remove(os.path.join(OUT, e["out"]))
    save_json(EXP_FILE, EXPORTED)


def restore_original(name):
    """If the photo was replaced in place, put the untouched original back (so moves carry the original)."""
    bk = os.path.join(BACKUP, name)
    if os.path.exists(bk):
        shutil.copy2(bk, os.path.join(SRC, name))
        os.remove(bk)


app = Flask(__name__, static_folder=None)


@app.before_request
def check_token():
    if request.args.get("k") == TOKEN or request.cookies.get("k") == TOKEN:
        return
    abort(403)


@app.after_request
def set_cookie(resp):
    if request.args.get("k") == TOKEN:
        resp.set_cookie("k", TOKEN, max_age=30 * 86400, samesite="Strict", httponly=True)
    return resp


@app.get("/")
def index():
    return send_file(os.path.join(HERE, "index.html"))


@app.get("/api/list")
def api_list():
    return jsonify([n for n in NAMES if n in NAMESET])


def companions(name):
    """The photo plus any Live Photo video with the same name."""
    stem = os.path.splitext(name)[0]
    return [name] + sorted(f for f in os.listdir(SRC)
                           if os.path.splitext(f)[0] == stem and os.path.splitext(f)[1].lower() in VIDEO_EXTS)


move_lock = threading.Lock()


@app.post("/api/trash")
def api_trash():
    """Move the original (and its Live Photo video) to "To delete" or "My photos". Never deletes anything."""
    d = request.get_json()
    name, dest = d["name"], DESTS.get(d.get("to", "del"))
    if dest is None:
        abort(400)
    with move_lock:
        if name not in NAMESET or not os.path.exists(os.path.join(SRC, name)):
            abort(400)
        os.makedirs(dest, exist_ok=True)
        files = companions(name)
        if any(os.path.exists(os.path.join(dest, f)) for f in files):
            abort(409)          # a file with the same name already exists: don't overwrite
        restore_original(name)
        for f in files:
            shutil.move(os.path.join(SRC, f), os.path.join(dest, f))
        NAMESET.discard(name)
        drop_export(name)
    with open(os.path.join(HERE, "moved.log"), "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\tmoved to {os.path.basename(dest)}\t{' | '.join(files)}\n")
    return jsonify(ok=True, files=files)


@app.post("/api/untrash")
def api_untrash():
    """Undo: put the files back in "All photos"."""
    d = request.get_json()
    name, files, dest = d["name"], d["files"], DESTS.get(d.get("to", "del"))
    if dest is None:
        abort(400)
    with move_lock:
        if name not in NAMES or name in NAMESET or files[0] != name:
            abort(400)
        for f in files:
            if os.path.basename(f) != f or os.path.exists(os.path.join(SRC, f)) \
                    or not os.path.exists(os.path.join(dest, f)):
                abort(409)
        for f in files:
            shutil.move(os.path.join(dest, f), os.path.join(SRC, f))
        NAMESET.add(name)
    with open(os.path.join(HERE, "moved.log"), "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\trestored from {os.path.basename(dest)}\t{' | '.join(files)}\n")
    return jsonify(ok=True)


@app.get("/api/status")
def api_status():
    for n in request.args.get("want", "").split("|"):
        if n in NAMESET and not is_done(n):
            urgent.put(n)
    unknown = request.args.get("u", "1") == "1"
    mode = mode_now()
    stale = sum(1 for n in exportable() if not up_to_date(n, unknown, mode)) if mode else 0
    return jsonify(info={n: c for n, c in INFO.items() if n in NAMESET}, total=len(NAMESET),
                   export=dict(EXPORT, stale=stale, mode=mode))


@app.get("/api/settings")
def get_settings():
    return jsonify(output=SETTINGS.get("output"), asked=ASKED,
                   out=os.path.basename(OUT), src=os.path.basename(SRC),
                   backup=os.path.relpath(BACKUP, BASE).replace(os.sep, "/"))


@app.post("/api/settings")
def set_settings():
    global ASKED
    out = (request.get_json() or {}).get("output")
    if out not in ("folder", "replace") or EXPORT["running"]:
        abort(400 if out not in ("folder", "replace") else 409)
    SETTINGS["output"] = out
    ASKED = True
    save_json(SETTINGS_FILE, SETTINGS)
    return get_settings()


@app.post("/api/export")
def api_export():
    """Export (or update) every analysed photo, with corrections applied, according to the output setting."""
    unknown = bool(request.get_json().get("u", 1))
    mode = mode_now()
    if mode is None:
        abort(409)                          # output not chosen yet in this run
    with export_lock:
        if not EXPORT["running"]:
            EXPORT["running"] = True
            threading.Thread(target=export_job, args=(unknown, mode), daemon=True).start()
    return jsonify(EXPORT)


@app.get("/api/people")
def api_people():
    """Outlines of the people in a photo, used to mark errors."""
    name = request.args.get("name", "")
    if name not in NAMESET or not is_done(name):
        abort(404)
    jpg_p, lab_p, js_p = paths(name)
    info = json.load(open(js_p))
    lab = cv2.imread(lab_p, cv2.IMREAD_UNCHANGED)
    out = []
    for i, p in enumerate(info["people"]):
        cs, _ = cv2.findContours((lab == i + 1).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys = [cv2.approxPolyDP(c, 1.5, True).reshape(-1).tolist() for c in cs if cv2.contourArea(c) >= 20]
        if polys:
            out.append({"id": i, "pf": p["pf"], "sex": eff(p), "fixed": "fix" in p, "polys": polys})
    return jsonify(w=info["w"], h=info["h"], people=out)


@app.post("/api/fix")
def api_fix():
    """Mark an error: sex = "F" (blur), "M" (keep visible) or null (back to the estimate)."""
    d = request.get_json()
    name, i, sex = d["name"], int(d["id"]), d.get("sex")
    if name not in NAMESET or not is_done(name) or sex not in ("F", "M", None):
        abort(400)
    jpg_p, lab_p, js_p = paths(name)
    with fix_lock:
        info = json.load(open(js_p))
        p = info["people"][i]
        fx = FIXES.setdefault(name, {})
        if sex is None or sex == p["sex"]:
            p.pop("fix", None)
            fx.pop(str(i), None)
        else:
            p["fix"] = sex
            fx[str(i)] = sex
        if not fx:
            FIXES.pop(name)
        save_json(FIX_FILE, FIXES)
        info["v"] = info.get("v", 0) + 1
        json.dump(info, open(js_p, "w"))
        for t in ("00", "01", "10", "11"):
            if os.path.exists(f"{jpg_p[:-4]}.t{t}.jpg"):
                os.remove(f"{jpg_p[:-4]}.t{t}.jpg")
        count(name)
    with open(os.path.join(HERE, "corrections.log"), "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{name}\t{i}\testimate={p['sex']}\tpf={p['pf']}\tcorrected={sex}\n")
    return jsonify(INFO[name])


@app.get("/img")
def img():
    name = request.args.get("name", "")
    if name not in NAMESET or not is_done(name):
        abort(404)
    a = request.args
    data = render(name, a.get("u") == "1", a.get("full") == "1", a.get("o") == "1")
    resp = make_response(data)
    resp.headers["Content-Type"] = "image/jpeg"
    resp.headers["Cache-Control"] = "private, max-age=3600"
    return resp


if __name__ == "__main__":
    for n in NAMES:
        if os.path.exists(paths(n)[2]):
            count(n)
            # corrections made before the separate file existed: copy them there
            for i, p in enumerate(json.load(open(paths(n)[2]))["people"]):
                if "fix" in p:
                    FIXES.setdefault(n, {})[str(i)] = p["fix"]
    save_json(FIX_FILE, FIXES)
    threading.Thread(target=worker, daemon=True).start()
    print(f"ready: http://<ip>:{PORT}/?k={TOKEN}", flush=True)
    app.run(host="0.0.0.0", port=PORT, threaded=True)
