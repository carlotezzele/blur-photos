import os, sys, csv, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from facelib import load, faces
here = os.path.dirname(os.path.abspath(__file__))
base = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PHOTOS_BASE", ".")  # folder containing "Le mie foto" and "Tutte le foto"
exts = {".jpg", ".jpeg", ".png", ".heic"}
R2 = np.load(os.path.join(here, "ref_emb2.npy"))
# riferimenti: in ogni foto di "Le mie foto" il volto piu' simile (media top-5) ai 290 riferimenti
refs = []
for n in sorted(os.listdir(os.path.join(base, "Le mie foto"))):
    if os.path.splitext(n)[1].lower() not in exts: continue
    fs = faces(load(os.path.join(base, "Le mie foto", n)))
    if not fs: continue
    sc = [np.sort(R2 @ e)[::-1][:5].mean() for e, b, s, d in fs]
    refs.append(fs[int(np.argmax(sc))][0])
R = np.array(refs); np.save(os.path.join(here, "ref_emb3.npy"), R)
print("riferimenti:", len(R), flush=True)
# escludi le foto gia' controllate e scartate
cand = os.path.join(here, "cand.txt")
seen = {l.split("\t")[1] for l in open(cand)} if os.path.exists(cand) else set()
src = os.path.join(base, "Tutte le foto")
files = sorted(n for n in os.listdir(src) if os.path.splitext(n)[1].lower() in exts and n not in seen)
print("da analizzare:", len(files), flush=True)
with open(os.path.join(here, "scan3.csv"), "w", newline="") as f:
    w = csv.writer(f)
    for k, n in enumerate(files, 1):
        best = (0.0, 0.0, 0)
        try:
            for e, b, s, d in faces(load(os.path.join(src, n))):
                sims = np.sort(R @ e)[::-1]
                t5 = float(sims[:5].mean())
                if t5 > best[0]: best = (t5, float(sims[0]), int(s))
        except Exception as ex:
            print("errore", n, ex, flush=True); best = (-1, -1, 0)
        w.writerow([n, round(best[0], 3), round(best[1], 3), best[2]])
        if k % 500 == 0: f.flush(); print(k, "/", len(files), flush=True)
print("finito", flush=True)
