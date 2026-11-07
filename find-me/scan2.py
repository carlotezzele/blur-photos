import os, sys, csv, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from facelib import load, faces
here = os.path.dirname(os.path.abspath(__file__))
base = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PHOTOS_BASE", ".")  # folder containing "Le mie foto" and "Tutte le foto"
exts = {".jpg", ".jpeg", ".png", ".heic"}
R0 = np.load(os.path.join(here, "ref_emb.npy"))
# 1) riferimenti: in ogni foto di "Le mie foto" il volto piu' simile ai riferimenti iniziali
refs = []
for n in sorted(os.listdir(os.path.join(base, "Le mie foto"))):
    if os.path.splitext(n)[1].lower() not in exts: continue
    best = None
    for e, b, s, d in faces(load(os.path.join(base, "Le mie foto", n))):
        sc = float((R0 @ e).max())
        if sc >= 0.45 and (best is None or sc > best[0]): best = (sc, e)
    if best: refs.append(best[1])
R = np.array(refs); np.save(os.path.join(here, "ref_emb2.npy"), R)
print("riferimenti:", len(R), flush=True)
# 2) scansione: punteggio = media delle 5 somiglianze piu' alte
src = os.path.join(base, "Tutte le foto")
files = sorted(n for n in os.listdir(src) if os.path.splitext(n)[1].lower() in exts)
with open(os.path.join(here, "scan2.csv"), "w", newline="") as f:
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
