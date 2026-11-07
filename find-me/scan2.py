"""Second pass: rebuild the references from "My photos", then rescan "All photos".

Score of a photo = mean of the 5 highest similarities to the references (more robust than the max).
Writes ref_emb2.npy and scan2.csv. Usage: python scan2.py "/path/From Google"
"""
import os, sys, csv, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from facelib import load, faces
here = os.path.dirname(os.path.abspath(__file__))
base = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PHOTOS_BASE", ".")  # folder containing "My photos" and "All photos"
exts = {".jpg", ".jpeg", ".png", ".heic"}
R0 = np.load(os.path.join(here, "ref_emb.npy"))
# 1) references: in each photo of "My photos", the face most similar to the initial references
refs = []
for n in sorted(os.listdir(os.path.join(base, "My photos"))):
    if os.path.splitext(n)[1].lower() not in exts: continue
    best = None
    for e, b, s, d in faces(load(os.path.join(base, "My photos", n))):
        sc = float((R0 @ e).max())
        if sc >= 0.45 and (best is None or sc > best[0]): best = (sc, e)
    if best: refs.append(best[1])
R = np.array(refs); np.save(os.path.join(here, "ref_emb2.npy"), R)
print("references:", len(R), flush=True)
# 2) scan: score = mean of the 5 highest similarities
src = os.path.join(base, "All photos")
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
            print("error", n, ex, flush=True); best = (-1, -1, 0)
        w.writerow([n, round(best[0], 3), round(best[1], 3), best[2]])
        if k % 500 == 0: f.flush(); print(k, "/", len(files), flush=True)
print("finished", flush=True)
