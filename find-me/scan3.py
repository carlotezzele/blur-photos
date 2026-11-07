"""Third pass: refine the references again (top-5 mean against ref_emb2.npy) and rescan,
skipping photos already reviewed in cand.txt.

Writes ref_emb3.npy and scan3.csv. Usage: python scan3.py "/path/From Google"
"""
import os, sys, csv, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from facelib import load, faces
here = os.path.dirname(os.path.abspath(__file__))
base = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PHOTOS_BASE", ".")  # folder containing "My photos" and "All photos"
exts = {".jpg", ".jpeg", ".png", ".heic"}
R2 = np.load(os.path.join(here, "ref_emb2.npy"))
# references: in each photo of "My photos", the face most similar (top-5 mean) to the previous references
refs = []
for n in sorted(os.listdir(os.path.join(base, "My photos"))):
    if os.path.splitext(n)[1].lower() not in exts: continue
    fs = faces(load(os.path.join(base, "My photos", n)))
    if not fs: continue
    sc = [np.sort(R2 @ e)[::-1][:5].mean() for e, b, s, d in fs]
    refs.append(fs[int(np.argmax(sc))][0])
R = np.array(refs); np.save(os.path.join(here, "ref_emb3.npy"), R)
print("references:", len(R), flush=True)
# skip photos already reviewed and rejected
cand = os.path.join(here, "cand.txt")
seen = {l.split("\t")[1] for l in open(cand)} if os.path.exists(cand) else set()
src = os.path.join(base, "All photos")
files = sorted(n for n in os.listdir(src) if os.path.splitext(n)[1].lower() in exts and n not in seen)
print("to analyse:", len(files), flush=True)
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
            print("error", n, ex, flush=True); best = (-1, -1, 0)
        w.writerow([n, round(best[0], 3), round(best[1], 3), best[2]])
        if k % 500 == 0: f.flush(); print(k, "/", len(files), flush=True)
print("finished", flush=True)
