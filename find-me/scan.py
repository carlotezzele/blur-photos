"""First pass: score every photo in a folder against your reference faces (ref_emb.npy).

Writes scan.csv: filename, best similarity, similarity to the mean reference, face size.
Resumable: photos already in scan.csv are skipped. Usage: python scan.py "/path/From Google/All photos"
"""
import os, sys, csv, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from facelib import load, faces

src = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))
R = np.load(os.path.join(here, "ref_emb.npy"))
m = R.mean(0); m /= np.linalg.norm(m)
out = os.path.join(here, "scan.csv")

done = set()
if os.path.exists(out):
    with open(out) as f:
        done = {row[0] for row in csv.reader(f)}

exts = {".jpg", ".jpeg", ".png", ".heic"}
files = sorted(n for n in os.listdir(src) if os.path.splitext(n)[1].lower() in exts and n not in done)
print("to analyse:", len(files), flush=True)
with open(out, "a", newline="") as f:
    w = csv.writer(f)
    for k, n in enumerate(files, 1):
        best = bmean = 0.0; size = 0
        try:
            for e, b, s, d in faces(load(os.path.join(src, n))):
                sc = float((R @ e).max())
                if sc > best:
                    best, bmean, size = sc, float(m @ e), int(s)
        except Exception as ex:
            print("error", n, ex, flush=True)
            best = -1
        w.writerow([n, round(best, 3), round(bmean, 3), size])
        if k % 200 == 0:
            f.flush(); print(k, "/", len(files), flush=True)
print("finished", flush=True)
