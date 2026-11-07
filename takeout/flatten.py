import zipfile, glob, os, sys, shutil, time

src = sys.argv[1]
dst = os.path.join(src, "Tutte le foto")
os.makedirs(dst, exist_ok=True)

seen = {}  # basename lower -> list of (crc, size)
used = set(n.lower() for n in os.listdir(dst))
written = skipped = 0
for z in sorted(glob.glob(os.path.join(src, "*.zip"))):
    with zipfile.ZipFile(z) as f:
        for i in f.infolist():
            if i.is_dir():
                continue
            base = os.path.basename(i.filename)
            key = base.lower()
            sig = (i.CRC, i.file_size)
            if sig in seen.get(key, []):
                skipped += 1
                continue
            seen.setdefault(key, []).append(sig)
            stem, ext = os.path.splitext(base)
            name, n = base, 0
            while name.lower() in used:
                n += 1
                name = f"{stem}_{n}{ext}"
            used.add(name.lower())
            out = os.path.join(dst, name)
            with f.open(i) as r, open(out, "wb") as w:
                shutil.copyfileobj(r, w, 1 << 20)
            t = time.mktime(i.date_time + (0, 0, -1))
            os.utime(out, (t, t))
            written += 1
    print("fatto:", os.path.basename(z), flush=True)
print("scritti", written, "duplicati identici saltati", skipped)
