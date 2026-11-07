import zipfile, glob, os, sys, zlib

src = sys.argv[1]
dst = os.path.join(src, "Tutte le foto")

have = set()
for name in os.listdir(dst):
    p = os.path.join(dst, name)
    crc = 0
    with open(p, "rb") as f:
        while chunk := f.read(1 << 20):
            crc = zlib.crc32(chunk, crc)
    have.add((crc, os.path.getsize(p)))

missing = 0
for z in sorted(glob.glob(os.path.join(src, "*.zip"))):
    with zipfile.ZipFile(z) as f:
        for i in f.infolist():
            if not i.is_dir() and (i.CRC, i.file_size) not in have:
                missing += 1
                print("MANCA:", z, i.filename)
print("file estratti verificati:", len(have), "| mancanti:", missing)
