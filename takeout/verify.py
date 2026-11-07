"""Check (by CRC and size) that every file inside the Takeout zips exists in "All photos".

Run it before deleting the zips. Usage: python verify.py "/path/From Google"
"""
import zipfile, glob, os, sys, zlib

src = sys.argv[1]
dst = os.path.join(src, "All photos")

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
                print("MISSING:", z, i.filename)
print("extracted files checked:", len(have), "| missing:", missing)
