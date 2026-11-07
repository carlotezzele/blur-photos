import csv, os, sys, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from facelib import load
here=os.path.dirname(os.path.abspath(__file__)); src=os.path.join(sys.argv[1] if len(sys.argv)>1 else os.environ.get("PHOTOS_BASE","."),"Tutte le foto")
rows=sorted(([x[0],float(x[1]),float(x[2])] for x in csv.reader(open(os.path.join(here,"scan2.csv"))) if float(x[1])>=0.2), key=lambda r:-r[1])
print(len(rows))
for p in range(0,len(rows),30):
    tiles=[]
    for n,t,m in rows[p:p+30]:
        im=load(os.path.join(src,n),400); h,w=im.shape[:2]; s=300/max(h,w)
        im=cv2.resize(im,(int(w*s),int(h*s))); c=np.zeros((330,300,3),np.uint8); c[:im.shape[0],:im.shape[1]]=im
        cv2.putText(c,f"{p+len(tiles)} {t:.2f}/{m:.2f}",(4,322),0,0.55,(0,255,255),1); tiles.append(c)
    while len(tiles)%6: tiles.append(np.zeros((330,300,3),np.uint8))
    g=np.vstack([np.hstack(tiles[i:i+6]) for i in range(0,len(tiles),6)])
    cv2.imwrite(os.path.join(here,f"t_{p:03d}.jpg"),g)
with open(os.path.join(here,"cand.txt"),"w") as f:
    for i,(n,t,m) in enumerate(rows): f.write(f"{i}\t{n}\t{t}\t{m}\n")
