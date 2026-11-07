# Hide People

A local program that **blurs or covers with a silhouette** every person in a photo except you.

- It recognises your face (InsightFace) and finds the exact outline of every person (YOLO11 segmentation).
- **Automatic mode**: prepares the cover for everyone else and asks you to confirm photo by photo.
- **Manual mode**: start from scratch and pick who to cover with a click, or with brush and eraser.
- Two effects: **blur** (adjustable strength) or a solid-colour **silhouette** with the same shape as the person.
- Originals are never modified. Results go to `Obscured photos/`, with date and EXIF preserved.
- Everything runs locally, in the browser: no photo ever leaves your computer.

## Installation

Requires Python 3.10–3.12.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

With an NVIDIA GPU, face recognition uses CUDA. Otherwise replace `onnxruntime-gpu` with `onnxruntime` in `requirements.txt`. On first launch the models (`buffalo_l` and `yolo11m-seg.pt`) are downloaded automatically.

## Usage

1. **Create your face fingerprint** from a folder of photos that you appear in. 10–20 are enough, the more varied the better:

   ```bash
   python make_ref.py "/path/to/photos/of/me"
   ```

   The script generates `ref_emb.npy`. The file contains biometric data, so it is excluded from git and must not be shared.

2. **Start the program**, pointing it at the folder that contains your photo subfolders:

   ```bash
   python app.py "/path/to/Photos"
   ```

   The browser opens by itself at <http://localhost:5055>. Pick the subfolder to work on at the top.

### Controls

| Key | Action |
|---|---|
| `Enter` | confirm, save and move to the next photo |
| `→` / `←` | skip / go back |
| `C` | click a person: toggles them between covered (red) and visible (green) |
| `B` / `E` | brush to cover / eraser to uncover |
| `[` `]` | brush size |
| `Ctrl+Z` | undo the last brush stroke |
| hold `O` | show the original |

The number next to each person shows how closely their face resembles yours. The most similar face in the photo, if it scores above 0.35, is taken to be you.

## Limitations

- A person cut off by the edge or heavily covered may get an incomplete outline: finish it with the brush.
- If your face isn't visible, the program won't recognise you: one click on yourself makes you visible.
- Blurring still hints at colours and body shape. For real anonymity, use the silhouette.

## License

AGPL-3.0, the same as [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), which the program uses. InsightFace's pre-trained models are released for non-commercial use only.

---

## Other tools in this repo

These are the scripts used around the main program to organise a Google Photos export. Comments and UI text are in Italian. They expect a base folder (for example `From Google/`) with these subfolders:

```
From Google/
├── Tutte le foto/   all photos, flattened
├── Le mie foto/     photos you appear in
├── Da eliminare/    photos to delete later (created by the gallery)
└── Foto sfocate/    blurred copies exported by the gallery
```

### `takeout/`: merge Google Takeout zips into a single folder

```bash
python takeout/flatten.py "/path/From Google"   # extracts every *.zip into "Tutte le foto", renaming name clashes (foo_1.jpg) and skipping identical duplicates
python takeout/verify.py  "/path/From Google"   # checks via CRC that every file in the zips was extracted, before deleting the zips
```

### `find-me/`: find every photo with your face

Uses InsightFace (`buffalo_l`) on the GPU if one is available. Copy your `ref_emb.npy` (from `make_ref.py`) into `find-me/`, then:

1. `python find-me/scan.py "/path/From Google/Tutte le foto"` writes a first pass to `scan.csv` (filename, best similarity, mean similarity, face size).
2. `python find-me/scan2.py "/path/From Google"` rebuilds the references from every photo in `Le mie foto` and scans again. A photo's score is the average of its 5 closest matches, written to `scan2.csv`.
3. `python find-me/contact_sheet.py "/path/From Google"` renders candidates as numbered contact sheets (`t_000.jpg`, …) for review and writes `cand.txt`.
4. `python find-me/scan3.py "/path/From Google"` runs a final pass with references refined again, skipping the photos already reviewed in `cand.txt`, and writes `scan3.csv`.

Then move the matches you've confirmed into `Le mie foto`.

### `gallery/`: LAN gallery that automatically blurs women

`femmine.py` serves a web gallery of `Tutte le foto` that you can open from another device on the local network, e.g. Safari on a Mac. For each person it finds the outline (YOLO11 segmentation) and estimates P(female) with CLIP ViT-L/14 on the masked silhouette, then blurs anyone it judges female (and, optionally, uncertain cases).

- **Mark errors** (`E`): tap a person to flip them between blurred and visible. Corrections are saved in `correzioni_femmine.json`.
- **Ci sono io** (`M`) moves the photo to `Le mie foto`, and **Da eliminare** (`X`) moves it to `Da eliminare`. Both have undo, and Live Photo videos move along with their photo. Nothing is ever deleted.
- **Esporta foto sfocate** writes full-resolution blurred copies, with EXIF preserved, to `Foto sfocate/`. Originals are never modified.

```bash
pip install torch open_clip_torch      # only needed once, for the export below
python gallery/export_clip.py          # creates clip_vitl14.onnx + clip_text.npy (~1.2 GB, not in git)
python gallery/femmine.py "/path/From Google"
```

On first start it prints `http://<ip>:5056/?k=<token>`. The token is stored in `femmine_token.txt` and is required for access. The server listens on all interfaces, so use it only on a trusted network.
