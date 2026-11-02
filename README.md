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
