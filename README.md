# Oscura persone

Programma locale per **sfocare o coprire con una sagoma** tutte le persone in una foto tranne te.

- Riconosce il tuo volto (InsightFace) e trova il contorno esatto di ogni persona (YOLO11 segmentation).
- **Modalità automatica**: prepara la copertura di tutti gli altri e chiede conferma foto per foto.
- **Modalità a mano**: parti da zero e scegli tu chi coprire con un clic, oppure con pennello e gomma.
- Due effetti: **sfocatura** (intensità regolabile) o **sagoma** a tinta unita della stessa forma della persona.
- Gli originali non vengono mai modificati. I risultati vanno in `Foto oscurate/`, con data ed EXIF conservati.
- Funziona tutto in locale, nel browser: nessuna foto lascia il computer.

## Installazione

Serve Python 3.10–3.12.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Con una GPU NVIDIA il riconoscimento dei volti usa CUDA. Altrimenti sostituisci `onnxruntime-gpu` con `onnxruntime` in `requirements.txt`. Al primo avvio i modelli (`buffalo_l` e `yolo11m-seg.pt`) vengono scaricati in automatico.

## Uso

1. **Crea l'impronta del tuo volto** partendo da una cartella di foto in cui ci sei tu. Ne bastano 10–20, meglio se varie:

   ```bash
   python make_ref.py "/percorso/foto/di/me"
   ```

   Lo script genera `ref_emb.npy`. Il file contiene dati biometrici, quindi è escluso da git e non va condiviso.

2. **Avvia il programma** indicando la cartella che contiene le sottocartelle di foto:

   ```bash
   python app.py "/percorso/Foto"
   ```

   Il browser si apre da solo su <http://localhost:5055>. In alto scegli la sottocartella da lavorare.

### Comandi

| Tasto | Azione |
|---|---|
| `Invio` | conferma, salva e passa alla foto successiva |
| `→` / `←` | salta / torna indietro |
| `C` | clic su una persona: la fa passare da coperta (rosso) a visibile (verde) e viceversa |
| `B` / `G` | pennello per coprire / gomma per scoprire |
| `[` `]` | dimensione del pennello |
| `Ctrl+Z` | annulla l'ultima pennellata |
| tieni `O` | mostra l'originale |

Il numero accanto a ogni persona indica quanto il suo volto somiglia al tuo. Il volto più simile della foto, se supera 0,35, viene considerato il tuo.

## Limiti

- Una persona tagliata dal bordo o molto coperta può avere un contorno incompleto: completalo con il pennello.
- Se il tuo volto non si vede, il programma non ti riconosce: basta un clic su di te per renderti visibile.
- La sfocatura lascia intuire colori e corporatura. Per un anonimato vero usa la sagoma.

## Licenza

AGPL-3.0, la stessa di [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), che il programma usa. I modelli pre-addestrati di InsightFace sono rilasciati solo per uso non commerciale.
