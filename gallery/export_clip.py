"""Esporta l'encoder immagini di CLIP ViT-L-14 in ONNX (per la GPU) e i vettori di testo femmina/maschio."""
import os, torch, open_clip, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="datacomp_xl_s13b_b90k"); model.eval()
tok = open_clip.get_tokenizer("ViT-L-14")
F = ["a photo of a woman", "a photo of a girl", "a photo of a teenage girl"]
M = ["a photo of a man", "a photo of a boy", "a photo of a teenage boy"]
with torch.no_grad():
    t = model.encode_text(tok(F + M)); t = t / t.norm(dim=-1, keepdim=True)
tf, tm = t[:3].mean(0), t[3:].mean(0)
np.save(os.path.join(HERE, "clip_text.npy"), torch.stack([tf / tf.norm(), tm / tm.norm()]).numpy())
print("testo ok", flush=True)


class V(torch.nn.Module):
    def __init__(s, v): super().__init__(); s.v = v
    def forward(s, x): e = s.v(x); return e / e.norm(dim=-1, keepdim=True)


# senza no_grad: altrimenti torch usa l'attention "veloce" che ONNX non supporta
if True:
    torch.onnx.export(V(model.visual), torch.randn(1, 3, 224, 224), os.path.join(HERE, "clip_vitl14.onnx"),
                      input_names=["x"], output_names=["e"], dynamic_axes={"x": {0: "n"}, "e": {0: "n"}},
                      opset_version=17, dynamo=False)
print("esportato", flush=True)
