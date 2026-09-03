"""ONNX export of X-CLIP's two towers, pinned to imbcmdth/xclip-onnx.

Source model: microsoft/xclip-base-patch16-kinetics-600, transformers
revision e4921c41fc296102aae210d43d4127c5e3e51928 (the repo's HEAD as of
this script's writing; pinned explicitly rather than floating "main" so
a re-export years from now still exports the same weights). MIT-licensed,
code and weights.

There is no official or community ONNX export of X-CLIP, so this script
builds two graphs straight out of `transformers.XCLIPModel`, opset 17,
no dynamic axes:

- **video tower**: `get_video_features`, input `pixel_values`
  `[1, 8, 3, 224, 224]` fp32 (one clip, 8 frames, 224x224 RGB), output
  `video_embeds` `[1, 512]`.
- **text tower**: `get_text_features`, inputs `input_ids` and
  `attention_mask`, both `[1, 77]` i64 (the tokenizer's own max length),
  output `text_embeds` `[1, 512]`.

`get_video_features`/`get_text_features` return a
`BaseModelOutputWithPooling` whose `pooler_output` field is overwritten
with the projected [1, 512] embedding - the tower's own pooled output,
not the embedding, would otherwise land there. The wrapper modules below
pull that field explicitly; without it, `torch.onnx.export` flattens the
whole dataclass and the graph's first output is `last_hidden_state`
(`[8, 197, 768]`), not the embedding.

Deterministic: `torch.manual_seed(0)` before building the video tower's
random validation input, and export takes no other randomness. Each
graph is validated against the PyTorch model on one input immediately
after export (max absolute diff, printed and written to manifest.json);
both are float32 rounding noise (~1e-6), not a real divergence.

Run from the package root, inside a venv with torch (CPU wheel),
transformers, onnx and onnxruntime:
    python scripts/export_xclip.py
Writes into build/xclip/ (gitignored) - video_tower.onnx, text_tower.onnx,
manifest.json. Upload the two .onnx files to imbcmdth/xclip-onnx and pin
the resulting revision, filenames and sha256s in ffrwd.json's `models`
block under `clips` (video tower) and `embed` (text tower).
"""

import hashlib
import json
import pathlib

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoTokenizer, XCLIPModel

MODEL_ID = "microsoft/xclip-base-patch16-kinetics-600"
MODEL_REVISION = "e4921c41fc296102aae210d43d4127c5e3e51928"
OUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "build" / "xclip"

NUM_FRAMES = 8
IMAGE_SIZE = 224
OPSET = 17


class VideoTower(torch.nn.Module):
    """Wraps get_video_features so torch.onnx.export sees one tensor in, one out."""

    def __init__(self, model: XCLIPModel):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model.get_video_features(pixel_values=pixel_values).pooler_output


class TextTower(torch.nn.Module):
    """Wraps get_text_features so torch.onnx.export sees two tensors in, one out."""

    def __init__(self, model: XCLIPModel):
        super().__init__()
        self.model = model

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.model.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        ).pooler_output


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"loading {MODEL_ID} @ {MODEL_REVISION} ...")
    model = XCLIPModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)

    # ---- video tower ----
    torch.manual_seed(0)
    pixel_values = torch.rand(1, NUM_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE)

    with torch.no_grad():
        video_ref = model.get_video_features(pixel_values=pixel_values).pooler_output

    video_path = OUT_DIR / "video_tower.onnx"
    torch.onnx.export(
        VideoTower(model),
        (pixel_values,),
        str(video_path),
        input_names=["pixel_values"],
        output_names=["video_embeds"],
        opset_version=OPSET,
        dynamo=False,
    )
    print(f"wrote {video_path} ({video_path.stat().st_size} bytes)")

    sess = ort.InferenceSession(str(video_path), providers=["CPUExecutionProvider"])
    (video_ort,) = sess.run(["video_embeds"], {"pixel_values": pixel_values.numpy()})
    video_diff = float(np.abs(video_ref.numpy() - video_ort).max())
    print(f"video tower max abs diff vs pytorch: {video_diff:.3e}")

    # ---- text tower ----
    text = ["a photo of a person doing something"]
    text_inputs = tokenizer(text, return_tensors="pt", padding="max_length")
    input_ids = text_inputs["input_ids"]
    attention_mask = text_inputs["attention_mask"]

    with torch.no_grad():
        text_ref = model.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        ).pooler_output

    text_path = OUT_DIR / "text_tower.onnx"
    torch.onnx.export(
        TextTower(model),
        (input_ids, attention_mask),
        str(text_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["text_embeds"],
        opset_version=OPSET,
        dynamo=False,
    )
    print(f"wrote {text_path} ({text_path.stat().st_size} bytes)")

    sess = ort.InferenceSession(str(text_path), providers=["CPUExecutionProvider"])
    (text_ort,) = sess.run(
        ["text_embeds"],
        {
            "input_ids": input_ids.numpy().astype(np.int64),
            "attention_mask": attention_mask.numpy().astype(np.int64),
        },
    )
    text_diff = float(np.abs(text_ref.numpy() - text_ort).max())
    print(f"text tower max abs diff vs pytorch: {text_diff:.3e}")

    manifest = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "opset": OPSET,
        "graphs": {
            "video_tower": {
                "file": video_path.name,
                "bytes": video_path.stat().st_size,
                "sha256": sha256(video_path),
                "input": {
                    "name": "pixel_values",
                    "dims": list(pixel_values.shape),
                    "dtype": "f32",
                },
                "output": {"name": "video_embeds", "dims": list(video_ref.shape)},
                "max_abs_diff_vs_pytorch": video_diff,
            },
            "text_tower": {
                "file": text_path.name,
                "bytes": text_path.stat().st_size,
                "sha256": sha256(text_path),
                "inputs": [
                    {"name": "input_ids", "dims": list(input_ids.shape), "dtype": "i64"},
                    {
                        "name": "attention_mask",
                        "dims": list(attention_mask.shape),
                        "dtype": "i64",
                    },
                ],
                "output": {"name": "text_embeds", "dims": list(text_ref.shape)},
                "max_abs_diff_vs_pytorch": text_diff,
            },
        },
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {manifest_path}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
