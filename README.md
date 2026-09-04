<!-- draft: maintainer to rewrite -->
# ffrwd/describe

Three describers and one embedder over a file, hosted in wasm: `clips`
and `sounds` read the picture and the sound into rows and vectors,
`embed`/`embed_text` put the same 512-d space under a row or a search
prompt, and `find.sql` ranks one against the other with
`cos_similarity`. Speech is not this package's job - it composes
[`ffrwd/whisper`](https://github.com/imbcmdth/ffrwd-package-whisper)
and [`ffrwd/vad`](https://github.com/imbcmdth/ffrwd-package-vad)
instead of shipping a third model for it.

No module is built yet. This is the manifest, the SQL declarations,
and the model export - `cargo` layout and the wasm modules themselves
come next.

## Models

| export | model | source | license | pinned at |
| --- | --- | --- | --- | --- |
| `clips` | X-CLIP video tower | [microsoft/xclip-base-patch16-kinetics-600](https://huggingface.co/microsoft/xclip-base-patch16-kinetics-600), exported by us | MIT | [imbcmdth/xclip-onnx](https://huggingface.co/imbcmdth/xclip-onnx) |
| `embed`, `embed_text` | all-MiniLM-L6-v2, fp32 | [Xenova/all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2) | Apache-2.0 | pinned in `ffrwd.json` |
| `embed_clip`, `embed_clip_text` | X-CLIP text tower | same source model as `clips`, exported by us | MIT | [imbcmdth/xclip-onnx](https://huggingface.co/imbcmdth/xclip-onnx) |
| `sounds` | AST fp32 | [onnx-community/ast-finetuned-audioset-10-10-0.4593-ONNX](https://huggingface.co/onnx-community/ast-finetuned-audioset-10-10-0.4593-ONNX) | BSD-3-Clause | pinned in `ffrwd.json` |
| speech | whisper-medium, int8 | not this package - `ffrwd/whisper` depends on [imbcmdth/whisper-medium-onnx](https://huggingface.co/imbcmdth/whisper-medium-onnx) | Apache-2.0 | `ffrwd/whisper`'s own manifest |

The package is Apache-2.0; each model keeps its own license above. The
weights are not in the archive - `ffrwd.json`'s `models` block pins
exact repo, revision, file and sha256, and `ffrwd install` fetches and
verifies them. `scripts/export_xclip.py` is the X-CLIP export, checked
in and re-runnable; AST and whisper are pinned as published, nothing
to export.

whisper (int8) has no CPU or CUDA path in this fleet - see
`ffrwd/whisper`'s own notes. `ModelPin` has no field to say so in the
manifest (`repo`, `revision`, `file`, `sha256` only); this is the only
place it's written down.

## Exports

- `clips(v)` - X-CLIP's video tower, one shot at a time: the video
  passes through untouched, one 512-d vector lands beside each shot.
- `sounds(a)` - AST over 10-second windows: the audio passes through
  untouched, one AudioSet label lands beside each window, as a cue
  (`text` the label, `start_t`/`end_t` the window).
- `embed(rows)` - X-CLIP's text tower over rows already in the graph
  (sounds's labels, or a transcript's words - both `cue[]`): one
  vector per row, no stream involved.
- `embed_text(prompt)` - the same text tower over a literal prompt,
  computed once at compile time like any other value call.

## Recipes

- `describe` - everything above, one row per file: streams, cue rows,
  and every vector, written to a table destination (`.ndjson`, or any
  `COPY ... WITH (format ...)` shape).
- `find` - cut every span whose row scores above a threshold against a
  search prompt, video and audio together.

```
ffrwd ffrwd.describe.describe -v src=film.mp4 -v dest=film.ndjson
ffrwd ffrwd.describe.find -v src=film.mp4 -v prompt='a dog barking' -v threshold=0.25 -v dest=clips.mp4
```

## Building

Not yet - there is no `Cargo.toml` in this checkout. Once the three
modules exist, the pattern is the one every other wasm package here
uses:

```
ffrwd install -g ffrwd/wasm
cargo build --target wasm32-wasip2 --release
```
