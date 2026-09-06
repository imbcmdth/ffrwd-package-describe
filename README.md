# ffrwd/describe

Three video describers and one embedder over a file, hosted in wasm: `clips`
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

### X-CLIP

Function: `ffrwd.describe.clips()`, `ffrwd.describe.embed_clip()`, `ffrwd.describe.embed_clip_text()`

Model: [microsoft/xclip-base-patch16-kinetics-600](https://huggingface.co/microsoft/xclip-base-patch16-kinetics-600), exported by us to [imbcmdth/xclip-onnx](https://huggingface.co/imbcmdth/xclip-onnx).

License: MIT

### all-MiniLM-L6-v2

Function: `ffrwd.describe.embed()`, `ffrwd.describe.embed_text()`

Model: [Xenova/all-MiniLM-L6-v2](https://huggingface.co/Xenova/all-MiniLM-L6-v2) 

License: Apache-2.0

### AST

Function: `ffrwd.describe.sounds()`

Model: [onnx-community/ast-finetuned-audioset-10-10-0.4593-ONNX](https://huggingface.co/onnx-community/ast-finetuned-audioset-10-10-0.4593-ONNX)

License: BSD-3-Clause

---

## Recipes

- `describe` - everything above, one row per file: streams, cue rows,
  and every vector, written to a table destination (`.ndjson`, or any
  `COPY ... WITH (format ...)` shape).
- `find` - cut every span whose row scores above a threshold against a
  search prompt, video and audio together.

```
ffrwd run ffrwd/describe:describe -v src=film.mp4 -v dest=film.ndjson
ffrwd run ffrwd/describe:find -v src=film.mp4 -v prompt='a dog barking' -v threshold=0.25 -v dest=clips.mp4
```

## Building

Not yet - there is no `Cargo.toml` in this checkout. Once the three
modules exist, the pattern is the one every other wasm package here
uses:

```
ffrwd install -g ffrwd/wasm
cargo build --target wasm32-wasip2 --release
```
