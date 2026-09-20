# ffrwd/describe

Three video describers and one embedder over a file, hosted in wasm: `clips`
and `sounds` read the picture and the sound into rows and vectors,
`embed`/`embed_text` put the same 384-d space under a row or a search
prompt, `embed_clip`/`embed_clip_text` do the same in X-CLIP's 512-d space,
and `find.sql` ranks one against the other with `cos_similarity`. Speech is
not this package's job - it composes
[`ffrwd/whisper`](https://github.com/imbcmdth/ffrwd-package-whisper)
and [`ffrwd/vad`](https://github.com/imbcmdth/ffrwd-package-vad)
instead of shipping a third model for it.

The four modules are built and run:
`target/wasm32-wasip2/release/{clips,sounds,embed,embed_clip}.wasm`, all
against the `ffrwd:av@0.16.0` world. That world is unreleased, so is
`ffrwd/index`, and so is the ffrwd that hosts a packet filter; **Local
development** below is the whole route.

## Where the vectors go

A described file carries its vectors in its own encoded video packets, as
SEI messages in H.264 and HEVC and a metadata OBU in AV1, written by
[`ffrwd/index`](https://github.com/imbcmdth/ffrwd-package-index)'s `weave`.
The Matroska vector track this package used to write is gone. What that
changes for a caller:

- The destination is any container, not an `.mkv`. MP4 takes the same query.
- A described file survives `-c copy`, a `-ss` cut and a `UNION ALL`, which
  the track never did.
- `find.sql` reads the vectors back with `ffrwd.index.records`, a packet
  sink the compiler runs while the query compiles. It copies the keyframes
  alone: on the file below, 0.11% of the bytes.
- Nothing about the file is ffrwd's. The format is
  [SPEC.md](https://github.com/imbcmdth/ffrwd-package-index/blob/main/SPEC.md),
  and `ffrwd-index read --mp4 film.described.mp4` prints every record and
  every space declaration without ffrwd involved.

A space declaration says which model made the vectors and which model turns
a search into the same space. Those are two different networks for the clip
space - X-CLIP's video tower wrote the vectors, its text tower embeds the
prompt - and one network reached twice for the text spaces. The URIs and the
digests are this package's own `models` table, written into the recipe as
the `spaces` literal `weave` reads.

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

Four, in two pairs. One pair reads a file that has sound, the other a file
that does not, because the dialect has no way to make a stream optional
inside one query: `f.audio[1]` on a silent source is
`STREAM_NOT_FOUND: 'f.audio[1]' does not exist`, at compile time, before
anything runs. Splitting the recipe is the whole fix.

| recipe | source | writes |
| --- | --- | --- |
| `describe` | has audio | video with vectors woven in, the audio, and two caption tracks |
| `describe-video` | any | video with vectors woven in |
| `find` | described, has audio | the matching spans, video and audio |
| `find-video` | described | the matching spans, video alone |

```
ffrwd run ffrwd/describe:describe-video -v src=film.mp4 -v dest=film.described.mp4
ffrwd run ffrwd/describe:find-video -v src=film.described.mp4 -v prompt='a car driving at night' -v threshold=0.24 -v dest=clips.mp4
```

### What each container gets

`describe` writes the AudioSet labels and the transcript as caption tracks
beside the video and the audio. They are text, not vectors: `find` never
reads them, and they are there to be read by a person or by
`unnest(f.cues)`.

Both columns are written without an alias. An alias on a cue column is the
track's TITLE, and a title is Matroska's: every other container rewrites or
drops it, so ffrwd refuses a titled track anywhere but an `.mkv`. Untitled,
the same column writes wherever captions do. So:

- `.mkv`: the video, the audio, and two WebVTT subtitle tracks, in SELECT
  order - the sound labels first, the transcript second, neither carrying a
  name.
- `.mp4`: the same four streams, the captions as `mov_text`, in the same
  order.

Naming them again would cost MP4, which is the container this rewrite was
for.

### The three spaces

`describe` declares and weaves three, one per rows argument of `weave`, and
the argument's own name is what puts a row in a space.

| space | modality | dims | model | query |
| --- | --- | --- | --- | --- |
| `clip` | `picture` | 512 | X-CLIP video tower | X-CLIP text tower |
| `sound` | `sound-text` | 384 | all-MiniLM-L6-v2 | the same |
| `speech` | `speech` | 384 | all-MiniLM-L6-v2 | the same |

`clip` is the one whose two models differ: the video tower wrote the vectors
and the text tower is what a search goes through. `sound` and `speech` are
the same embedder over two kinds of text, and they are two spaces rather
than one because a declaration carries one modality, and a reader that
wanted only what was said could not tell them apart under one id.

`describe-video` declares one. A space declared to `weave` rides every
keyframe whether a record uses it or not, so a picture-only file described
by the three-space recipe would tell every later reader it carries vectors
it does not have.

### The audio is decoded on the way through, for now

`describe` writes the audio it read, sample for sample: the decoded PCM is
bit-identical to the source's. It is not stream-copied, though. Placing a
packet filter on the video makes the muxer read the audio off the pipe that
already carries it for `sounds` and `vad` instead of opening the source for
it, so the AAC is decoded to `pcm_f32le` and muxed as PCM, which costs about
half the file size again. The same query without the `weave` call copies the
AAC. `describe-video` is unaffected, and so is a `weave` whose query reads
the audio nowhere else.

### Which space is which

A record carries a numeric space id, and `weave` hands ids out in the order
its space table declares them, from zero. `find` does not rely on that. It
joins `ffrwd.index.spaces`, which reads the declarations out of the file's
first packet, and selects on `modality`. An id is a position in one
producer's table; a described file outlives the recipe that wrote it, and
may be remuxed, cut, joined to another file, or written by something that is
not this package. `modality` is a field of the format and means the same
thing in every file. The extra read is one packet.

## A run

Sixty seconds of H.264 at 23.976 fps with a pink-noise AAC track, 45 MB.
Trimmed output of the real commands, on 2026-09-19, against ffrwd 0.17.4 and
the `ffrwd/index` modules built from its checkout.

```
$ ffrwd run ffrwd/describe:describe -v src=toned.mp4 -v dest=described.mp4
```

Fifteen records in three spaces: thirteen shots, one sound label, one
transcript cue. 72 seconds to MP4, 59 to Matroska. The records cost the
video stream 29341 bytes, 0.066% of it, and the same in both containers.
Every picture is the source's own: `framemd5` over the video agrees across
the source, the described MP4, the described Matroska and the same query
with the `weave` call taken out, and every video packet's presentation time,
decode time and duration is the unwoven query's. The decoded audio is
bit-identical too, which is the section above on how it got there.

```
$ ffrwd-index read --mp4 described.mp4
{"space":{"id":0,"dims":512,"encoding":"i8","unit_length":false,"modality":"picture","source":0,
  "model":"hf:imbcmdth/xclip-onnx@649f3c91b59cd24be316dc505e26eacf5cd00801/video_tower.onnx",
  "model_hash":"87a87b51ae52efa549e6e48f3a3b41f4",
  "query":"hf:imbcmdth/xclip-onnx@649f3c91b59cd24be316dc505e26eacf5cd00801/text_tower.onnx",
  "query_hash":"c14df0fdda1ba530330e278b8fabeb0b","producer":"ffrwd/describe 0.1.2"}}
{"space":{"id":1,"dims":384,"encoding":"i8","unit_length":false,"modality":"sound-text","source":0,
  "model":"hf:Xenova/all-MiniLM-L6-v2@751bff37182d3f1213fa05d7196b954e230abad9/onnx/model.onnx",
  "model_hash":"759c3cd2b7fe7e93933ad23c4c9181b7",
  "query":"hf:Xenova/all-MiniLM-L6-v2@751bff37182d3f1213fa05d7196b954e230abad9/onnx/model.onnx",
  "query_hash":"759c3cd2b7fe7e93933ad23c4c9181b7","producer":"ffrwd/describe 0.1.2"}}
{"space":{"id":2,"dims":384,"encoding":"i8","unit_length":false,"modality":"speech","source":0,
  "model":"hf:Xenova/all-MiniLM-L6-v2@751bff37182d3f1213fa05d7196b954e230abad9/onnx/model.onnx",
  "model_hash":"759c3cd2b7fe7e93933ad23c4c9181b7",
  "query":"hf:Xenova/all-MiniLM-L6-v2@751bff37182d3f1213fa05d7196b954e230abad9/onnx/model.onnx",
  "query_hash":"759c3cd2b7fe7e93933ad23c4c9181b7","producer":"ffrwd/describe 0.1.2"}}
described.mp4: scan keyframes: 30 of 1438 samples, 73032 of 67352161 bytes read (0.11% of the file), 36 seeks
```

(each space is one line in the file; wrapped here.)

Then two searches, one aimed at the picture and one at the sound. The same
recipe serves both: the branches are told apart by modality, and the ones a
prompt does not clear drop out.

```
$ ffrwd compile -f recipes/find.sql -v src=described.mp4 \
    -v prompt='a car driving at night' -v threshold=0.23 -v dest=found.mp4
... trim=start=-0.083:end=3.838 ... trim=start=23.315:end=25.276
... trim=start=31.449:end=38.122 ... trim=start=41.167:end=43.419
... trim=start=46.338:end=53.596 ...

$ ffrwd compile -f recipes/find.sql -v src=described.mkv \
    -v prompt='a hiss of static noise' -v threshold=0.30 -v dest=found.mkv
... trim=start=-0.083:end=9.917 ... atrim=start=-0.083:end=9.917 ...
```

Five spans from the picture prompt, 22.022 seconds cut out of sixty, and the
same five off the `.mkv`. One span from the sound prompt, 9.927 seconds, and
it is the sound record: that prompt scores 0.496 against the sound space and
0.014 against the speech space, so only one branch survives the threshold.

The compile of `find` is 31 seconds on the machine this ran on, nearly all
of it the two text towers loading; `find-video`, which the picture prompt
alone needs, compiled the same file in 3.6 seconds with one tower. The read
of the file's own records is the keyframe scan above and is not where the
time goes.

The same three rankings over the binary32 the models produced, before the
8-bit encoding and before the file:

```
$ ffrwd-index search --rows rows.ndjson --space 1 --query query_clip.json --top 5
{"rank":1,"score":0.247439,"start_t":41.250,"end_t":43.502,"space":1,"record_id":8}
{"rank":2,"score":0.243536,"start_t":31.532,"end_t":38.205,"space":1,"record_id":6}
{"rank":3,"score":0.240230,"start_t":46.421,"end_t":53.679,"space":1,"record_id":10}
{"rank":4,"score":0.238564,"start_t":23.398,"end_t":25.359,"space":1,"record_id":4}
{"rank":5,"score":0.236714,"start_t":0.000,"end_t":3.921,"space":1,"record_id":0}

$ ffrwd-index search --rows rows.ndjson --space 3 --query query_noise.json --top 3
{"rank":1,"score":0.496058,"start_t":0.000,"end_t":10.000,"space":3,"record_id":0}

$ ffrwd-index search --rows rows.ndjson --space 2 --query query_noise.json --top 3
{"rank":1,"score":0.013573,"start_t":30.000,"end_t":36.000,"space":2,"record_id":0}
```

Out of the file the same three read 0.247323, 0.496017 and 0.014074, in the
same order, with the clip space's five records ranked the same way. The
scores differ in the third or fourth decimal, which is the 8-bit encoding,
and the spans by 83 milliseconds, which is two frames at 23.976 fps: the
rows are on the source's clock, and a span read back out of the file is an
offset from the picture it rides on, against presentation times an encode
with B-frames does not put in coding order. (`ffrwd-index` numbers its own
rows from one and `weave` numbers the file's from zero, which is why the
space ids differ between the two halves and why `find` looks a space up
rather than naming a number.)

## Building

```
ffrwd install -g ffrwd/wasm
cargo build --target wasm32-wasip2 --release
```

`build.rs` copies the `ffrwd:av` wit out of the installed `ffrwd/wasm`
package. `FFRWD_WIT_DIR` overrides that lookup, which is what a checkout
building against an unreleased world needs.

The X-CLIP ONNX export under `build/xclip` is `scripts/export_xclip.py`'s
output, pinned to the revision the manifest names.

## Checking

```
sh scripts/check.sh
```

`ffrwd.json`'s `test` command, so `ffrwd publish` runs it too. It generates
an eight-second fixture with lavfi, silent and with a noise track, compiles
every recipe, then describes and searches both, checking the pictures
against the source's own and that the file with sound declares all three
spaces. The two compile checks load no model: nothing in `describe.sql` or
`describe-video.sql` runs while the query compiles, which is what makes them
checkable on a machine with an empty model cache. The find recipes embed
their prompt at compile time and read a file that has records in it, so they
need the towers and a described fixture, and the script makes one.

## Local development

None of `ffrwd/index` 0.1.0, the `ffrwd:av@0.16.0` world or the ffrwd that
hosts a packet filter is released, and `ffrwd/wasm` 0.16.0 is not in the
registry either, so `ffrwd link` in the `ffrwd/index` checkout stops at

```
UNSUPPORTED_SQL: the registry has no version 0.16.0 of 'ffrwd/wasm'
(hint: published: 0.9.0, ... 0.15.0)
```

`ffrwd/wasm` carries nothing but the wit a `build.rs` reads, and
`FFRWD_WIT_DIR` supplies that directly, so the way through is a copy of the
index checkout with the dependency dropped:

1. Build the ffrwd CLI and its sidecar from the unreleased checkout, into a
   virtualenv of their own, and point `FFRWD_WASM` at the built
   `ffrwd-wasm`. Give the session an ffrwd home of its own as well (`HOME`,
   and `USERPROFILE` on Windows), because `ffrwd link` writes machine-wide
   and a home of your own keeps the real record alone. Junction or symlink
   `packages`, `models` and `nn-runtime` under it onto the real ones so
   nothing is downloaded twice.
2. Copy the `ffrwd/index` checkout. In the copy, delete `dependencies` from
   `ffrwd.json`.
3. Build its three modules in the copy, against the same sidecar's wit:
   `FFRWD_WIT_DIR=<cli>/sidecar/wit cargo build --release --target wasm32-wasip2 -p weave -p records -p spaces`.
   They have to land in the copy's own `target/`, because that is the path
   `src/index.sql` names and a linked package resolves it in place. The
   build fetches `ffrwd-nal` and `ffrwd-bmff` from GitHub by tag, so it
   needs the network once.
4. `ffrwd link` in the copy: it installs nothing and records
   `ffrwd/index -> <copy>` in that home's links file.
5. `ffrwd link ffrwd/index` here. The name goes in this project's
   `ffrwd.links`, never the lockfile, and shadows the `0.1.0` the manifest
   pins. Every command afterwards says so, which is the point:
   `warning: package 'ffrwd/index' is linked to <copy>, so this command
   depends on files no lockfile pins`.

`ffrwd unlink ffrwd/index` puts the pin back, and `ffrwd unlink -g
ffrwd/index` drops the machine-wide record from anywhere.

## License

Apache-2.0.
