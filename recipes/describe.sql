-- Everything describe knows about a file, written into the file: three
-- spaces of embedding vectors woven into the video's own encoded
-- packets by ffrwd.index.weave, the audio beside them, and the
-- AudioSet labels and the transcript as caption tracks. find.sql ranks
-- the vectors; the caption tracks are text, for reading.
--
-- THE THREE SPACES. One per rows argument, and the argument's own name
-- is what puts a row in a space, so the names in the table below are
-- weave's parameter names. clip is X-CLIP's video tower over each
-- shot, searched by its text tower - two different networks, which is
-- why model and query differ there and not in the other two. sound and
-- speech are the same sentence embedder over two different kinds of
-- text, kept apart because a space declaration carries one modality
-- and a reader that wanted only what was said could not tell them
-- apart if they shared an id. The URIs and the digests are this
-- package's own manifest, and a digest is the first sixteen bytes of
-- the SHA-256 the manifest gives.
--
-- THE DESTINATION is any container, not an .mkv: the vectors are in
-- the picture's own packets and no longer in a Matroska track. The two
-- caption columns are written without an alias, because an alias is a
-- track TITLE and a title is Matroska's alone; untitled they reach any
-- container that carries captions.
--
-- sounds() and transcribe() each appear twice, once for a caption
-- track and once under embed(). That is two nodes and two runs of the
-- model: a caption track is an input of the muxer and a rows document
-- is a stage of its own, so the two cannot share one.
--
-- ZERO-LENGTH ROWS. simple_detector reports a zero-length shot on a
-- fast cut, and a record whose span ends where it starts describes no
-- frame. The clip rows are narrowed here, before the weave. The other
-- two cannot be: embed is a rows function, and neither its output nor
-- its input may be narrowed. find.sql filters all three again at read
-- time, which is what catches those.
--
-- describe-video.sql is this recipe without the audio, for a source
-- that has none.
-- variables: src (input media path, with an audio track), dest (output path, .mp4 or .mkv)
-- example: ffrwd compile -f packages/ffrwd/describe/recipes/describe.sql -v src=film.mp4 -v dest=film.described.mp4
COPY (
  SELECT ffrwd.index.weave(
           f.video[1],
           ARRAY(SELECT s
                 FROM unnest(ffrwd.describe.clips(ffrwd.shots.simple_detector(f.video[1])).shots) s
                 WHERE s.end_t > s.start_t),
           ffrwd.describe.embed(ffrwd.describe.sounds(f.audio[1]).labels),
           ffrwd.describe.embed(ffrwd.whisper.transcribe(ffrwd.vad.speech(f.audio[1])).words),
           '[{"name":"clip","dims":512,"encoding":"i8","unit_length":false,
              "modality":"picture",
              "model":"hf:imbcmdth/xclip-onnx@649f3c91b59cd24be316dc505e26eacf5cd00801/video_tower.onnx",
              "model_hash":"87a87b51ae52efa549e6e48f3a3b41f4",
              "query":"hf:imbcmdth/xclip-onnx@649f3c91b59cd24be316dc505e26eacf5cd00801/text_tower.onnx",
              "query_hash":"c14df0fdda1ba530330e278b8fabeb0b",
              "producer":"ffrwd/describe 0.2.0"},
             {"name":"sound","dims":384,"encoding":"i8","unit_length":false,
              "modality":"sound-text",
              "model":"hf:Xenova/all-MiniLM-L6-v2@751bff37182d3f1213fa05d7196b954e230abad9/onnx/model.onnx",
              "model_hash":"759c3cd2b7fe7e93933ad23c4c9181b7",
              "query":"hf:Xenova/all-MiniLM-L6-v2@751bff37182d3f1213fa05d7196b954e230abad9/onnx/model.onnx",
              "query_hash":"759c3cd2b7fe7e93933ad23c4c9181b7",
              "producer":"ffrwd/describe 0.2.0"},
             {"name":"speech","dims":384,"encoding":"i8","unit_length":false,
              "modality":"speech",
              "model":"hf:Xenova/all-MiniLM-L6-v2@751bff37182d3f1213fa05d7196b954e230abad9/onnx/model.onnx",
              "model_hash":"759c3cd2b7fe7e93933ad23c4c9181b7",
              "query":"hf:Xenova/all-MiniLM-L6-v2@751bff37182d3f1213fa05d7196b954e230abad9/onnx/model.onnx",
              "query_hash":"759c3cd2b7fe7e93933ad23c4c9181b7",
              "producer":"ffrwd/describe 0.2.0"}]'),
         f.audio[1],
         ffrwd.describe.sounds(f.audio[1]).labels,
         ffrwd.whisper.transcribe(ffrwd.vad.speech(f.audio[1])).words
  FROM input(:'src') f
) TO :'dest'
