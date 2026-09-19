-- Everything describe knows about a file, written into the file: one
-- X-CLIP vector a shot, woven into the video's own encoded packets by
-- ffrwd.index.weave, the audio beside it, and the AudioSet labels and
-- the transcript as caption tracks. find.sql ranks the vectors; the
-- caption tracks are text, for reading.
--
-- The vectors no longer ride a Matroska subtitle track, so the
-- destination is no longer a .mkv: MP4 takes the same query. The two
-- caption columns are written without an alias, because an alias is a
-- track TITLE and a title is Matroska's alone; untitled they reach any
-- container that carries captions. Matroska writes them as WebVTT and
-- MP4 as mov_text, both in SELECT order, sound first and speech
-- second, and neither says which is which.
--
-- Shots come from ffrwd.shots.simple_detector, which reports a
-- zero-length shot on a fast cut. Those are filtered out before the
-- weave: a record whose span ends where it starts describes no frame,
-- and find.sql would turn it into a zero-length trim.
--
-- SOUND AND SPEECH VECTORS ARE NOT WOVEN YET. They belong in weave's
-- `sound` and `speech` arguments, written as
--
--   ffrwd.describe.embed(ffrwd.describe.sounds(f.audio[1]).labels),
--   ffrwd.describe.embed(ffrwd.whisper.transcribe(ffrwd.vad.speech(f.audio[1])).words),
--
-- with their two spaces added to the table below. ffrwd refuses that
-- today, because embed is a rows module and a packet filter's rows
-- argument has to be a stream module's annotation column:
--
--   UDF_ARG_TYPE: ffrwd.index.weave() takes 'sound' as STRUCT(start_t
--   number, end_t number, vector vector)[], and its argument produces
--   no annotation column
--
-- Until that lifts, the two arguments are NULL and the space table
-- declares only the space that has vectors: a space declared here
-- rides every keyframe whether a record uses it or not, so declaring
-- three would tell every later reader the file carries vectors it does
-- not have.
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
           NULL,
           NULL,
           '[{"name":"clip","dims":512,"encoding":"i8","unit_length":false,
              "modality":"picture",
              "model":"hf:imbcmdth/xclip-onnx@649f3c91b59cd24be316dc505e26eacf5cd00801/video_tower.onnx",
              "model_hash":"87a87b51ae52efa549e6e48f3a3b41f4",
              "query":"hf:imbcmdth/xclip-onnx@649f3c91b59cd24be316dc505e26eacf5cd00801/text_tower.onnx",
              "query_hash":"c14df0fdda1ba530330e278b8fabeb0b",
              "producer":"ffrwd/describe 0.1.2"}]'),
         f.audio[1],
         ffrwd.describe.sounds(f.audio[1]).labels,
         ffrwd.whisper.transcribe(ffrwd.vad.speech(f.audio[1])).words
  FROM input(:'src') f
) TO :'dest'
