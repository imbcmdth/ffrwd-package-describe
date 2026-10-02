-- X-CLIP's video tower, one shot at a time: each shot leaves as one row,
-- its seconds and a 512-d vector, ready for cos_similarity against
-- embed_clip_text's own. No picture leaves; a reader takes it from the
-- source.
--
-- The tower takes a fixed 8-frame, 224x224 clip; the module samples and
-- resizes internally, so callers never see the model geometry. A shot's
-- bounds are not this module's to find: it reads them off `cuts`, a
-- `{"shot": n}` row per frame from a cut detector over the same picture,
-- ffrwd.shots.simple_detector for one. Left out, the whole input is one
-- shot. A shot longer than ten seconds leaves a row per ten seconds, and
-- a row leaves when its stretch ends.
--
--   ffrwd.describe.clips(f.video[1], ffrwd.shots.simple_detector(f.video[1]))
CREATE FUNCTION clips(v video_stream, cuts STRUCT(shot number)[] DEFAULT NULL)
RETURNS STRUCT(start_t number, end_t number, vector vector)[]
  AS 'target/wasm32-wasip2/release/clips.wasm', 'clips' LANGUAGE wasm;
