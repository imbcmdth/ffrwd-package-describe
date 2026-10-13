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

-- `clips` with each kept frame made into the tower's input by the host, and
-- the eight joined there, so the module never reads the picture. With
-- `memory` 'gpu' the picture stays on the GPU: each frame the module keeps
-- is resized and normalised on the device, and where the tower's session
-- reads the device's memory the eight are joined there and bound to it in
-- place. With 'host' the same is made on the CPU from frames in host memory.
-- The frame's centre square is resized in one pass, where `clips` resizes
-- the short side and then crops, and nothing is rounded to eight bits, so a
-- shot's vector is near `clips`' rather than equal to it: a cosine of 0.994
-- to 0.9999 on a five-shot test clip, and the same ranking for a prompt. The
-- cuts, the stretches and the spans are `clips`', and the model is the same
-- one. ffrwd.shots.simple_detector_gpu reads the same picture on the device.
--
--   ffrwd.describe.clips_gpu(f.video[1], ffrwd.shots.simple_detector_gpu(f.video[1]))
CREATE FUNCTION clips_gpu(v video_stream, cuts STRUCT(shot number)[] DEFAULT NULL,
                          memory text DEFAULT 'gpu')
RETURNS STRUCT(start_t number, end_t number, vector vector)[]
  AS 'target/wasm32-wasip2/release/clips_gpu.wasm', 'clips' LANGUAGE wasm;
