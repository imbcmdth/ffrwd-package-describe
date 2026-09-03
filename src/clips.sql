-- X-CLIP's video tower, one shot at a time. The video passes through
-- untouched; each shot lands beside it as one 512-d vector, ready for
-- cos_similarity against embed_text's own.
--
-- The tower takes a fixed 8-frame, 224x224 clip; the module samples
-- and resizes internally, so callers never see the model geometry. A
-- shot's bounds are the module's own cut detection, not an upstream
-- argument - there is nothing to configure.
CREATE FUNCTION clips(v video_stream)
RETURNS STRUCT(v video_stream, shots STRUCT(start_t number, end_t number, vector vector)[])
  AS 'target/wasm32-wasip2/release/clips.wasm', 'clips' LANGUAGE wasm;
