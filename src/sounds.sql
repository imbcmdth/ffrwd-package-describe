-- AST over the audio, one 10-second window at a time. The audio passes
-- through untouched; each window's AudioSet label lands beside it as a
-- cue - `text` the label, `start_t`/`end_t` the window's own seconds -
-- the same shape speech comes in, so one embed() reads either.
--
-- AST classifies rather than transcribes: one label per window, its own
-- top AudioSet class, not a sentence. windows overlap by half, so an event on a window edge is caught by the next.
CREATE FUNCTION sounds(a audio_stream)
RETURNS STRUCT(a audio_stream, labels cue[])
  AS 'target/wasm32-wasip2/release/sounds.wasm', 'sounds' LANGUAGE wasm;
