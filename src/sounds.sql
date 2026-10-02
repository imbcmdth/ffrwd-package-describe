-- AST over the audio, one 10-second window at a time: each window's
-- AudioSet labels leave as cues, `text` the label and `start_t`/`end_t`
-- the window's own seconds, the same shape speech comes in, so one
-- embed() reads either. No sound leaves; a reader takes it from the
-- source.
--
-- AST classifies rather than transcribes: up to `top` labels a window
-- whose score clears `threshold`, not a sentence. Windows overlap by
-- half, so an event on a window edge is caught by the next, and a window
-- whose labels are the window before's writes none.
CREATE FUNCTION sounds(a audio_stream, threshold number DEFAULT 0.3, top number DEFAULT 3)
RETURNS cue[]
  AS 'target/wasm32-wasip2/release/sounds.wasm', 'sounds' LANGUAGE wasm;
