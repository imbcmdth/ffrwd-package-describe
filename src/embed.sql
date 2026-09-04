-- Two text embedders, each reached two ways: over rows already in the
-- graph, and over a literal prompt, so each space ranks either the same
-- way. `embed`/`embed_text` put a sentence embedder (all-MiniLM-L6-v2)
-- under a cue's `text` - an AudioSet label, a transcript word - and under
-- a search prompt, so cos_similarity ranks a sentence against a sentence.
-- `embed_clip`/`embed_clip_text` do the same with X-CLIP's text tower, the
-- space `clips`'s video vectors live in: a CLIP text tower matches images,
-- not sentences, so it is kept apart rather than used as the default row
-- embedder.
--
-- The rows form reads whatever produced its cue[] argument, and writes one
-- vector beside each row, no stream involved. The value form takes one
-- prompt and returns one vector, computed at compile time like any other
-- value call.
CREATE FUNCTION embed(rows cue[])
RETURNS STRUCT(start_t number, end_t number, vector vector)[]
  AS 'target/wasm32-wasip2/release/embed.wasm', 'embed' LANGUAGE wasm;

CREATE FUNCTION embed_text(prompt text)
RETURNS vector
  AS 'target/wasm32-wasip2/release/embed.wasm', 'embed_text' LANGUAGE wasm;

CREATE FUNCTION embed_clip(rows cue[])
RETURNS STRUCT(start_t number, end_t number, vector vector)[]
  AS 'target/wasm32-wasip2/release/embed_clip.wasm', 'embed_clip' LANGUAGE wasm;

CREATE FUNCTION embed_clip_text(prompt text)
RETURNS vector
  AS 'target/wasm32-wasip2/release/embed_clip.wasm', 'embed_clip_text' LANGUAGE wasm;
