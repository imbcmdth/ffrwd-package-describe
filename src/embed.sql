-- X-CLIP's text tower, twice: once over rows already in the graph,
-- once over a literal prompt, so the same 512-d space ranks either
-- against clips's video vectors.
--
-- `embed` is a rows function: it reads whatever produced its cue[]
-- argument - sounds's labels, or a transcript's words, either shape -
-- and writes one vector beside each row, no stream involved. `embed_text`
-- is a value function: one prompt in, one vector out, computed at
-- compile time like any other value call. Both load the same text_tower
-- graph the manifest pins under `embed`.
CREATE FUNCTION embed(rows cue[])
RETURNS STRUCT(start_t number, end_t number, vector vector)[]
  AS 'target/wasm32-wasip2/release/embed.wasm', 'embed' LANGUAGE wasm;

CREATE FUNCTION embed_text(prompt text)
RETURNS vector
  AS 'target/wasm32-wasip2/release/embed.wasm', 'embed_text' LANGUAGE wasm;
