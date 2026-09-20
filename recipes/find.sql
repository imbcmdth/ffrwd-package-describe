-- Cut every span whose vector scores above threshold against a text
-- prompt, video and audio together - a search over the vectors
-- describe.sql wove into the file's own encoded packets.
-- ffrwd.index.records reads them while this query compiles, so the
-- trims are numbers in the filtergraph before ffmpeg starts.
--
-- Three spaces, kept apart: sound and speech vectors are a sentence
-- embedder's space (embed_text), clip vectors are X-CLIP's
-- (embed_clip_text) - one branch per space, UNION ALL'd, rather than
-- one WHERE mixing them. An OR across the conditions evaluates every
-- cos_similarity per row whichever space the row is in, and
-- cos_similarity refuses a 384-d row against a 512-d prompt (or the
-- reverse) even where that row's branch would never have kept it.
-- A branch filtered to its own space by AND before its own
-- cos_similarity keeps every call inside one space; a branch whose
-- space the file does not carry matches nothing and drops out.
--
-- Every branch also drops a record whose span ends where it starts,
-- which would otherwise become a zero-length trim. describe.sql
-- narrows the shot rows before it weaves them, so a file this package
-- wrote carries none; this is for every other file, which a search
-- has no reason to trust the writer of.
--
-- WHICH SPACE IS WHICH. A record carries a space id, and weave hands
-- ids out in the order its space table declares them, from zero. This
-- recipe does not rely on that. It joins ffrwd.index.spaces, which
-- reads the declarations out of the file's first packet, and selects
-- on modality: an id is a position in one producer's table, and a
-- described file outlives the recipe that wrote it - remuxed, cut,
-- joined to another file, or written by something that is not this
-- package at all. Modality is a field of the format and means the same
-- thing in every file.
-- variables: src (input media path, already described), prompt (search text), threshold (cosine similarity cutoff, e.g. 0.25), dest (output path)
-- example: ffrwd compile -f packages/ffrwd/describe/recipes/find.sql -v src=film.described.mp4 -v prompt='a dog barking' -v threshold=0.25 -v dest=clips.mp4
COPY (
  SELECT concat(VARIADIC array_agg(ffmpeg.trim(f.video[1],  start => v.start_t, end => v.end_t))),
         concat(VARIADIC array_agg(ffmpeg.atrim(f.audio[1], start => v.start_t, end => v.end_t)))
  FROM input(:'src') f, ffrwd.index.records(f.video[1]) v
       JOIN ffrwd.index.spaces(f.video[1]) s ON v.space = s.space
  WHERE s.modality = 'picture'
    AND v.end_t > v.start_t
    AND cos_similarity(v.vector, ffrwd.describe.embed_clip_text(:'prompt')) > :threshold
  UNION ALL
  SELECT concat(VARIADIC array_agg(ffmpeg.trim(g.video[1],  start => w.start_t, end => w.end_t))),
         concat(VARIADIC array_agg(ffmpeg.atrim(g.audio[1], start => w.start_t, end => w.end_t)))
  FROM input(:'src') g, ffrwd.index.records(g.video[1]) w
       JOIN ffrwd.index.spaces(g.video[1]) t ON w.space = t.space
  WHERE t.modality = 'sound-text'
    AND w.end_t > w.start_t
    AND cos_similarity(w.vector, ffrwd.describe.embed_text(:'prompt')) > :threshold
  UNION ALL
  SELECT concat(VARIADIC array_agg(ffmpeg.trim(h.video[1],  start => x.start_t, end => x.end_t))),
         concat(VARIADIC array_agg(ffmpeg.atrim(h.audio[1], start => x.start_t, end => x.end_t)))
  FROM input(:'src') h, ffrwd.index.records(h.video[1]) x
       JOIN ffrwd.index.spaces(h.video[1]) y ON x.space = y.space
  WHERE y.modality = 'speech'
    AND x.end_t > x.start_t
    AND cos_similarity(x.vector, ffrwd.describe.embed_text(:'prompt')) > :threshold
) TO :'dest'
