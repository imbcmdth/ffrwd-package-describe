-- Cut every span whose row scores above threshold against a text prompt,
-- video and audio together - a search over every vector track describe.sql
-- wrote into the file. Two spaces, kept apart: sound/speech vectors are a
-- sentence embedder's space (embed_text), clip vectors are X-CLIP's
-- (embed_clip_text) - one branch per track, UNION ALL'd, rather than one
-- WHERE mixing both: an OR across the two conditions evaluates both
-- cos_similarity calls per row regardless of which one the row's own track
-- matches, and cos_similarity refuses a 384-d row against a 512-d prompt
-- (or the reverse) even when that row's branch would never have kept it.
-- Splitting into two UNION ALL branches, each filtered to its own track by
-- AND before its own cos_similarity, keeps every call inside its own space.
-- variables: src (input media path, already described), prompt (search text), threshold (cosine similarity cutoff, e.g. 0.25), dest (output path)
-- example: ffrwd compile -f packages/ffrwd/describe/recipes/find.sql -v src=film.described.mkv -v prompt='a dog barking' -v threshold=0.25 -v dest=clips.mp4
COPY (
  SELECT concat(VARIADIC array_agg(ffmpeg.trim(f.video[1],  start => v.start_t, end => v.end_t))),
         concat(VARIADIC array_agg(ffmpeg.atrim(f.audio[1], start => v.start_t, end => v.end_t)))
  FROM input(:'src') f, unnest(f.embeddings) v
  WHERE v.track <> 'clip_vectors'
    AND cos_similarity(v.vector, ffrwd.describe.embed_text(:'prompt')) > :threshold
  UNION ALL
  SELECT concat(VARIADIC array_agg(ffmpeg.trim(g.video[1],  start => w.start_t, end => w.end_t))),
         concat(VARIADIC array_agg(ffmpeg.atrim(g.audio[1], start => w.start_t, end => w.end_t)))
  FROM input(:'src') g, unnest(g.embeddings) w
  WHERE w.track = 'clip_vectors'
    AND cos_similarity(w.vector, ffrwd.describe.embed_clip_text(:'prompt')) > :threshold
) TO :'dest'
