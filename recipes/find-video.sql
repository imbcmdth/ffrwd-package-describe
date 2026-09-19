-- find.sql without the audio, for a file that has none - the one
-- describe-video.sql wrote, or any picture-only file with vectors in
-- it. The search is the same; only the atrim column is gone, and with
-- it the STREAM_NOT_FOUND a missing audio track would raise.
-- variables: src (input media path, already described), prompt (search text), threshold (cosine similarity cutoff, e.g. 0.25), dest (output path)
-- example: ffrwd compile -f packages/ffrwd/describe/recipes/find-video.sql -v src=film.described.mp4 -v prompt='a dog barking' -v threshold=0.25 -v dest=clips.mp4
COPY (
  SELECT concat(VARIADIC array_agg(ffmpeg.trim(f.video[1], start => v.start_t, end => v.end_t)))
  FROM input(:'src') f, ffrwd.index.records(f.video[1]) v
       JOIN ffrwd.index.spaces(f.video[1]) s ON v.space = s.space
  WHERE s.modality = 'picture'
    AND cos_similarity(v.vector, ffrwd.describe.embed_clip_text(:'prompt')) > :threshold
  UNION ALL
  SELECT concat(VARIADIC array_agg(ffmpeg.trim(g.video[1], start => w.start_t, end => w.end_t)))
  FROM input(:'src') g, ffrwd.index.records(g.video[1]) w
       JOIN ffrwd.index.spaces(g.video[1]) t ON w.space = t.space
  WHERE t.modality = 'sound-text'
    AND cos_similarity(w.vector, ffrwd.describe.embed_text(:'prompt')) > :threshold
  UNION ALL
  SELECT concat(VARIADIC array_agg(ffmpeg.trim(h.video[1], start => x.start_t, end => x.end_t)))
  FROM input(:'src') h, ffrwd.index.records(h.video[1]) x
       JOIN ffrwd.index.spaces(h.video[1]) y ON x.space = y.space
  WHERE y.modality = 'speech'
    AND cos_similarity(x.vector, ffrwd.describe.embed_text(:'prompt')) > :threshold
) TO :'dest'
