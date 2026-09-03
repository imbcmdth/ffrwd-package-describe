-- Cut every span whose row scores above threshold against a text
-- prompt, video and audio together - a search over whatever describe.sql
-- already wrote into the file's data track.
-- variables: src (input media path, already describe'd), prompt (search text), threshold (cosine similarity cutoff, e.g. 0.25), dest (output path)
-- example: ffrwd compile -f packages/ffrwd/describe/recipes/find.sql -v src=film.mp4 -v prompt='a dog barking' -v threshold=0.25 -v dest=clips.mp4
COPY (
  SELECT concat(VARIADIC array_agg(ffmpeg.trim(f.video[1],  start => v.start_t, end => v.end_t))),
         concat(VARIADIC array_agg(ffmpeg.atrim(f.audio[1], start => v.start_t, end => v.end_t)))
  FROM input(:'src') f, unnest(f.data) v
  WHERE cos_similarity(v.vector, ffrwd.describe.embed_text(:'prompt')) > :threshold
) TO :'dest'
