-- Everything describe knows about a file, one row: the streams
-- themselves, sounds and speech as cue rows, and every embedding
-- find.sql later ranks against - clip vectors straight from the video
-- tower, sound/speech vectors from a sentence embedder over their own
-- rows.
-- variables: src (input media path), dest (output path, an .mkv: the rows become its metadata tracks)
-- example: ffrwd compile -f packages/ffrwd/describe/recipes/describe.sql -v src=film.mp4 -v dest=film.described.mkv
COPY (
  WITH d AS (
    SELECT f.video[1] AS v, f.audio[1] AS a,
           ffrwd.describe.clips(ffrwd.shots.simple_detector(f.video[1])).shots AS clip,
           ffrwd.describe.sounds(f.audio[1]).labels AS sound,
           ffrwd.whisper.transcribe(ffrwd.vad.speech(f.audio[1])).words AS speech
    FROM input(:'src') f
  )
  SELECT d.v, d.a, d.sound, d.speech,
         d.clip                         AS clip_vectors,
         ffrwd.describe.embed(d.sound)  AS sound_vectors,
         ffrwd.describe.embed(d.speech) AS speech_vectors
  FROM d
) TO :'dest'
