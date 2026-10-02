-- The picture half of describe, for a source with no audio track: one
-- X-CLIP vector a shot, woven into the video's own encoded packets by
-- ffrwd.index.weave. No audio is read, so nothing here needs one.
--
-- describe.sql is the same recipe over a file that has sound. The two
-- are separate because a stream a source lacks cannot be left out of
-- one query: describe.sql reads f.audio[1], and a source with no audio
-- track stops it before anything runs, with
-- STREAM_NOT_FOUND: 'f.audio[1]' does not exist. weave's sound and
-- speech inputs are simply left unbound here.
--
-- Shots come from ffrwd.shots.simple_detector, which reports a
-- zero-length shot on a fast cut. Those are narrowed out here rather
-- than woven: a record whose span ends where it starts describes no
-- frame, and a search over it would cut nothing.
--
-- One space, not describe.sql's three. A space declared to weave
-- rides every keyframe whether a record uses it or not, so declaring
-- sound and speech here would tell every later reader the file
-- carries vectors it does not have.
-- variables: src (input media path), dest (output path, .mp4 or .mkv)
-- example: ffrwd compile -f packages/ffrwd/describe/recipes/describe-video.sql -v src=film.mp4 -v dest=film.described.mp4
COPY (
  SELECT ffrwd.index.weave(
           f.video[1],
           clip => ARRAY(SELECT s
                         FROM unnest(ffrwd.describe.clips(f.video[1],
                                                          ffrwd.shots.simple_detector(f.video[1]))) s
                         WHERE s.end_t > s.start_t),
           spaces => '[{"name":"clip","dims":512,"encoding":"i8","unit_length":false,
                        "modality":"picture",
                        "model":"hf:imbcmdth/xclip-onnx@649f3c91b59cd24be316dc505e26eacf5cd00801/video_tower.onnx",
                        "model_hash":"87a87b51ae52efa549e6e48f3a3b41f4",
                        "query":"hf:imbcmdth/xclip-onnx@649f3c91b59cd24be316dc505e26eacf5cd00801/text_tower.onnx",
                        "query_hash":"c14df0fdda1ba530330e278b8fabeb0b",
                        "producer":"ffrwd/describe 0.3.0"}]')
  FROM input(:'src') f
) TO :'dest'
