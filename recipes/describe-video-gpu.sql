-- describe-video.sql with the shots and the vectors made on the GPU:
-- ffrwd.shots.simple_detector_gpu and clips_gpu read the picture where
-- the sidecar decodes it, on the device, and only each frame's luma grid
-- and each shot's vector come back to host memory. The file written is
-- describe-video.sql's: the same space, the same shots, each vector made
-- by the same tower from frames resized on the device, which find.sql
-- ranks the same way. Its comment explains the rest.
-- variables: src (input media path), dest (output path, .mp4 or .mkv)
-- example: ffrwd compile -f packages/ffrwd/describe/recipes/describe-video-gpu.sql -v src=film.mp4 -v dest=film.described.mp4
COPY (
  SELECT ffrwd.index.weave(
           f.video[1],
           clip => ARRAY(SELECT s
                         FROM unnest(ffrwd.describe.clips_gpu(f.video[1],
                                                              ffrwd.shots.simple_detector_gpu(f.video[1]))) s
                         WHERE s.end_t > s.start_t),
           spaces => '[{"name":"clip","dims":512,"encoding":"i8","unit_length":false,
                        "modality":"picture",
                        "model":"hf:imbcmdth/xclip-onnx@649f3c91b59cd24be316dc505e26eacf5cd00801/video_tower.onnx",
                        "model_hash":"87a87b51ae52efa549e6e48f3a3b41f4",
                        "query":"hf:imbcmdth/xclip-onnx@649f3c91b59cd24be316dc505e26eacf5cd00801/text_tower.onnx",
                        "query_hash":"c14df0fdda1ba530330e278b8fabeb0b",
                        "producer":"ffrwd/describe 0.3.1"}]')
  FROM input(:'src') f
) TO :'dest'
