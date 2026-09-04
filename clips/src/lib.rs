//! X-CLIP's video tower, one shot at a time. Every frame passes through
//! untouched; each shot leaves behind one row - its own seconds and a 512-d
//! vector, unit length, in the space `embed_text` writes into.
//!
//! Where one shot ends and the next begins is not this module's to decide: a
//! `{"shot": n}` row arriving with a frame - `ffrwd.shots.simple_detector`,
//! composed ahead of this call - closes the shot running when its value
//! changes. A repeated value does not, and a frame with no such row leaves
//! the shot open. With no shot rows at all, the whole input is one shot.
//!
//! A shot is described by eight frames evenly spread over it, resized and
//! normalized the way the checkpoint's own preprocessing declares. The eight
//! are chosen while the shot runs - see `reservoir` - and the tower runs once
//! when the shot ends: at a new shot value, at ten seconds, or when the
//! stream does. A shot longer than ten seconds therefore leaves several rows,
//! each covering its own stretch.
//!
//! The module never opens a file - the host binds the graph to a name with
//! `-nn clips=<path>` and this module asks for that name and nothing else.

// `generate_all`: the world's interfaces come from two other packages -
// ffrwd:av and wasi:nn - and without it bindgen expects them to have been
// generated somewhere else.
wit_bindgen::generate!({
    path: ["wit", "wit-world"],
    // Fully qualified: three packages are in scope, and each has worlds.
    world: "ffrwd:describe-clips/clips",
    generate_all,
});

mod prep;
mod reservoir;

use std::cell::RefCell;

use exports::ffrwd::av::window_filter::{
    Format, FramePayload, Guest, InWindow, Meta, OutFrame, Processed, StreamInfo, WindowMeta,
};
use ffrwd::av::types::Rational;
use serde::{Deserialize, Serialize};
use wasi::nn::graph::{load_by_name, Graph};
use wasi::nn::inference::GraphExecutionContext;
use wasi::nn::tensor::{Tensor, TensorType};

use prep::{PixFmt, SIDE, TENSOR};
use reservoir::{Reservoir, FRAMES};

/// The name the host binds the graph to. `-nn clips=<path>`.
const MODEL: &str = "clips";

/// What the export calls its input and its output.
const INPUT_NAME: &str = "pixel_values";
const OUTPUT_NAME: &str = "video_embeds";

/// The width of the space both towers write into.
const EMBEDDING: usize = 512;

/// How long a stretch one row may cover, in seconds. A shot that runs longer
/// is described in pieces: eight frames spread over ten seconds still say
/// something about them, where eight spread over five minutes do not.
const CAP: f64 = 10.0;

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{},"additionalProperties":false}"#;

// `vector`'s minItems/maxItems fix EMBEDDING as the track's `vector_dims`:
// the compiler reads them at compile time, since it cannot count a
// run-time module's rows itself.
const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"start_t":{"type":"number"},"end_t":{"type":"number"},"vector":{"type":"array","items":{"type":"number"},"minItems":512,"maxItems":512}},"required":["start_t","end_t","vector"],"additionalProperties":false}"#;

/// One shot's row: the seconds it covers, and what the tower made of it.
#[derive(Serialize)]
struct Row {
    start_t: f64,
    end_t: f64,
    vector: Vec<f32>,
}

/// A row naming which shot its frame belongs to. A row shaped any other way
/// is not one of these and is ignored.
#[derive(Deserialize)]
struct ShotRow {
    shot: i64,
}

/// The shot index a frame's rows name, if any of them does. The last one
/// wins.
fn shot_of(rows: &[String]) -> Option<i64> {
    rows.iter()
        .filter_map(|row| serde_json::from_str::<ShotRow>(row).ok())
        .map(|row| row.shot)
        .next_back()
}

/// Whether an incoming shot value ends the shot running: a value different
/// from the one running does, the first value seen sets the baseline
/// instead, and a frame with no shot row changes nothing.
fn is_new_shot(current: Option<i64>, incoming: Option<i64>) -> bool {
    matches!((current, incoming), (Some(a), Some(b)) if a != b)
}

/// The stretch being described: where it started, where it has reached, and
/// the frames chosen to stand for it.
struct Segment {
    start_pts: i64,
    last_pts: i64,
    frames: Reservoir,
}

impl Segment {
    fn new(pts: i64) -> Segment {
        Segment {
            start_pts: pts,
            last_pts: pts,
            frames: Reservoir::default(),
        }
    }
}

/// Whether the frame at `pts` ends the stretch running: a cut does, and so
/// does the cap, which is what keeps one long shot from being described by
/// eight frames spread across minutes of it. The first frame of a stream ends
/// nothing - there is no stretch yet.
fn ends_stretch(cut: bool, segment: Option<&Segment>, pts: i64, tick: f64) -> bool {
    let Some(segment) = segment else {
        return false;
    };
    cut || (pts - segment.start_pts) as f64 * tick >= CAP
}

/// What `init` settled, plus the graph it loaded and where the stream has
/// reached.
struct Opened {
    width: usize,
    height: usize,
    pix_fmt: PixFmt,
    /// Seconds one timestamp tick is worth, from the stream's time base.
    tick: f64,
    /// The shot value running, absent until a shot row has arrived.
    current_shot: Option<i64>,
    /// The newest frame's pixels, kept so a shot's last frame can join the
    /// eight even when the reservoir's spacing passed over it.
    newest: Vec<u8>,
    segment: Option<Segment>,
    /// Held for the life of the instance: building it once is what keeps a
    /// provider's kernels from being chosen again per frame.
    context: GraphExecutionContext,
    /// Kept alive because the context is only valid while its graph is.
    _graph: Graph,
}

thread_local! {
    static OPENED: RefCell<Option<Opened>> = const { RefCell::new(None) };
}

/// This module takes no parameters: a shot's bounds are its own and the
/// tower's geometry is fixed.
fn validate_params(params: &str) -> Result<(), String> {
    match params.trim() {
        "" | "{}" => Ok(()),
        other => Err(format!("clips takes no params, got: {other}")),
    }
}

/// Seconds one timestamp tick is worth, which is what turns a frame's pts
/// into the seconds a row carries.
fn seconds_per_tick(time_base: Rational) -> f64 {
    f64::from(time_base.num) / f64::from(time_base.den)
}

/// The spec's spelling of an error code, so a message says what actually
/// went wrong rather than how this module happens to format things.
fn failed(what: &str, error: &wasi::nn::errors::Error) -> String {
    use wasi::nn::errors::ErrorCode;
    let code = match error.code() {
        ErrorCode::InvalidArgument => "invalid-argument",
        ErrorCode::InvalidEncoding => "invalid-encoding",
        ErrorCode::Timeout => "timeout",
        ErrorCode::RuntimeError => "runtime-error",
        ErrorCode::UnsupportedOperation => "unsupported-operation",
        ErrorCode::TooLarge => "too-large",
        ErrorCode::NotFound => "not-found",
        ErrorCode::Security => "security",
        ErrorCode::Unknown => "unknown",
    };
    format!("clips: {what}: {code} ({})", error.data())
}

/// A tensor's floats, out of the little-endian bytes it arrived as.
fn le_f32s(data: &[u8]) -> Vec<f32> {
    let (whole, _) = data.as_chunks::<4>();
    whole.iter().copied().map(f32::from_le_bytes).collect()
}

/// The floats as the little-endian bytes a tensor travels in.
fn le_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = vec![0u8; values.len() * 4];
    let (words, _) = bytes.as_chunks_mut::<4>();
    for (word, value) in words.iter_mut().zip(values) {
        *word = value.to_le_bytes();
    }
    bytes
}

/// The embedding scaled to unit length, which is what makes a dot product a
/// cosine. A vector of all zeros has no direction and is left alone.
fn normalize(mut vector: Vec<f32>) -> Vec<f32> {
    let length = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if length > 0.0 {
        for value in &mut vector {
            *value /= length;
        }
    }
    vector
}

/// The named tensor out of what a compute answered, checked against the
/// length the export's shape fixes.
fn take(outputs: Vec<(String, Vec<u8>)>) -> Result<Vec<f32>, String> {
    let Some((_, bytes)) = outputs.iter().find(|(name, _)| name == OUTPUT_NAME) else {
        let unclaimed: Vec<&str> = outputs.iter().map(|(n, _)| n.as_str()).collect();
        return Err(format!(
            "clips: the graph answered without {OUTPUT_NAME:?} (unclaimed: {unclaimed:?}); \
             this module wants X-CLIP's video tower export"
        ));
    };
    let values = le_f32s(bytes);
    if values.len() != EMBEDDING {
        return Err(format!(
            "clips: {OUTPUT_NAME} came back as {} floats where the tower's shape fixes \
             {EMBEDDING}",
            values.len()
        ));
    }
    Ok(values)
}

/// One clip through the tower: eight frames in, one unit vector out.
fn run(opened: &mut Opened, input: &[f32]) -> Result<Vec<f32>, String> {
    let dims = [1, FRAMES as u32, 3, SIDE as u32, SIDE as u32];
    let feeds = vec![(
        INPUT_NAME.to_string(),
        Tensor::new(&dims, TensorType::Fp32, &le_bytes(input)),
    )];
    let returned = opened
        .context
        .compute(feeds)
        .map_err(|e| failed("compute", &e))?;
    let outputs: Vec<(String, Vec<u8>)> = returned
        .into_iter()
        .map(|(name, tensor)| (name, tensor.data()))
        .collect();
    Ok(normalize(take(outputs)?))
}

/// The row a finished stretch leaves behind, the tower run once for it.
fn close(opened: &mut Opened, mut segment: Segment) -> Result<String, String> {
    let last = if segment.frames.holds_last() {
        None
    } else {
        Some(prep::preprocess(
            &opened.newest,
            opened.pix_fmt,
            opened.width,
            opened.height,
        ))
    };
    let input = segment.frames.close(last);
    debug_assert_eq!(input.len(), FRAMES * TENSOR);
    let vector = run(opened, &input)?;
    let row = Row {
        start_t: segment.start_pts as f64 * opened.tick,
        end_t: segment.last_pts as f64 * opened.tick,
        vector,
    };
    serde_json::to_string(&row).map_err(|e| format!("clips: the row would not serialize: {e}"))
}

/// One frame: the shot-row check, the row a new shot or the cap ends, and
/// the frame joining whatever stretch it belongs to.
fn step(
    opened: &mut Opened,
    pts: i64,
    rows_in: &[String],
    frame: &[u8],
) -> Result<Vec<String>, String> {
    let shot = shot_of(rows_in);
    let cut = is_new_shot(opened.current_shot, shot);
    if let Some(shot) = shot {
        opened.current_shot = Some(shot);
    }

    let mut rows = Vec::new();
    if ends_stretch(cut, opened.segment.as_ref(), pts, opened.tick) {
        let segment = opened.segment.take().expect("a stretch was running");
        rows.push(close(opened, segment)?);
    }

    let segment = opened.segment.get_or_insert_with(|| Segment::new(pts));
    segment.last_pts = pts;
    let (pix_fmt, width, height) = (opened.pix_fmt, opened.width, opened.height);
    segment
        .frames
        .offer(|| prep::preprocess(frame, pix_fmt, width, height));

    opened.newest.clear();
    opened.newest.extend_from_slice(frame);
    Ok(rows)
}

struct Clips;

impl Guest for Clips {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "clips".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: ROWS_SCHEMA.to_string(),
                pixel_formats: vec!["yuv420p".to_string(), "rgba".to_string()],
                // Not an audio module, so it names no sample formats.
                sample_formats: vec![],
                sample_rates: vec![],
                channel_counts: vec![],
                rows_language: vec![],
            },
            window: 1,
            stride: 1,
            // The reservoir and the running shot value carry over between
            // calls.
            pure: false,
            one_to_one: true,
            // The shot bounds come from rows a module upstream produced.
            reads_rows: true,
            // What leaves a frame is this module's vectors and nothing else.
            forwards_rows: false,
            // One stream in, which is every module here.
            inputs: 1,
        }
    }

    fn init(format: Format, stream_info: StreamInfo, params: String) -> Result<(), String> {
        validate_params(&params)?;
        let Format::Video(video) = format else {
            return Err("clips reads pictures, and this stream is audio".to_string());
        };
        let pix_fmt = PixFmt::parse(&video.pix_fmt)?;
        let tick = seconds_per_tick(stream_info.time_base);

        // The graph is loaded once per instance, and the session built once:
        // the first clip is what a provider picks its kernels on, and every
        // clip after it reuses them.
        let graph =
            load_by_name(MODEL).map_err(|e| failed(&format!("load-by-name({MODEL:?})"), &e))?;
        let context = graph
            .init_execution_context()
            .map_err(|e| failed("init-execution-context", &e))?;

        OPENED.with(|o| {
            *o.borrow_mut() = Some(Opened {
                width: video.width as usize,
                height: video.height as usize,
                pix_fmt,
                tick,
                current_shot: None,
                newest: Vec::new(),
                segment: None,
                context,
                _graph: graph,
            });
        });
        Ok(())
    }

    fn set_params(params: String) -> Result<(), String> {
        validate_params(&params)
    }

    fn process(window: &InWindow, _trailing: Vec<String>, last: bool) -> Processed {
        let mut frames = Vec::with_capacity(window.len() as usize);
        let mut trailing = Vec::new();
        OPENED.with(|opened| {
            let mut borrowed = opened.borrow_mut();
            let opened = borrowed
                .as_mut()
                .expect("init loads the graph before any frame arrives");
            for i in 0..window.len() {
                let pts = window.pts(i);
                let rows_in = window.rows(i);
                let frame = window.fetch(i);
                // `process` has no way to say no, so a graph that failed
                // mid-stream stops the run rather than writing a vector that
                // describes nothing.
                let rows = step(opened, pts, &rows_in, &frame)
                    .unwrap_or_else(|message| panic!("{message}"));
                frames.push(OutFrame {
                    pts,
                    frame: FramePayload::Same,
                    rows,
                });
            }
            // The last call carries no frame - window and stride are 1, so
            // none is ever left over - and the final shot's row has nothing
            // to ride.
            if last {
                if let Some(segment) = opened.segment.take() {
                    trailing.push(close(opened, segment).unwrap_or_else(|m| panic!("{m}")));
                }
            }
        });
        Processed { frames, trailing }
    }
}

export!(Clips);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_time_base_field_becomes_seconds_per_tick() {
        assert!((seconds_per_tick(Rational { num: 1, den: 15360 }) * 15360.0 - 1.0).abs() < 1e-12);
        assert!(
            (seconds_per_tick(Rational {
                num: 1001,
                den: 30000
            }) * 30000.0
                - 1001.0)
                .abs()
                < 1e-9
        );
    }

    #[test]
    fn a_vector_comes_out_of_the_tower_at_unit_length() {
        let vector = normalize(vec![3.0, 4.0]);
        assert!((vector[0] - 0.6).abs() < 1e-6);
        assert!((vector[1] - 0.8).abs() < 1e-6);
        let length: f32 = normalize(vec![1.0; EMBEDDING]).iter().map(|v| v * v).sum();
        assert!((length - 1.0).abs() < 1e-5);
    }

    #[test]
    fn a_vector_with_no_direction_is_left_alone() {
        assert_eq!(normalize(vec![0.0; 4]), vec![0.0; 4]);
    }

    #[test]
    fn floats_survive_the_round_trip_through_a_tensors_bytes() {
        let values = vec![0.0f32, -1.5, 1e-8, 3.25];
        assert_eq!(le_f32s(&le_bytes(&values)), values);
    }

    #[test]
    fn an_answer_without_the_towers_output_is_refused_by_name() {
        let outputs = vec![("last_hidden_state".to_string(), vec![0u8; 16])];
        let error = take(outputs).expect_err("the tower's output is not there");
        assert!(error.contains("video_embeds"), "{error}");
        assert!(error.contains("last_hidden_state"), "{error}");
    }

    #[test]
    fn an_embedding_of_the_wrong_width_is_refused() {
        let outputs = vec![(OUTPUT_NAME.to_string(), vec![0u8; 4 * 256])];
        let error = take(outputs).expect_err("256 is not 512");
        assert!(error.contains("512"), "{error}");
    }

    /// A stretch that began at second zero of a stream counted in
    /// thousandths, and has reached `at` seconds.
    fn running_since_zero(at: f64) -> Segment {
        let mut segment = Segment::new(0);
        segment.last_pts = (at * 1000.0) as i64;
        segment
    }

    #[test]
    fn a_cut_ends_the_stretch_it_arrives_in() {
        let segment = running_since_zero(2.0);
        assert!(ends_stretch(true, Some(&segment), 2_000, 0.001));
        assert!(!ends_stretch(false, Some(&segment), 2_000, 0.001));
    }

    #[test]
    fn the_first_frame_of_a_stream_ends_nothing() {
        assert!(!ends_stretch(true, None, 0, 0.001));
        assert!(!ends_stretch(false, None, 0, 0.001));
    }

    #[test]
    fn ten_seconds_ends_a_stretch_no_cut_has() {
        let segment = running_since_zero(9.9);
        assert!(
            !ends_stretch(false, Some(&segment), 9_999, 0.001),
            "just under the cap the stretch runs on"
        );
        assert!(
            ends_stretch(false, Some(&segment), 10_000, 0.001),
            "and at it the stretch ends"
        );
    }

    #[test]
    fn the_cap_is_counted_from_the_stretchs_own_start() {
        // A stretch that began at the twentieth second is capped at the
        // thirtieth, not the tenth.
        let mut segment = Segment::new(20_000);
        segment.last_pts = 29_000;
        assert!(!ends_stretch(false, Some(&segment), 29_999, 0.001));
        assert!(ends_stretch(false, Some(&segment), 30_000, 0.001));
    }

    #[test]
    fn params_are_refused_because_there_are_none() {
        assert!(validate_params("").is_ok());
        assert!(validate_params("{}").is_ok());
        let error = validate_params(r#"{"threshold":9}"#).expect_err("no params exist");
        assert!(error.contains("clips takes no params"), "{error}");
    }

    #[test]
    fn a_shot_row_is_read_and_a_row_shaped_any_other_way_is_not() {
        assert_eq!(shot_of(&[r#"{"shot":3}"#.to_string()]), Some(3));
        assert_eq!(shot_of(&[r#"{"start_t":0,"end_t":1}"#.to_string()]), None);
        assert_eq!(shot_of(&[]), None);
        assert_eq!(
            shot_of(&[r#"{"shot":1}"#.to_string(), r#"{"shot":2}"#.to_string()]),
            Some(2),
            "the last shot row a frame carries is the one that counts"
        );
    }

    #[test]
    fn a_new_shot_value_ends_the_shot_running_and_a_repeat_does_not() {
        assert!(is_new_shot(Some(0), Some(1)));
        assert!(!is_new_shot(Some(1), Some(1)));
    }

    #[test]
    fn the_first_shot_row_sets_the_baseline_without_ending_anything() {
        assert!(!is_new_shot(None, Some(0)));
    }

    #[test]
    fn a_frame_with_no_shot_row_ends_nothing() {
        assert!(!is_new_shot(Some(0), None));
        assert!(!is_new_shot(None, None));
    }
}
