//! What AudioSet class the audio sounds like: the audio passes through
//! untouched and each 10 s window's labels above a threshold leave as cues
//! beside it.
//!
//! The graph is the Audio Spectrogram Transformer, run through `wasi:nn`.
//! The module never opens a file - the host binds the graph to a name with
//! `-nn sounds=<path>` and this module asks for that name and nothing else.
//! The 527 AudioSet class names travel as source (`audioset-labels.txt`,
//! `src/labels.rs`) rather than as a second model file: the world this
//! module exports imports no filesystem, so a file the package installed
//! beside the graph is as unreachable as everything else on disk.
//!
//! # The window
//!
//! A window is 10 s (`WINDOW`), and consecutive windows overlap by half: the
//! stride is 5 s (`STRIDE`). AST classifies one clip at a time with no
//! timing finer than the clip, so a sound that starts or ends near a window
//! edge would be half-covered by a disjoint tiling and easy to miss; the
//! overlap means every two-and-a-half-second stretch of audio is judged
//! from the middle of some window rather than only ever from an edge.
//!
//! The fbank feature the AST checkpoint reads is the same either way -
//! `src/fbank.rs` doesn't know about the overlap, only about however many
//! samples one call hands it.
//!
//! # Passing audio through under overlap
//!
//! `window-filter`'s own contract says `frame-payload::same` may only be
//! used when a module's stride is its window, "since overlapping windows
//! passed through would emit every sample more than once" - and this
//! module's stride is half its window, on purpose. So instead of `same`,
//! each call emits `new` bytes holding only the samples this instance has
//! not already sent downstream: the whole window on the first call, and
//! after that the trailing `STRIDE` samples that were not part of the
//! previous one. `emitted_through`, kept in `Opened`, is the second up to
//! which that has already happened; it is what makes a call's outcome
//! depend on the calls before it, which is also why `describe` declares
//! `pure: false`.
//!
//! # The rows
//!
//! One row per label kept - `text`, `start_t`, `end_t` - the same cue shape
//! `ffrwd/vad`'s spans and `ffrwd/whisper`'s captions use, so one `embed()`
//! reads any of the three. `text` is the AudioSet class name; the model
//! gives no timing finer than the window itself, so `start_t`/`end_t` are
//! the window's own span, in seconds from the start of the stream. `top`
//! labels are kept out of however many clear `threshold`, ranked by score.
//!
//! A window whose label set is exactly the window before's emits no rows:
//! consecutive windows overlap by half, so a sound that hasn't changed says
//! the same thing twice for free, and a caller reading these rows back
//! would otherwise see one label repeated across every window it spans.
//! The comparison ignores rank order - the same labels in a different order
//! still count as the same set - and does not affect the audio passed
//! through, only which rows ride beside it.

// `generate_all`: the world's interfaces come from two other packages -
// ffrwd:av and wasi:nn - and without it bindgen expects them to have been
// generated somewhere else.
wit_bindgen::generate!({
    path: ["wit", "wit-world"],
    world: "ffrwd:describe-sounds/sounds",
    generate_all,
});

mod fbank;
mod labels;

use std::cell::RefCell;

use exports::ffrwd::av::window_filter::{
    Format, FramePayload, Guest, InWindow, Meta, OutFrame, Processed, StreamInfo, WindowMeta,
};
use serde::{Deserialize, Serialize};
use wasi::nn::graph::{load_by_name, Graph};
use wasi::nn::inference::GraphExecutionContext;
use wasi::nn::tensor::{Tensor, TensorType};

/// The name the host binds the graph to. `-nn sounds=<path>`.
const MODEL: &str = "sounds";

/// The graph's own names for the tensors it takes and returns.
const INPUT_NAME: &str = "input_values";
const OUTPUT_NAME: &str = "logits";

/// 10 s at 16 kHz: the clip AST classifies in one call.
const WINDOW: u32 = fbank::SAMPLE_RATE * 10;
/// 5 s at 16 kHz: half the window, so consecutive clips overlap.
const STRIDE: u32 = fbank::SAMPLE_RATE * 5;

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"threshold":{"type":"number","minimum":0,"maximum":1,"default":0.3},"top":{"type":"number","minimum":0,"default":3}},"additionalProperties":false}"#;
const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"text":{"type":"string"},"start_t":{"type":"number"},"end_t":{"type":"number"}},"required":["text","start_t","end_t"],"additionalProperties":false}"#;

fn default_threshold() -> f64 {
    0.3
}

fn default_top() -> f64 {
    3.0
}

/// `top` is a count but the schema spells it a JSON number like every other
/// parameter here, so it is parsed as one and checked whole below.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawParams {
    #[serde(default = "default_threshold")]
    threshold: f64,
    #[serde(default = "default_top")]
    top: f64,
}

struct Params {
    threshold: f64,
    top: usize,
}

/// Parses and validates params, shared by `init` and `set_params`.
fn parse_params(params: &str) -> Result<Params, String> {
    let trimmed = params.trim();
    let raw: RawParams = if trimmed.is_empty() {
        RawParams {
            threshold: default_threshold(),
            top: default_top(),
        }
    } else {
        serde_json::from_str(trimmed).map_err(|e| format!("sounds: bad params: {e}"))?
    };
    if !(0.0..=1.0).contains(&raw.threshold) {
        return Err(format!(
            "sounds: threshold is a probability, and {} is not between 0 and 1",
            raw.threshold
        ));
    }
    if raw.top < 0.0 || raw.top.fract() != 0.0 {
        return Err(format!(
            "sounds: top is a count of labels, and {} is not a whole number at least 0",
            raw.top
        ));
    }
    Ok(Params {
        threshold: raw.threshold,
        top: raw.top as usize,
    })
}

/// One label as the NDJSON line it leaves as.
#[derive(Serialize)]
struct Row<'a> {
    text: &'a str,
    start_t: f64,
    end_t: f64,
}

fn row(text: &str, start_t: f64, end_t: f64) -> String {
    serde_json::to_string(&Row {
        text,
        start_t,
        end_t,
    })
    .expect("row serializes")
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
    format!("sounds: {what}: {code} ({})", error.data())
}

/// Floats as the little-endian bytes a tensor carries.
fn to_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// A tensor's bytes back as floats.
fn to_floats(bytes: &[u8]) -> Vec<f32> {
    let (words, _) = bytes.as_chunks::<4>();
    words.iter().copied().map(f32::from_le_bytes).collect()
}

/// A timestamp in the stream's own unit, as seconds. `den` is always
/// positive, so a negative timestamp stays negative.
fn seconds(ticks: i64, num: i32, den: i32) -> f64 {
    ticks as f64 * f64::from(num) / f64::from(den)
}

/// `sample_count` samples, as ticks in the stream's own time base.
fn ticks_of_samples(sample_count: usize, num: i32, den: i32) -> i64 {
    (sample_count as f64 / f64::from(fbank::SAMPLE_RATE) * f64::from(den) / f64::from(num)).round()
        as i64
}

/// One window through the graph: its AudioSet labels at or above
/// `threshold`, the highest-scoring `top` of them, ranked score first.
fn classify(
    context: &GraphExecutionContext,
    samples: &[f32],
    threshold: f64,
    top: usize,
) -> Result<Vec<&'static str>, String> {
    if fbank::num_frames(samples.len()) == 0 {
        // Not even one 25 ms frame: nothing here but padding, and the model
        // has nothing worth guessing at.
        return Ok(Vec::new());
    }
    let features = fbank::extract(samples);
    let outputs = context
        .compute(vec![(
            INPUT_NAME.to_string(),
            Tensor::new(
                &[1, fbank::MAX_FRAMES as u32, fbank::NUM_MEL_BINS as u32],
                TensorType::Fp32,
                &to_bytes(&features),
            ),
        )])
        .map_err(|e| failed("compute", &e))?;

    let logits = outputs
        .iter()
        .find(|(name, _)| name == OUTPUT_NAME)
        .map(|(_, tensor)| to_floats(&tensor.data()))
        .ok_or_else(|| format!("sounds: the graph returned no tensor named {OUTPUT_NAME}"))?;
    if logits.len() != labels::COUNT {
        return Err(format!(
            "sounds: the graph returned {} logit(s), expected {}",
            logits.len(),
            labels::COUNT
        ));
    }

    Ok(labels::top_labels(&logits, threshold, top)
        .into_iter()
        .filter_map(|(index, _score)| labels::label(index))
        .collect())
}

/// `current`, sorted, so two labels sets compare equal regardless of the
/// order the model ranked them in.
fn label_set(current: &[&str]) -> Vec<String> {
    let mut set: Vec<String> = current.iter().map(|s| s.to_string()).collect();
    set.sort_unstable();
    set
}

/// True when `current`'s label set is the same as `previous`'s - the
/// window's own emitted rows are then skipped, since an overlapping window
/// repeating the same labels (half the window is shared audio) says
/// nothing an earlier row didn't already. `previous` is `None` before the
/// first window, which is never a repeat.
fn is_repeat(previous: &Option<Vec<String>>, current: &[&str]) -> bool {
    match previous {
        Some(prev) => *prev == label_set(current),
        None => false,
    }
}

/// What `init` settled, plus the graph it loaded.
struct Opened {
    /// The unit this stream's timestamps are counted in.
    time_base: (i32, i32),
    /// Held for the life of the instance: building it once is what keeps a
    /// provider's kernels from being chosen again per window.
    context: GraphExecutionContext,
    /// Kept alive because the context is only valid while its graph is.
    _graph: Graph,
    threshold: f64,
    top: usize,
    /// The second up to which this instance has already sent audio
    /// downstream, so an overlapping window's shared portion is not sent
    /// twice. `None` before the first call.
    emitted_through: Option<f64>,
    /// The previous window's label set, sorted - `None` before the first
    /// window that found any frame to classify. A window whose own set
    /// matches this one emits no rows; see `is_repeat`.
    previous_labels: Option<Vec<String>>,
}

thread_local! {
    static OPENED: RefCell<Option<Opened>> = const { RefCell::new(None) };
}

struct Sounds;

impl Guest for Sounds {
    fn describe() -> WindowMeta {
        WindowMeta {
            meta: Meta {
                name: "sounds".to_string(),
                version: "0.1.0".to_string(),
                params_schema: PARAMS_SCHEMA.to_string(),
                rows_schema: ROWS_SCHEMA.to_string(),
                // An audio module, so it names no pixel formats.
                pixel_formats: vec![],
                sample_formats: vec!["f32".to_string()],
                // What the model was trained on, and the host conforms to it.
                sample_rates: vec![fbank::SAMPLE_RATE],
                channel_counts: vec![1],
                // The rows say what a window sounds like, not what language
                // is spoken in it, so there is nothing here to tag a track
                // with.
                rows_language: vec![],
            },
            window: WINDOW,
            stride: STRIDE,
            // `emitted_through` makes a call's output depend on the calls
            // before it.
            pure: false,
            // The samples pass through as they arrived, just not as `same` -
            // see the module doc comment.
            one_to_one: true,
            reads_rows: false,
            // What leaves is this module's own labels and nothing else.
            forwards_rows: false,
            // One stream in: the audio it listens to.
            inputs: 1,
        }
    }

    fn init(format: Format, stream_info: StreamInfo, params: String) -> Result<(), String> {
        let Format::Audio(audio) = format else {
            return Err("sounds listens to samples, and this stream is video".to_string());
        };
        if audio.sample_fmt != "f32" {
            return Err(format!(
                "sounds does not accept sample format {}",
                audio.sample_fmt
            ));
        }
        if audio.sample_rate != fbank::SAMPLE_RATE {
            return Err(format!(
                "sounds listens at {} Hz, and this instance is {} Hz",
                fbank::SAMPLE_RATE,
                audio.sample_rate
            ));
        }
        if audio.channels != 1 {
            return Err(format!(
                "sounds listens in mono, and this instance has {} channels",
                audio.channels
            ));
        }
        let parsed = parse_params(&params)?;

        // The graph is loaded once per instance, and the session built once:
        // the first window is what a provider picks its kernels on, and
        // every window after it reuses them.
        let graph =
            load_by_name(MODEL).map_err(|e| failed(&format!("load-by-name({MODEL:?})"), &e))?;
        let context = graph
            .init_execution_context()
            .map_err(|e| failed("init-execution-context", &e))?;

        OPENED.with(|o| {
            *o.borrow_mut() = Some(Opened {
                time_base: (stream_info.time_base.num, stream_info.time_base.den),
                context,
                _graph: graph,
                threshold: parsed.threshold,
                top: parsed.top,
                emitted_through: None,
                previous_labels: None,
            });
        });
        Ok(())
    }

    fn set_params(params: String) -> Result<(), String> {
        let parsed = parse_params(&params)?;
        OPENED.with(|o| {
            if let Some(opened) = o.borrow_mut().as_mut() {
                opened.threshold = parsed.threshold;
                opened.top = parsed.top;
            }
        });
        Ok(())
    }

    fn process(window: &InWindow, _trailing: Vec<String>, _last: bool) -> Processed {
        OPENED.with(|o| {
            let mut borrowed = o.borrow_mut();
            let opened = borrowed
                .as_mut()
                .expect("init loads the graph before any audio arrives");
            let Opened {
                time_base,
                context,
                threshold,
                top,
                emitted_through,
                previous_labels,
                ..
            } = opened;

            let mut out: Vec<OutFrame> = Vec::with_capacity(window.len() as usize);
            for index in 0..window.len() {
                let pts = window.pts(index);
                let samples = to_floats(&window.fetch(index));
                let count = samples.len();
                let start_s = seconds(pts, time_base.0, time_base.1);

                let rows = if count == 0 {
                    Vec::new()
                } else {
                    let found = classify(context, &samples, *threshold, *top)
                        .unwrap_or_else(|message| panic!("{message}"));
                    let rows = if is_repeat(previous_labels, &found) {
                        // Same labels as the window before: half this
                        // window's audio is the same audio, so nothing here
                        // is new information.
                        Vec::new()
                    } else {
                        let end_s = start_s + count as f64 / f64::from(fbank::SAMPLE_RATE);
                        found
                            .iter()
                            .map(|&name| row(name, start_s, end_s))
                            .collect()
                    };
                    *previous_labels = Some(label_set(&found));
                    rows
                };

                // How much of this window's front overlaps a window already
                // sent downstream.
                let skip = match *emitted_through {
                    Some(through) if through > start_s => {
                        (((through - start_s) * f64::from(fbank::SAMPLE_RATE)).round() as usize)
                            .min(count)
                    }
                    _ => 0,
                };
                let new_samples = &samples[skip..];
                if !new_samples.is_empty() || !rows.is_empty() {
                    let shift = ticks_of_samples(skip, time_base.0, time_base.1);
                    out.push(OutFrame {
                        pts: pts + shift,
                        frame: FramePayload::New(to_bytes(new_samples)),
                        rows,
                    });
                }
                if count > 0 {
                    *emitted_through = Some(start_s + count as f64 / f64::from(fbank::SAMPLE_RATE));
                }
            }
            Processed {
                frames: out,
                trailing: Vec::new(),
            }
        })
    }
}

export!(Sounds);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_params_at_all_are_the_defaults() {
        for written in ["", "{}", "  "] {
            let parsed = parse_params(written).expect("the defaults");
            assert_eq!(parsed.threshold, 0.3);
            assert_eq!(parsed.top, 3);
        }
    }

    #[test]
    fn each_parameter_can_be_set_on_its_own() {
        let parsed = parse_params(r#"{"threshold":0.8}"#).expect("one of them");
        assert_eq!(parsed.threshold, 0.8);
        assert_eq!(parsed.top, 3, "the rest keep their defaults");

        let parsed = parse_params(r#"{"top":5}"#).expect("the other one");
        assert_eq!(parsed.threshold, 0.3);
        assert_eq!(parsed.top, 5);
    }

    #[test]
    fn a_threshold_outside_a_probability_is_refused() {
        assert!(parse_params(r#"{"threshold":1.5}"#).is_err());
        assert!(parse_params(r#"{"threshold":-0.1}"#).is_err());
    }

    #[test]
    fn a_top_that_is_not_a_non_negative_whole_number_is_refused() {
        assert!(parse_params(r#"{"top":-1}"#).is_err());
        assert!(parse_params(r#"{"top":2.5}"#).is_err());
        assert!(parse_params(r#"{"top":0}"#).is_ok(), "zero is whole");
    }

    #[test]
    fn a_parameter_this_module_does_not_have_is_refused() {
        assert!(parse_params(r#"{"treshold":0.5}"#).is_err());
    }

    #[test]
    fn label_set_ignores_rank_order() {
        assert_eq!(
            label_set(&["Speech", "Music"]),
            label_set(&["Music", "Speech"])
        );
    }

    #[test]
    fn is_repeat_is_false_before_the_first_window() {
        assert!(!is_repeat(&None, &["Speech"]));
    }

    #[test]
    fn is_repeat_is_true_for_the_same_set_in_a_different_order() {
        let previous = Some(label_set(&["Speech", "Music"]));
        assert!(is_repeat(&previous, &["Music", "Speech"]));
    }

    #[test]
    fn is_repeat_is_false_for_a_different_set() {
        let previous = Some(label_set(&["Speech"]));
        assert!(!is_repeat(&previous, &["Music"]));
        // A subset or superset is still a different set.
        assert!(!is_repeat(&previous, &["Speech", "Music"]));
    }

    #[test]
    fn a_label_becomes_a_cue_shaped_row() {
        let written = row("Sine wave", 1.5, 2.25);
        assert_eq!(
            written, r#"{"text":"Sine wave","start_t":1.5,"end_t":2.25}"#,
            "the three columns a cue declares, and nothing else"
        );
    }

    #[test]
    fn the_stride_is_half_the_window() {
        assert_eq!(WINDOW, STRIDE * 2, "consecutive clips overlap by half");
        assert_eq!(WINDOW, fbank::SAMPLE_RATE * 10);
    }

    #[test]
    fn floats_survive_the_trip_through_a_tensors_bytes() {
        let values = [0.0f32, -1.0, 0.5, f32::MIN_POSITIVE];
        assert_eq!(to_floats(&to_bytes(&values)), values);
    }

    #[test]
    fn a_timestamp_becomes_seconds_in_the_streams_own_unit() {
        assert_eq!(seconds(16_000, 1, 16_000), 1.0);
        assert!((seconds(0, 1, 16_000) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn samples_become_ticks_at_one_tick_per_sample() {
        // The documented audio contract: at 1/48000 one tick is one sample,
        // and this module only ever opens at 1/SAMPLE_RATE.
        assert_eq!(
            ticks_of_samples(80_000, 1, fbank::SAMPLE_RATE as i32),
            80_000
        );
        assert_eq!(ticks_of_samples(0, 1, fbank::SAMPLE_RATE as i32), 0);
    }
}
