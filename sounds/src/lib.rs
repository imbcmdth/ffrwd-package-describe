//! What AudioSet class the audio sounds like: each 10 s window's labels above
//! a threshold leave on `labels` as cues. Nothing else leaves; a reader takes
//! the sound from its source.
//!
//! The graph is the Audio Spectrogram Transformer, run through `wasi:nn`.
//! The module never opens a file: the host binds the graph to a name with
//! `-nn sounds=<path>` and this module asks for that name and nothing else.
//! The 527 AudioSet class names travel as source (`audioset-labels.txt`,
//! `src/labels.rs`) rather than as a second model file: the module imports no
//! filesystem, so a file the package installed beside the graph is as
//! unreachable as everything else on disk.
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
//! The fbank feature the AST checkpoint reads is the same either way:
//! `src/fbank.rs` doesn't know about the overlap, only about however many
//! samples one call hands it.
//!
//! # The rows
//!
//! One cue per label kept: `text`, `start_t`, `end_t`, the same shape
//! `ffrwd/vad`'s spans and `ffrwd/whisper`'s captions use, so one `embed()`
//! reads any of the three. `text` is the AudioSet class name; the model
//! gives no timing finer than the window itself, so `start_t`/`end_t` are
//! the window's own span, in seconds from the start of the stream, and each
//! leaves with its window. `top` labels are kept out of however many clear
//! `threshold`, ranked by score.
//!
//! A window whose label set is exactly the window before's emits no rows:
//! consecutive windows overlap by half, so a sound that hasn't changed says
//! the same thing twice for free, and a caller reading these rows back
//! would otherwise see one label repeated across every window it spans.
//! The comparison ignores rank order: the same labels in a different order
//! still count as the same set. That memory of the window before is why the
//! node is not pure.

mod fbank;
mod labels;

use describe_common::{le_bytes, le_f32s, Element, Feed, Model};
use ffrwd_node::{Bound, Cue, Init, Input, Node, Out, Output, Result, Shape, Tick};
use serde::Deserialize;

/// The name the host binds the graph to. `-nn sounds=<path>`.
const MODEL: &str = "sounds";

/// The graph's own names for the tensors it takes and returns.
const INPUT_NAME: &str = "input_values";
const OUTPUT_NAME: &str = "logits";

/// 10 s at 16 kHz: the clip AST classifies in one call.
const WINDOW: u32 = fbank::SAMPLE_RATE * 10;
/// 5 s at 16 kHz: half the window, so consecutive clips overlap.
const STRIDE: u32 = fbank::SAMPLE_RATE * 5;

const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"threshold":{"type":"number","minimum":0,"maximum":1,"default":0.3},"top":{"type":"integer","minimum":0,"default":3}},"additionalProperties":false}"#;

#[derive(Deserialize)]
struct Params {
    threshold: f64,
    top: usize,
}

/// One window through the graph: its AudioSet labels at or above
/// `threshold`, the highest-scoring `top` of them, ranked score first.
fn classify(
    model: &Model,
    samples: &[f32],
    threshold: f64,
    top: usize,
) -> Result<Vec<&'static str>, String> {
    if fbank::num_frames(samples.len()) == 0 {
        // Not even one 25 ms frame: nothing here but padding, and the model
        // has nothing worth guessing at.
        return Ok(Vec::new());
    }
    let features = le_bytes(&fbank::extract(samples));
    let dims = [1, fbank::MAX_FRAMES as u32, fbank::NUM_MEL_BINS as u32];
    let outputs = model.run(&[Feed {
        name: INPUT_NAME,
        dimensions: &dims,
        element: Element::F32,
        data: &features,
    }])?;

    let logits = outputs
        .iter()
        .find(|(name, _)| name == OUTPUT_NAME)
        .map(|(_, data)| le_f32s(data))
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

/// True when `current`'s label set is the same as `previous`'s: the
/// window's own rows are then skipped, since an overlapping window
/// repeating the same labels (half the window is shared audio) says
/// nothing an earlier row didn't already. `previous` is `None` before the
/// first window, which is never a repeat.
fn is_repeat(previous: &Option<Vec<String>>, current: &[&str]) -> bool {
    match previous {
        Some(prev) => *prev == label_set(current),
        None => false,
    }
}

struct Sounds {
    a: u32,
    model: Model,
    threshold: f64,
    top: usize,
    /// The previous window's label set, sorted: `None` before the first
    /// window that found any frame to classify. A window whose own set
    /// matches this one emits no rows; see `is_repeat`.
    previous_labels: Option<Vec<String>>,
}

impl Node for Sounds {
    const NAME: &'static str = "sounds";
    const VERSION: &'static str = "0.2.0";
    const PARAMS_SCHEMA: &'static str = PARAMS_SCHEMA;
    type Params = Params;

    fn shape(_: &Params, _: &Bound) -> Result<Shape> {
        Ok(Shape::new()
            .input(
                Input::audio("a")
                    .clock()
                    .window(WINDOW, STRIDE)
                    .sample_formats(&["f32"])
                    .sample_rates(&[fbank::SAMPLE_RATE])
                    .channel_counts(&[1]),
            )
            .output(Output::rows("labels").schema::<Cue>()))
    }

    fn init(params: Params, init: &Init) -> Result<Sounds> {
        let a = init.stream("a")?;
        let audio = a
            .audio_format()
            .ok_or("sounds listens to samples, and `a` is not audio")?;
        if audio.sample_fmt != "f32" {
            return Err(
                format!("sounds does not accept sample format {}", audio.sample_fmt).into(),
            );
        }
        if audio.sample_rate != fbank::SAMPLE_RATE || audio.channels != 1 {
            return Err(format!(
                "sounds listens at {} Hz in mono, and `a` is {} Hz in {} channels",
                fbank::SAMPLE_RATE,
                audio.sample_rate,
                audio.channels
            )
            .into());
        }
        Ok(Sounds {
            a: a.id,
            model: Model::load("sounds", MODEL)?,
            threshold: params.threshold,
            top: params.top,
            previous_labels: None,
        })
    }

    fn set_params(&mut self, params: Params) -> Result<()> {
        self.threshold = params.threshold;
        self.top = params.top;
        Ok(())
    }

    fn process(&mut self, tick: &Tick, out: &mut Out) -> Result<()> {
        let time_base = tick.time_base();
        for frame in tick.frames(self.a) {
            let samples = le_f32s(&tick.fetch(self.a, frame.index));
            if samples.is_empty() {
                continue;
            }
            let found = classify(&self.model, &samples, self.threshold, self.top)?;
            if !is_repeat(&self.previous_labels, &found) {
                let start_t = time_base.seconds(frame.pts);
                let end_t = start_t + samples.len() as f64 / f64::from(fbank::SAMPLE_RATE);
                for name in &found {
                    out.row("labels", frame.pts, &Cue::new(start_t, end_t, *name))?;
                }
            }
            self.previous_labels = Some(label_set(&found));
        }
        Ok(())
    }
}

ffrwd_node::export!(Sounds);

#[cfg(test)]
mod tests {
    use super::*;
    use ffrwd_node::{Kind, Runner};

    fn shape(params: &str) -> std::result::Result<Shape, String> {
        Runner::<Sounds>::shape(params, &["a".to_owned()])
    }

    #[test]
    fn a_hopping_window_of_mono_16k_in_and_cues_out() {
        let shape = shape("").expect("a shape");
        let a = shape.find_input("a").expect("the sound");
        assert_eq!((a.window, a.stride), (WINDOW, STRIDE));
        assert_eq!(a.accepts.sample_formats, ["f32"]);
        assert_eq!(a.accepts.sample_rates, [16_000]);
        assert_eq!(a.accepts.channel_counts, [1]);
        assert_eq!(shape.outputs.len(), 1, "no sound leaves");
        let labels = &shape.outputs[0];
        assert_eq!((labels.name.as_str(), labels.kind), ("labels", Kind::Data));
        assert_eq!(labels.latency, 0.0, "a window's cues leave with it");
        assert!(!shape.pure, "the window before is remembered");
    }

    #[test]
    fn no_params_at_all_are_the_defaults() {
        for written in ["", "{}"] {
            let (params, _) =
                ffrwd_node::read_params::<Params>(PARAMS_SCHEMA, written).expect("the defaults");
            assert_eq!(params.threshold, 0.3);
            assert_eq!(params.top, 3);
        }
    }

    #[test]
    fn each_parameter_can_be_set_on_its_own() {
        let (params, _) =
            ffrwd_node::read_params::<Params>(PARAMS_SCHEMA, r#"{"threshold":0.8}"#).expect("one");
        assert_eq!((params.threshold, params.top), (0.8, 3));
        let (params, _) =
            ffrwd_node::read_params::<Params>(PARAMS_SCHEMA, r#"{"top":5}"#).expect("the other");
        assert_eq!((params.threshold, params.top), (0.3, 5));
    }

    #[test]
    fn a_threshold_outside_a_probability_is_refused() {
        assert!(shape(r#"{"threshold":1.5}"#).is_err());
        assert!(shape(r#"{"threshold":-0.1}"#).is_err());
    }

    #[test]
    fn a_top_that_is_not_a_non_negative_whole_number_is_refused() {
        assert!(shape(r#"{"top":-1}"#).is_err());
        assert!(shape(r#"{"top":2.5}"#).is_err());
        assert!(shape(r#"{"top":0}"#).is_ok(), "zero is whole");
        assert!(
            shape(r#"{"top":4.0}"#).is_ok(),
            "a whole number written as one"
        );
    }

    #[test]
    fn a_parameter_this_module_does_not_have_is_refused() {
        assert!(shape(r#"{"treshold":0.5}"#).is_err());
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
        let written = serde_json::to_string(&Cue::new(1.5, 2.25, "Sine wave")).expect("a row");
        assert_eq!(
            written, r#"{"start_t":1.5,"end_t":2.25,"text":"Sine wave"}"#,
            "the three columns a cue declares, and nothing else"
        );
    }

    #[test]
    fn the_stride_is_half_the_window() {
        assert_eq!(WINDOW, STRIDE * 2, "consecutive clips overlap by half");
        assert_eq!(WINDOW, fbank::SAMPLE_RATE * 10);
    }
}
