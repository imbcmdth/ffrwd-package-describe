//! X-CLIP's video tower, one shot at a time. Each shot leaves one row on
//! `shots`: its own seconds and a 512-d vector, unit length, in the space
//! `embed_clip_text` writes into. Nothing else leaves; a reader takes the
//! picture from its source.
//!
//! Where one shot ends and the next begins is not this module's to decide: a
//! `{"shot": n}` row on `cuts` beside a frame (`ffrwd.shots.simple_detector`,
//! read from the same picture) closes the shot running when its value
//! changes. A repeated value does not, and a frame with no such row leaves
//! the shot open. With `cuts` unbound, the whole input is one shot.
//!
//! A shot is described by eight frames evenly spread over it, resized and
//! normalized the way the checkpoint's own preprocessing declares. The eight
//! are chosen while the shot runs (see `reservoir`) and the tower runs once
//! when the shot ends: at a new shot value, at ten seconds, or when the
//! stream does. A shot longer than ten seconds therefore leaves several rows,
//! each covering its own stretch, and a row leaves as late as the cap and the
//! frame that reaches it.
//!
//! Frames arrive as rgba, or as yuv420p converted here by ffrwd-frame in the
//! range and matrix the stream declares.
//!
//! The `gpu` feature builds `clips_gpu`, the same node with a `memory` param,
//! whose frames the host makes into the tower's input (see `gpu`): a frame
//! the reservoir keeps is held as the host's tensor rather than as floats
//! here, and the eight are joined by the host, so the module never fetches a
//! frame.

mod prep;
mod reservoir;

#[cfg(feature = "gpu")]
mod gpu;

#[cfg(not(feature = "gpu"))]
use describe_common::{le_bytes, Element, Feed};
use describe_common::{le_f32s, normalize, Model};
#[cfg(not(feature = "gpu"))]
use ffrwd_frame::yuv::{self, Colour, Yuv420p};
#[cfg(not(feature = "gpu"))]
use ffrwd_frame::Rgba;
#[cfg(not(feature = "gpu"))]
use ffrwd_node::NoParams;
use ffrwd_node::{Bound, Init, Input, Node, Out, Output, Rational, Result, Shape, Tick};
use serde::{Deserialize, Serialize};

#[cfg(not(feature = "gpu"))]
use prep::{SIDE, TENSOR};
use reservoir::{Reservoir, FRAMES};

/// The GPU build's params: where `v` is held.
#[cfg(feature = "gpu")]
const PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"memory":{"type":"string","enum":["host","gpu"],"default":"host"}},"additionalProperties":false}"#;

/// What a kept frame is held as: its tensor's floats, made here.
#[cfg(not(feature = "gpu"))]
type Picture = Vec<f32>;

/// What a kept frame is held as: the host's tensor of it.
#[cfg(feature = "gpu")]
type Picture = gpu::Tensor;

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

/// The frame past the cap that closes a stretch, where the call leaves the
/// clock's rate unknown: half a second covers 2 fps and up.
const FRAME_MARGIN: f64 = 0.5;

/// How late a row may leave: the cap, and the frame that reaches it, which
/// is one frame of the clock's rate where the call says what that is.
fn latency(rate: Option<Rational>) -> f64 {
    CAP + rate.map_or(FRAME_MARGIN, |rate| rate.duration(1))
}

// `vector`'s minItems/maxItems fix EMBEDDING as the rows' `vector_dims`: the
// compiler reads them at compile time, since it cannot count a run-time
// module's rows itself.
const ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"start_t":{"type":"number"},"end_t":{"type":"number"},"vector":{"type":"array","items":{"type":"number"},"minItems":512,"maxItems":512}},"required":["start_t","end_t","vector"],"additionalProperties":false}"#;

/// One shot's row: the seconds it covers, and what the tower made of it.
#[derive(Debug, Serialize)]
struct Row {
    start_t: f64,
    end_t: f64,
    vector: Vec<f32>,
}

/// A row naming which shot its frame belongs to. A producer's other fields
/// pass unread.
#[derive(Default, Deserialize, Serialize)]
struct Cut {
    shot: i64,
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
    frames: Reservoir<Picture>,
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
/// nothing; there is no stretch yet.
fn ends_stretch(cut: bool, segment: Option<&Segment>, pts: i64, tick: f64) -> bool {
    let Some(segment) = segment else {
        return false;
    };
    cut || (pts - segment.start_pts) as f64 * tick >= CAP
}

/// Seconds one timestamp unit is worth, which is what turns a frame's pts
/// into the seconds a row carries.
fn seconds_per_tick(time_base: Rational) -> f64 {
    f64::from(time_base.num) / f64::from(time_base.den)
}

#[cfg(not(feature = "gpu"))]
#[derive(Clone, Copy, Debug, PartialEq)]
enum Arrives {
    Rgba,
    Yuv420p(Colour),
}

#[cfg(not(feature = "gpu"))]
fn colour(color: Option<&ffrwd_node::ColorInfo>) -> Result<Colour, String> {
    Colour::of(color.map(|color| yuv::ColorInfo {
        range: &color.range,
        primaries: &color.primaries,
        trc: &color.trc,
        space: &color.space,
    }))
}

/// One frame turned into the tower's tensor, by way of rgba.
#[cfg(not(feature = "gpu"))]
fn tensor_of(
    arrives: Arrives,
    bytes: &[u8],
    scratch: &mut [u8],
    width: usize,
    height: usize,
) -> Result<Vec<f32>, String> {
    let picture = match arrives {
        Arrives::Rgba => Rgba::new(bytes, width, height)?,
        Arrives::Yuv420p(colour) => {
            yuv::to_rgba(&Yuv420p::new(bytes, width, height)?, colour, scratch)?;
            Rgba::new(scratch, width, height)?
        }
    };
    Ok(prep::preprocess(&picture))
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

struct Clips {
    v: u32,
    cuts: Option<u32>,
    #[cfg(not(feature = "gpu"))]
    width: usize,
    #[cfg(not(feature = "gpu"))]
    height: usize,
    #[cfg(not(feature = "gpu"))]
    arrives: Arrives,
    #[cfg(not(feature = "gpu"))]
    rgba: Vec<u8>,
    /// Seconds one timestamp unit is worth, from the stream's time base.
    tick: f64,
    /// The shot value running, absent until a shot row has arrived.
    current_shot: Option<i64>,
    /// The newest frame's bytes, kept so a shot's last frame can join the
    /// eight even when the reservoir's spacing passed over it.
    #[cfg(not(feature = "gpu"))]
    newest: Vec<u8>,
    /// The newest frame's tensor where the reservoir passed over it, kept so
    /// a shot's last frame can join the eight: by the time a cut says which
    /// frame was the last, that frame's tick has gone.
    #[cfg(feature = "gpu")]
    newest: Option<gpu::Tensor>,
    /// How the host makes a frame into a tensor.
    #[cfg(feature = "gpu")]
    maker: gpu::Maker,
    /// Where `v` is held, settled at init.
    #[cfg(feature = "gpu")]
    memory: gpu::Memory,
    segment: Option<Segment>,
    model: Model,
}

impl Clips {
    /// One clip through the tower: the eight frames `picks` names of those
    /// kept in, one unit vector out.
    #[cfg(not(feature = "gpu"))]
    fn run(&self, kept: &[Picture], picks: &[usize; FRAMES]) -> Result<Vec<f32>, String> {
        let mut input = Vec::with_capacity(FRAMES * TENSOR);
        for &at in picks {
            input.extend_from_slice(&kept[at]);
        }
        debug_assert_eq!(input.len(), FRAMES * TENSOR);
        let dims = [1, FRAMES as u32, 3, SIDE as u32, SIDE as u32];
        let bytes = le_bytes(&input);
        let outputs = self.model.run(&[Feed {
            name: INPUT_NAME,
            dimensions: &dims,
            element: Element::F32,
            data: &bytes,
        }])?;
        Ok(normalize(take(outputs)?))
    }

    /// One clip through the tower: the eight frames `picks` names of those
    /// kept, joined by the host, in; one unit vector out.
    #[cfg(feature = "gpu")]
    fn run(&self, kept: &[Picture], picks: &[usize; FRAMES]) -> Result<Vec<f32>, String> {
        let input = gpu::clip(kept, picks)?;
        let outputs = self.model.run_tensor(INPUT_NAME, input)?;
        Ok(normalize(take(outputs)?))
    }

    /// The shot's last frame, where the reservoir passed over it.
    #[cfg(not(feature = "gpu"))]
    fn last(&mut self) -> Result<Picture, String> {
        tensor_of(
            self.arrives,
            &self.newest,
            &mut self.rgba,
            self.width,
            self.height,
        )
    }

    /// The shot's last frame, where the reservoir passed over it.
    #[cfg(feature = "gpu")]
    fn last(&mut self) -> Result<Picture, String> {
        self.newest
            .take()
            .ok_or_else(|| "clips: the shot's last frame was not kept".to_string())
    }

    /// The row a finished stretch leaves behind, the tower run once for it.
    fn close(&mut self, mut segment: Segment) -> Result<Row, String> {
        let last = if segment.frames.holds_last() {
            None
        } else {
            Some(self.last()?)
        };
        let (kept, picks) = segment.frames.close(last);
        Ok(Row {
            start_t: segment.start_pts as f64 * self.tick,
            end_t: segment.last_pts as f64 * self.tick,
            vector: self.run(&kept, &picks)?,
        })
    }

    /// One frame's time and shot: the row a new shot or the cap ends, and the
    /// stretch the frame belongs to begun or carried on. The frame itself
    /// joins it with `keep`.
    fn advance(&mut self, pts: i64, shot: Option<i64>) -> Result<Option<Row>, String> {
        let cut = is_new_shot(self.current_shot, shot);
        if let Some(shot) = shot {
            self.current_shot = Some(shot);
        }

        let mut row = None;
        if ends_stretch(cut, self.segment.as_ref(), pts, self.tick) {
            let segment = self.segment.take().expect("a stretch was running");
            row = Some(self.close(segment)?);
        }

        let segment = self.segment.get_or_insert_with(|| Segment::new(pts));
        segment.last_pts = pts;
        Ok(row)
    }

    /// The frame's bytes offered to the running stretch, and kept as the
    /// newest.
    #[cfg(not(feature = "gpu"))]
    fn keep(&mut self, frame: &[u8]) -> Result<(), String> {
        let (arrives, width, height) = (self.arrives, self.width, self.height);
        let scratch = &mut self.rgba;
        let segment = self.segment.as_mut().expect("a stretch is running");
        segment
            .frames
            .offer(|| tensor_of(arrives, frame, scratch, width, height))?;

        self.newest.clear();
        self.newest.extend_from_slice(frame);
        Ok(())
    }

    /// Frame `index` of `v` offered to the running stretch, the host making
    /// it into a tensor where the reservoir keeps it, or where it does not,
    /// kept as the newest.
    #[cfg(feature = "gpu")]
    fn keep(&mut self, index: u32) -> Result<(), String> {
        let maker = &self.maker;
        let segment = self.segment.as_mut().expect("a stretch is running");
        segment.frames.offer(|| maker.tensor(index))?;
        self.newest = if segment.frames.holds_last() {
            None
        } else {
            Some(maker.tensor(index)?)
        };
        Ok(())
    }
}

fn emit(out: &mut Out, row: &Row) -> Result<(), String> {
    let pts = out.pts("shots", row.start_t)?;
    out.row("shots", pts, row)
}

/// The formats `v` takes: yuv420p, converted here, or rgba.
#[cfg(not(feature = "gpu"))]
fn pixel_formats(_: &NoParams) -> &'static [&'static str] {
    &["yuv420p", "rgba"]
}

#[cfg(feature = "gpu")]
use gpu::pixel_formats;

impl Node for Clips {
    const NAME: &'static str = "clips";
    const VERSION: &'static str = "0.2.0";
    #[cfg(not(feature = "gpu"))]
    type Params = NoParams;
    #[cfg(feature = "gpu")]
    const PARAMS_SCHEMA: &'static str = PARAMS_SCHEMA;
    #[cfg(feature = "gpu")]
    type Params = gpu::Settings;

    fn shape(params: &Self::Params, bound: &Bound) -> Result<Shape> {
        Ok(Shape::new()
            .input(
                Input::video("v")
                    .clock()
                    .pixel_formats(pixel_formats(params)),
            )
            .input(Input::rows("cuts").optional().lockstep().schema::<Cut>())
            .output(
                Output::rows("shots")
                    .time_base(Rational::MICROS)
                    .latency(latency(bound.rate_of("v")))
                    .schema_json(ROWS_SCHEMA),
            ))
    }

    #[cfg(not(feature = "gpu"))]
    fn init(_: NoParams, init: &Init) -> Result<Clips> {
        let v = init.stream("v")?;
        let video = v
            .video_format()
            .ok_or("clips reads pictures, and `v` is not video")?;
        let (width, height) = (video.width as usize, video.height as usize);
        let arrives = match video.pix_fmt.as_str() {
            "rgba" => Arrives::Rgba,
            "yuv420p" => Arrives::Yuv420p(colour(video.color.as_ref())?),
            other => return Err(format!("clips does not accept pixel format {other}").into()),
        };
        Ok(Clips {
            v: v.id,
            cuts: init.optional("cuts").map(|cuts| cuts.id),
            width,
            height,
            arrives,
            rgba: match arrives {
                Arrives::Rgba => Vec::new(),
                Arrives::Yuv420p(_) => vec![0; width * height * 4],
            },
            tick: seconds_per_tick(v.info.time_base),
            current_shot: None,
            newest: Vec::new(),
            segment: None,
            model: Model::load("clips", MODEL)?,
        })
    }

    #[cfg(feature = "gpu")]
    fn init(settings: gpu::Settings, init: &Init) -> Result<Clips> {
        let v = init.stream("v")?;
        let video = v
            .video_format()
            .ok_or("clips reads pictures, and `v` is not video")?;
        Ok(Clips {
            v: v.id,
            cuts: init.optional("cuts").map(|cuts| cuts.id),
            tick: seconds_per_tick(v.info.time_base),
            current_shot: None,
            newest: None,
            maker: gpu::Maker::new(v.id, video.width, video.height, MODEL),
            memory: settings.memory,
            segment: None,
            model: Model::load("clips", MODEL)?,
        })
    }

    #[cfg(feature = "gpu")]
    fn set_params(&mut self, settings: gpu::Settings) -> Result<()> {
        // Where `v` is held is the shape's, settled at init.
        if settings.memory != self.memory {
            return Err("clips cannot move `v` between host and gpu memory while it runs".into());
        }
        Ok(())
    }

    fn process(&mut self, tick: &Tick, out: &mut Out) -> Result<()> {
        let shot = match self.cuts {
            Some(cuts) => tick.rows::<Cut>(cuts)?.last().map(|cut| cut.shot),
            None => None,
        };
        for frame in tick.frames(self.v) {
            let row = self.advance(frame.pts, shot)?;
            if let Some(row) = row {
                emit(out, &row)?;
            }
            // The host makes the frame into a tensor wherever it is held, so
            // the frame is never fetched here.
            #[cfg(feature = "gpu")]
            self.keep(frame.index)?;
            #[cfg(not(feature = "gpu"))]
            self.keep(&tick.fetch(self.v, frame.index))?;
        }
        if tick.last() {
            if let Some(segment) = self.segment.take() {
                let row = self.close(segment)?;
                emit(out, &row)?;
            }
        }
        Ok(())
    }
}

ffrwd_node::export!(Clips);

#[cfg(test)]
mod tests {
    use super::*;
    use ffrwd_node::{Clock, Kind, Pairing, Runner};

    #[test]
    fn the_shots_leave_as_rows_alone_and_the_cuts_are_optional() {
        let bound = ["v".to_owned(), "cuts".to_owned()];
        let shape = Runner::<Clips>::shape("", &bound).expect("a shape");
        assert_eq!(shape.clock, Some(Clock::Input("v".to_owned())));
        let cuts = shape.find_input("cuts").expect("a cuts port");
        assert_eq!(cuts.kind, Kind::Data);
        assert!(!cuts.required);
        assert_eq!(cuts.pairing, Pairing::Lockstep);
        assert!(cuts.schema.as_deref().is_some_and(|s| s.contains("shot")));
        assert_eq!(shape.outputs.len(), 1, "no picture leaves");
        let shots = &shape.outputs[0];
        assert_eq!(shots.name, "shots");
        assert_eq!(shots.time_base, Some(Rational::new(1, 1_000_000)));
        assert!(shots.latency > CAP, "a row leaves a frame past the cap");
        assert!(!shape.pure, "the reservoir and the shot running carry over");
        assert!(Runner::<Clips>::shape("", &["v".to_owned()]).is_ok());
    }

    #[test]
    fn a_row_leaves_one_frame_of_the_bound_rate_past_the_cap() {
        let at = |rate: Option<Rational>| {
            let bound = match rate {
                Some(rate) => Bound::new(&["v", "cuts"]).rate("v", rate),
                None => Bound::new(&["v", "cuts"]),
            };
            Runner::<Clips>::shape("", &bound).expect("a shape").outputs[0].latency
        };
        assert_eq!(at(Some(Rational::new(25, 1))), 10.04);
        assert_eq!(
            at(Some(Rational::new(30000, 1001))),
            10.0 + 1001.0 / 30000.0
        );
        assert_eq!(at(Some(Rational::new(2, 1))), 10.5);
        assert_eq!(at(None), 10.5, "half a second where the rate is unknown");
    }

    #[test]
    fn params_are_refused_because_there_are_none() {
        let error = Runner::<Clips>::shape(r#"{"threshold":9}"#, &["v".to_owned()])
            .expect_err("no params exist");
        assert!(error.contains("threshold"), "{error}");
    }

    #[test]
    fn the_time_base_field_becomes_seconds_per_tick() {
        assert!((seconds_per_tick(Rational::new(1, 15360)) * 15360.0 - 1.0).abs() < 1e-12);
        assert!((seconds_per_tick(Rational::new(1001, 30000)) * 30000.0 - 1001.0).abs() < 1e-9);
    }

    #[cfg(not(feature = "gpu"))]
    #[test]
    fn a_tv_range_picture_reaches_the_tower_at_full_contrast() {
        let tv = Arrives::Yuv420p(
            colour(Some(&ffrwd_node::ColorInfo {
                range: "tv".to_owned(),
                primaries: "bt709".to_owned(),
                trc: "bt709".to_owned(),
                space: "bt709".to_owned(),
            }))
            .expect("tv 709"),
        );
        let (width, height) = (64usize, 48usize);
        let mut white = vec![128u8; Yuv420p::size(width, height)];
        white[..width * height].fill(235);
        let mut scratch = vec![0u8; width * height * 4];
        let tensor = tensor_of(tv, &white, &mut scratch, width, height).expect("converts");
        let full = (1.0 - 0.485) / 0.229;
        assert!(
            (tensor[0] - full).abs() < 1e-4,
            "235 is white: {}",
            tensor[0]
        );
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
    fn a_cut_row_reads_its_shot_and_lets_a_producers_other_fields_pass() {
        let cut: Cut = ffrwd_node::parse(r#"{"start_t":1.5,"shot":3}"#).expect("a cut");
        assert_eq!(cut.shot, 3);
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
