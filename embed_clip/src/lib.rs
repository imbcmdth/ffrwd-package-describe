//! X-CLIP's text tower, reached two ways.
//!
//! `embed_clip_text` is a value function: one prompt in, one 512-d vector
//! out, computed at compile time like any other value call, ready to rank
//! against `clips`'s video-tower vectors. `embed_clip` is a node clocked by
//! its rows: it reads whatever produced its `cue[]` argument and writes one
//! vector a row on `vectors`, in the same clip space, at the row's own time. Both tokenize their text the
//! same way and run the same graph; the manifest pins the one ONNX file
//! both load, under the export name `embed_clip`.
//!
//! This is a different space from `embed`/`embed_text`, which put a
//! sentence embedder under sound and speech rows: a CLIP text tower is
//! trained to match images, not sentences, so it ranks a cue's short label
//! well but not a spoken sentence - `embed_clip`/`embed_clip_text` exist for
//! callers that want the clip space on purpose, not as the default row
//! embedder.
//!
//! Every vector this module returns is L2-normalized, so `cos_similarity`
//! between any two of them - or between one of these and a video-tower
//! vector from `clips` - is a plain dot product.
//!
//! The tokenizer is CLIP's own byte-level BPE: vocab 49408, `<|startoftext|>`
//! (49406) and `<|endoftext|>` (49407) as bos/eos, lowercased, the `</w>`
//! word-end convention. Rather than hand-writing that BPE, this module
//! embeds the `tokenizers` crate's own `tokenizer.json` - fetched from the
//! source model's revision, `assets/fetch-tokenizer.py` - compiled in with
//! `include_bytes!`. A wasm module has no filesystem: `wasi-nn`'s
//! `load-by-name` reaches the ONNX graph by name alone, and there is no
//! parallel mechanism for a non-graph file, so the tokenizer's vocabulary
//! travels inside the module rather than beside it.

// The values export rides on bindings of its own beside the node ffrwd-node
// exports; `../wit` is ffrwd:av@0.19.0 as the SDK carries it, byte for byte.
wit_bindgen::generate!({
    path: ["../wit", "wit-world"],
    world: "ffrwd:embed-clip/embed-clip",
    generate_all,
});

use std::cell::RefCell;
use std::sync::OnceLock;

use describe_common::{le_f32s, normalize, Element, Feed, Model};
use exports::ffrwd::av::values::{FunctionMeta, Guest as ValuesGuest};
use ffrwd_node::{
    Bound, Init, Input, NoParams, Node, Out, Output, Result as NodeResult, Shape, Tick,
};
use serde::{Deserialize, Serialize};
use tokenizers::{
    PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection, TruncationParams,
    TruncationStrategy,
};

/// The name the host binds the graph to: `-nn embed_clip=<path>`.
const MODEL: &str = "embed_clip";

/// What the export calls its two inputs, both `[1, MAX_TOKENS]` i64.
const INPUT_IDS: &str = "input_ids";
const ATTENTION_MASK: &str = "attention_mask";
/// What the export calls its output, `[1, VECTOR_LEN]` f32.
const OUTPUT_NAME: &str = "text_embeds";

/// CLIP's own fixed context length: every prompt is padded or truncated to
/// exactly this many tokens.
const MAX_TOKENS: usize = 77;
/// The tower's output width.
const VECTOR_LEN: usize = 512;
/// `<|endoftext|>`, doubling as the pad token - the tokenizer's own
/// convention, matched here rather than invented.
const EOS_ID: u32 = 49407;
const EOS_TOKEN: &str = "<|endoftext|>";

/// CLIP's byte-level BPE, exactly as `microsoft/xclip-base-patch16-kinetics-600`
/// published it at revision e4921c41fc296102aae210d43d4127c5e3e51928 -
/// `assets/fetch-tokenizer.py` re-fetches it, hash-checked. Compiled in
/// because the module has no other way to reach it; see the module doc
/// comment.
const TOKENIZER_JSON: &[u8] = include_bytes!("../assets/tokenizer.json");

/// The tokenizer, built twice from the same bytes: `raw` reports a prompt's
/// true token count, unpadded and untruncated, which is what tells whether
/// a prompt ran past `MAX_TOKENS`; `padded` is truncated and padded to
/// exactly `MAX_TOKENS`, which is what the graph's fixed-shape inputs need.
/// Built once per instance and kept for every call after.
struct Tok {
    raw: Tokenizer,
    padded: Tokenizer,
}

fn tok() -> &'static Tok {
    static TOK: OnceLock<Tok> = OnceLock::new();
    TOK.get_or_init(|| {
        let raw = Tokenizer::from_bytes(TOKENIZER_JSON).expect("embedded tokenizer.json parses");
        let mut padded =
            Tokenizer::from_bytes(TOKENIZER_JSON).expect("embedded tokenizer.json parses");
        padded
            .with_truncation(Some(TruncationParams {
                max_length: MAX_TOKENS,
                strategy: TruncationStrategy::LongestFirst,
                direction: TruncationDirection::Right,
                stride: 0,
            }))
            .expect("truncation params are well-formed");
        padded.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(MAX_TOKENS),
            direction: tokenizers::PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id: EOS_ID,
            pad_type_id: 0,
            pad_token: EOS_TOKEN.to_string(),
        }));
        Tok { raw, padded }
    })
}

/// One prompt, tokenized to the tower's fixed shape: `ids` and `mask` are
/// both exactly `MAX_TOKENS` long. `truncated` is true when the prompt's
/// true token count - bos and eos included - ran past `MAX_TOKENS` and some
/// of its text was therefore dropped.
struct Tokenized {
    ids: Vec<i64>,
    mask: Vec<i64>,
    truncated: bool,
}

fn tokenize(text: &str) -> Result<Tokenized, String> {
    let t = tok();
    let true_len = t
        .raw
        .encode(text, true)
        .map_err(|e| format!("tokenize: {e}"))?
        .get_ids()
        .len();
    let encoding = t
        .padded
        .encode(text, true)
        .map_err(|e| format!("tokenize: {e}"))?;
    let ids = encoding.get_ids().iter().map(|&id| id as i64).collect();
    let mask = encoding
        .get_attention_mask()
        .iter()
        .map(|&m| m as i64)
        .collect();
    Ok(Tokenized {
        ids,
        mask,
        truncated: true_len > MAX_TOKENS,
    })
}

/// The named tensor's bytes out of what a compute answered, checked against
/// the element count its shape fixes. Takes plain bytes rather than the
/// wit-bindgen `Tensor` resource - `run_tower` converts immediately after
/// `compute`, which is what lets this function (and its tests) run as
/// ordinary Rust: `Tensor` is a host resource, unreachable outside a real
/// wasm import.
fn take_output(
    mut outputs: Vec<(String, Vec<u8>)>,
    name: &str,
    len: usize,
) -> Result<Vec<f32>, String> {
    let Some(pos) = outputs.iter().position(|(n, _)| n == name) else {
        let unclaimed: Vec<&str> = outputs.iter().map(|(n, _)| n.as_str()).collect();
        return Err(format!(
            "embed_clip: the graph answered without {name:?} (unclaimed: {unclaimed:?}); \
             this module wants X-CLIP's text tower export"
        ));
    };
    let (_, bytes) = outputs.swap_remove(pos);
    let values = le_f32s(&bytes);
    if values.len() != len {
        return Err(format!(
            "embed_clip: {name} came back as {} numbers, expected {len}",
            values.len()
        ));
    }
    Ok(values)
}

/// One tokenized prompt through the graph: the raw, un-normalized 512-d
/// output.
fn run_tower(model: &Model, ids: &[i64], mask: &[i64]) -> Result<Vec<f32>, String> {
    let dims = [1, MAX_TOKENS as u32];
    let ids_bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mask_bytes: Vec<u8> = mask.iter().flat_map(|v| v.to_le_bytes()).collect();
    let feed = |name, data| Feed {
        name,
        dimensions: &dims,
        element: Element::I64,
        data,
    };
    let outputs = model.run(&[
        feed(INPUT_IDS, &ids_bytes),
        feed(ATTENTION_MASK, &mask_bytes),
    ])?;
    take_output(outputs, OUTPUT_NAME, VECTOR_LEN)
}

/// One prompt or row's text, tokenized, run through the tower and
/// normalized. `on_truncated` is called when the text ran past
/// `MAX_TOKENS`, so each caller can say what it was that got cut.
fn embed(model: &Model, text: &str, on_truncated: impl FnOnce()) -> Result<Vec<f32>, String> {
    let tokenized = tokenize(text)?;
    if tokenized.truncated {
        on_truncated();
    }
    let raw = run_tower(model, &tokenized.ids, &tokenized.mask)?;
    Ok(normalize(raw))
}

const EMBED_CLIP_TEXT: &str = "embed_clip_text";
const EMBED_CLIP_TEXT_PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"prompt":{"type":"string"}},"required":["prompt"],"additionalProperties":false}"#;
const EMBED_CLIP_TEXT_RESULT_SCHEMA: &str = r#"{"type":"array","items":{"type":"number"},"minItems":512,"maxItems":512,"description":"X-CLIP text tower output, L2-normalized so cos_similarity(a, b) is a plain dot product."}"#;

/// `args`' one string field named `field` - `embed_clip_text`'s `"prompt"`.
fn extract_string_arg(name: &str, args: &str, field: &str) -> Result<String, String> {
    let parsed: serde_json::Value =
        serde_json::from_str(args).map_err(|e| format!("{name}: args is not valid JSON: {e}"))?;
    parsed
        .as_object()
        .and_then(|o| o.get(field))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| format!("{name}: args must be an object with a string \"{field}\""))
}

/// What `process` reads: a cue-shaped row, extra keys ignored - the same
/// shape `sounds`'s labels and a transcript's words both take.
const INPUT_ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"start_t":{"type":"number"},"end_t":{"type":"number"},"text":{"type":"string"}},"required":["start_t","end_t","text"],"additionalProperties":true}"#;
/// What `process` writes: the same span, `text` replaced by its vector.
const OUTPUT_ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"start_t":{"type":"number"},"end_t":{"type":"number"},"vector":{"type":"array","items":{"type":"number"},"minItems":512,"maxItems":512}},"required":["start_t","end_t","vector"],"additionalProperties":false,"description":"vector is X-CLIP text tower output, L2-normalized so cos_similarity(a, b) is a plain dot product."}"#;

/// One input row `process` reads: a cue, extra keys ignored by `serde`'s
/// own default (unknown fields are simply not collected).
#[derive(Deserialize)]
struct InCue {
    start_t: f64,
    end_t: f64,
    text: String,
}

/// One output row `process` writes: the same span, `text` embedded.
#[derive(Serialize)]
struct OutRow {
    start_t: f64,
    end_t: f64,
    vector: Vec<f32>,
}

fn process_row(model: &Model, row: &str) -> Result<OutRow, String> {
    let cue: InCue =
        serde_json::from_str(row).map_err(|e| format!("embed_clip: {row}: not a cue: {e}"))?;
    let start_t = cue.start_t;
    let vector = embed(model, &cue.text, || {
        eprintln!(
            "embed_clip: row at start_t={start_t}: text ran past {MAX_TOKENS} tokens and was truncated"
        );
    })?;
    Ok(OutRow {
        start_t: cue.start_t,
        end_t: cue.end_t,
        vector,
    })
}

thread_local! {
    /// The graph a value call runs, loaded by the first one.
    static VALUES_MODEL: RefCell<Option<Model>> = const { RefCell::new(None) };
}

struct EmbedClipText;

impl ValuesGuest for EmbedClipText {
    fn list_functions() -> Vec<FunctionMeta> {
        vec![FunctionMeta {
            name: EMBED_CLIP_TEXT.to_string(),
            params_schema: EMBED_CLIP_TEXT_PARAMS_SCHEMA.to_string(),
            result_schema: EMBED_CLIP_TEXT_RESULT_SCHEMA.to_string(),
        }]
    }

    fn invoke(name: String, args: String) -> Result<String, String> {
        match name.as_str() {
            EMBED_CLIP_TEXT => {
                let prompt = extract_string_arg(EMBED_CLIP_TEXT, &args, "prompt")?;
                let vector = VALUES_MODEL.with(|cell| {
                    let mut model = cell.borrow_mut();
                    if model.is_none() {
                        *model = Some(Model::load("embed_clip_text", MODEL)?);
                    }
                    let model = model.as_ref().expect("just loaded");
                    embed(model, &prompt, || {
                        eprintln!(
                            "embed_clip_text: prompt ran past {MAX_TOKENS} tokens and was truncated: {prompt:?}"
                        );
                    })
                })?;
                serde_json::to_string(&vector)
                    .map_err(|e| format!("{EMBED_CLIP_TEXT}: serializing result: {e}"))
            }
            other => Err(format!(
                "embed_clip does not export {other}; it exports {EMBED_CLIP_TEXT}"
            )),
        }
    }
}

export!(EmbedClipText);

struct EmbedClip {
    rows: u32,
    model: Model,
}

impl Node for EmbedClip {
    const NAME: &'static str = "embed_clip";
    const VERSION: &'static str = "0.2.0";
    type Params = NoParams;

    fn shape(_: &NoParams, _: &Bound) -> NodeResult<Shape> {
        Ok(Shape::new()
            .input(Input::rows("rows").clock().schema_json(INPUT_ROWS_SCHEMA))
            .output(Output::rows("vectors").schema_json(OUTPUT_ROWS_SCHEMA))
            .pure())
    }

    fn init(_: NoParams, init: &Init) -> NodeResult<EmbedClip> {
        Ok(EmbedClip {
            rows: init.stream("rows")?.id,
            // Loaded here, so a run missing `-nn embed_clip=<path>` is refused
            // before any row is read.
            model: Model::load("embed_clip", MODEL)?,
        })
    }

    fn process(&mut self, tick: &Tick, out: &mut Out) -> NodeResult<()> {
        for message in tick.messages(self.rows) {
            let row = String::from_utf8_lossy(&message.data);
            out.row("vectors", message.pts, &process_row(&self.model, &row)?)?;
        }
        Ok(())
    }
}

ffrwd_node::export!(EmbedClip);

#[cfg(test)]
mod tests {
    use super::*;

    // Computed once from the reference tokenizer (transformers'
    // CLIPTokenizer, same revision as TOKENIZER_JSON) and pinned here as
    // plain expected values - see the crate's own report for how.
    const A_PHOTO_OF_A_CAT: [i64; 7] = [49406, 320, 1125, 539, 320, 2368, 49407];
    const A_DOG_BARKING_OUTSIDE: [i64; 6] = [49406, 320, 1929, 32676, 2782, 49407];
    const A_PERSON_RIDING: [i64; 12] = [
        49406, 320, 2533, 6765, 320, 11652, 1136, 518, 2012, 536, 3424, 49407,
    ];

    #[test]
    fn known_prompts_tokenize_to_the_reference_ids() {
        for (text, expected) in [
            ("a photo of a cat", &A_PHOTO_OF_A_CAT[..]),
            ("a dog barking outside", &A_DOG_BARKING_OUTSIDE[..]),
            (
                "a person riding a bicycle down the street at sunset",
                &A_PERSON_RIDING[..],
            ),
        ] {
            let tokenized = tokenize(text).expect("tokenizes");
            assert_eq!(&tokenized.ids[..expected.len()], expected, "text: {text}");
            assert!(!tokenized.truncated, "text: {text}");
            assert_eq!(tokenized.ids.len(), MAX_TOKENS);
            assert_eq!(tokenized.mask.len(), MAX_TOKENS);
            // The mask is 1 for bos..eos and 0 for every pad slot after.
            let ones = expected.len();
            assert!(
                tokenized.mask[..ones].iter().all(|&m| m == 1),
                "text: {text}"
            );
            assert!(
                tokenized.mask[ones..].iter().all(|&m| m == 0),
                "text: {text}"
            );
            // Padding fills with eos, matching transformers' own convention.
            assert!(
                tokenized.ids[ones..].iter().all(|&id| id == EOS_ID as i64),
                "text: {text}"
            );
        }
    }

    #[test]
    fn a_prompt_past_max_tokens_is_reported_truncated() {
        let long = "word ".repeat(100);
        let tokenized = tokenize(long.trim()).expect("tokenizes");
        assert!(tokenized.truncated);
        assert_eq!(tokenized.ids.len(), MAX_TOKENS);
        // Truncation still ends on eos: the content is cut, not the tail.
        assert_eq!(*tokenized.ids.last().unwrap(), EOS_ID as i64);
        assert!(tokenized.mask.iter().all(|&m| m == 1));
    }

    #[test]
    fn a_short_prompt_is_not_reported_truncated() {
        assert!(!tokenize("hello").expect("tokenizes").truncated);
    }

    #[test]
    fn rows_clock_the_node_and_vectors_leave_with_them() {
        use ffrwd_node::{Clock, Kind, Runner};
        let shape = Runner::<EmbedClip>::shape("", &["rows".to_owned()]).expect("a shape");
        assert_eq!(shape.clock, Some(Clock::Input("rows".to_owned())));
        assert_eq!(shape.inputs[0].kind, Kind::Data);
        let vectors = &shape.outputs[0];
        assert_eq!((vectors.name.as_str(), vectors.latency), ("vectors", 0.0));
        assert!(shape.pure);
    }

    #[test]
    fn no_params_is_accepted_and_anything_else_is_refused() {
        use ffrwd_node::Runner;
        let bound = ["rows".to_owned()];
        assert!(Runner::<EmbedClip>::shape("{}", &bound).is_ok());
        let error = Runner::<EmbedClip>::shape(r#"{"lang":"es"}"#, &bound).expect_err("no params");
        assert!(error.contains("lang"), "got: {error}");
    }

    #[test]
    fn take_output_names_what_it_got_when_the_wanted_tensor_is_missing() {
        let error = take_output(
            vec![("other".to_string(), zero_bytes())],
            "text_embeds",
            512,
        )
        .expect_err("text_embeds is not among the outputs");
        assert!(error.contains("text_embeds"), "{error}");
        assert!(error.contains("other"), "{error}");
    }

    #[test]
    fn take_output_refuses_a_tensor_of_the_wrong_length() {
        let error = take_output(
            vec![("text_embeds".to_string(), zero_bytes())],
            "text_embeds",
            512,
        )
        .expect_err("only 4 numbers, not 512");
        assert!(error.contains("text_embeds"), "{error}");
    }

    /// 4 zero f32s' little-endian bytes - too short to be a real 512-d
    /// output, which is the point in the two tests above. Plain bytes, the
    /// way `run_tower` hands `take_output` a compute's answer, rather than
    /// the wit-bindgen `Tensor` resource: `Tensor` is a host resource and
    /// unreachable outside a real wasm import, so nothing in this test
    /// module constructs one.
    fn zero_bytes() -> Vec<u8> {
        vec![0u8; 16]
    }

    #[test]
    fn extract_string_arg_reads_the_named_field() {
        assert_eq!(
            extract_string_arg("embed_clip_text", r#"{"prompt":"a cat"}"#, "prompt")
                .expect("present"),
            "a cat"
        );
    }

    #[test]
    fn extract_string_arg_refuses_a_missing_field() {
        let error = extract_string_arg("embed_clip_text", r#"{}"#, "prompt")
            .expect_err("prompt is missing");
        assert!(error.contains("prompt"), "{error}");
    }
}
