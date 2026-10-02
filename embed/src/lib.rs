//! A sentence embedder, reached two ways.
//!
//! `embed_text` is a value function: one prompt in, one 384-d vector out,
//! computed at compile time like any other value call. `embed` is a node
//! clocked by its rows: it reads whatever produced its `cue[]` argument
//! (sounds's labels, or a transcript's words, either shape) and writes one
//! vector a row on `vectors`, at the row's own time, no stream involved. Both tokenize their text the same
//! way and run the same graph; the manifest pins the one ONNX file both
//! load, under the export name `embed`.
//!
//! The model is all-MiniLM-L6-v2: a sentence embedder, trained to match one
//! sentence against another. A CLIP text tower - what `embed_clip`/
//! `embed_clip_text` put under the clip space instead - is trained to match
//! images, not sentences, and ranks a short label well but not a spoken
//! sentence; this module is what a cue's `text` (an AudioSet label, a
//! transcript word) gets embedded with, so it ranks correctly against a
//! search prompt in the same sentence space.
//!
//! Every vector this module returns is L2-normalized, so `cos_similarity`
//! between any two of them is a plain dot product.
//!
//! The tokenizer is bert-base-uncased's WordPiece (vocab 30522, `[CLS]`/
//! `[SEP]`/`[PAD]`). Rather than hand-writing that, this module embeds the
//! `tokenizers` crate's own `tokenizer.json` - fetched from the source
//! model's revision, `assets/fetch-tokenizer.py` - compiled in with
//! `include_bytes!`. A wasm module has no filesystem: `wasi-nn`'s
//! `load-by-name` reaches the ONNX graph by name alone, and there is no
//! parallel mechanism for a non-graph file, so the tokenizer's vocabulary
//! travels inside the module rather than beside it.
//!
//! The ONNX graph's `input_ids`/`attention_mask`/`token_type_ids` all take a
//! dynamic sequence length, so - unlike a CLIP tower's fixed context - a
//! prompt is truncated to `MAX_TOKENS` but never padded: one call is one
//! sequence, exactly as long as the text tokenized to. `last_hidden_state`
//! comes back one 384-d row per token; `mean_pool` collapses that to one
//! vector, weighted by the attention mask the same way sentence-transformers
//! itself pools this model's output.

// The values export rides on bindings of its own beside the node ffrwd-node
// exports; `../wit` is ffrwd:av@0.19.0 as the SDK carries it, byte for byte.
wit_bindgen::generate!({
    path: ["../wit", "wit-world"],
    world: "ffrwd:embed/embed",
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
use tokenizers::{Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy};

/// The name the host binds the graph to: `-nn embed=<path>`.
const MODEL: &str = "embed";

/// What the export calls its three inputs, each `[1, n]` i64.
const INPUT_IDS: &str = "input_ids";
const ATTENTION_MASK: &str = "attention_mask";
const TOKEN_TYPE_IDS: &str = "token_type_ids";
/// What the export calls its output, `[1, n, VECTOR_LEN]` f32.
const OUTPUT_NAME: &str = "last_hidden_state";

/// A prompt longer than this is truncated - the graph takes any sequence
/// length, so this is a cap chosen for the model's own training length
/// rather than a shape the graph fixes.
const MAX_TOKENS: usize = 256;
/// The model's output width.
const VECTOR_LEN: usize = 384;

/// bert-base-uncased's WordPiece, exactly as `Xenova/all-MiniLM-L6-v2`
/// published it at revision 751bff37182d3f1213fa05d7196b954e230abad9 -
/// `assets/fetch-tokenizer.py` re-fetches it, hash-checked. Compiled in
/// because the module has no other way to reach it; see the module doc
/// comment.
const TOKENIZER_JSON: &[u8] = include_bytes!("../assets/tokenizer.json");

/// The tokenizer, built twice from the same bytes: `raw` reports a prompt's
/// true token count, untruncated, which is what tells whether a prompt ran
/// past `MAX_TOKENS`; `truncated` is capped to `MAX_TOKENS` and nothing
/// more - no padding, since one call is one sequence exactly as long as it
/// tokenized to. Built once per instance and kept for every call after.
struct Tok {
    raw: Tokenizer,
    truncated: Tokenizer,
}

fn tok() -> &'static Tok {
    static TOK: OnceLock<Tok> = OnceLock::new();
    TOK.get_or_init(|| {
        // The published tokenizer.json bakes in its own truncation and
        // padding (128, fixed) - both are overridden explicitly here rather
        // than trusted: this module never pads (see the module doc comment),
        // and MAX_TOKENS is this module's own cap, not the file's.
        let mut raw =
            Tokenizer::from_bytes(TOKENIZER_JSON).expect("embedded tokenizer.json parses");
        raw.with_truncation(None)
            .expect("clearing truncation always succeeds");
        raw.with_padding(None);
        let mut truncated =
            Tokenizer::from_bytes(TOKENIZER_JSON).expect("embedded tokenizer.json parses");
        truncated
            .with_truncation(Some(TruncationParams {
                max_length: MAX_TOKENS,
                strategy: TruncationStrategy::LongestFirst,
                direction: TruncationDirection::Right,
                stride: 0,
            }))
            .expect("truncation params are well-formed");
        truncated.with_padding(None);
        Tok { raw, truncated }
    })
}

/// One prompt, tokenized to at most `MAX_TOKENS` - never padded, so `ids`,
/// `mask` and `type_ids` are all exactly the same length, the true token
/// count up to the cap. `truncated` is true when the prompt's true token
/// count - `[CLS]`/`[SEP]` included - ran past `MAX_TOKENS` and some of its
/// text was therefore dropped.
struct Tokenized {
    ids: Vec<i64>,
    mask: Vec<i64>,
    type_ids: Vec<i64>,
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
        .truncated
        .encode(text, true)
        .map_err(|e| format!("tokenize: {e}"))?;
    let ids = encoding.get_ids().iter().map(|&id| id as i64).collect();
    let mask = encoding
        .get_attention_mask()
        .iter()
        .map(|&m| m as i64)
        .collect();
    let type_ids = encoding.get_type_ids().iter().map(|&t| t as i64).collect();
    Ok(Tokenized {
        ids,
        mask,
        type_ids,
        truncated: true_len > MAX_TOKENS,
    })
}

/// The named tensor's bytes out of what a compute answered, checked against
/// the element count its shape fixes. Takes plain bytes rather than the
/// wit-bindgen `Tensor` resource - `run_model` converts immediately after
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
            "embed: the graph answered without {name:?} (unclaimed: {unclaimed:?}); \
             this module wants all-MiniLM-L6-v2's last_hidden_state"
        ));
    };
    let (_, bytes) = outputs.swap_remove(pos);
    let values = le_f32s(&bytes);
    if values.len() != len {
        return Err(format!(
            "embed: {name} came back as {} numbers, expected {len}",
            values.len()
        ));
    }
    Ok(values)
}

/// One tokenized prompt through the graph: `last_hidden_state`, one
/// `VECTOR_LEN`-wide row per token, flattened token-major.
fn run_model(
    model: &Model,
    ids: &[i64],
    mask: &[i64],
    type_ids: &[i64],
) -> Result<Vec<f32>, String> {
    let dims = [1, ids.len() as u32];
    let ids_bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mask_bytes: Vec<u8> = mask.iter().flat_map(|v| v.to_le_bytes()).collect();
    let type_bytes: Vec<u8> = type_ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let feed = |name, data| Feed {
        name,
        dimensions: &dims,
        element: Element::I64,
        data,
    };
    let outputs = model.run(&[
        feed(INPUT_IDS, &ids_bytes),
        feed(ATTENTION_MASK, &mask_bytes),
        feed(TOKEN_TYPE_IDS, &type_bytes),
    ])?;
    take_output(outputs, OUTPUT_NAME, ids.len() * VECTOR_LEN)
}

/// `last_hidden_state`'s `seq_len` rows of `VECTOR_LEN` floats each,
/// averaged into one row, each token weighted by `mask` - the same masked
/// mean sentence-transformers itself pools this model's output with.
/// `mask` is 0 only where a token was padding, so with no padding at all
/// (this module never pads) it reduces to a plain mean over every token.
fn mean_pool(hidden: &[f32], mask: &[i64], seq_len: usize) -> Vec<f32> {
    let mut pooled = vec![0.0f32; VECTOR_LEN];
    let mut weight = 0.0f32;
    for (row, &m) in (0..seq_len).zip(mask) {
        let m = m as f32;
        if m == 0.0 {
            continue;
        }
        weight += m;
        let start = row * VECTOR_LEN;
        for (p, &v) in pooled.iter_mut().zip(&hidden[start..start + VECTOR_LEN]) {
            *p += v * m;
        }
    }
    if weight > 0.0 {
        for p in &mut pooled {
            *p /= weight;
        }
    }
    pooled
}

/// One prompt or row's text, tokenized, run through the model, pooled and
/// normalized. `on_truncated` is called when the text ran past
/// `MAX_TOKENS`, so each caller can say what it was that got cut.
fn embed(model: &Model, text: &str, on_truncated: impl FnOnce()) -> Result<Vec<f32>, String> {
    let tokenized = tokenize(text)?;
    if tokenized.truncated {
        on_truncated();
    }
    let seq_len = tokenized.ids.len();
    let hidden = run_model(model, &tokenized.ids, &tokenized.mask, &tokenized.type_ids)?;
    let pooled = mean_pool(&hidden, &tokenized.mask, seq_len);
    Ok(normalize(pooled))
}

const EMBED_TEXT: &str = "embed_text";
const EMBED_TEXT_PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"prompt":{"type":"string"}},"required":["prompt"],"additionalProperties":false}"#;
const EMBED_TEXT_RESULT_SCHEMA: &str = r#"{"type":"array","items":{"type":"number"},"minItems":384,"maxItems":384,"description":"all-MiniLM-L6-v2 output, mean-pooled and L2-normalized so cos_similarity(a, b) is a plain dot product."}"#;

/// `args`' one string field named `field` - `embed_text`'s `"prompt"`.
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
const OUTPUT_ROWS_SCHEMA: &str = r#"{"type":"object","properties":{"start_t":{"type":"number"},"end_t":{"type":"number"},"vector":{"type":"array","items":{"type":"number"},"minItems":384,"maxItems":384}},"required":["start_t","end_t","vector"],"additionalProperties":false,"description":"vector is all-MiniLM-L6-v2 output, mean-pooled and L2-normalized so cos_similarity(a, b) is a plain dot product."}"#;

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
        serde_json::from_str(row).map_err(|e| format!("embed: {row}: not a cue: {e}"))?;
    let start_t = cue.start_t;
    let vector = embed(model, &cue.text, || {
        eprintln!(
            "embed: row at start_t={start_t}: text ran past {MAX_TOKENS} tokens and was truncated"
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

struct EmbedText;

impl ValuesGuest for EmbedText {
    fn list_functions() -> Vec<FunctionMeta> {
        vec![FunctionMeta {
            name: EMBED_TEXT.to_string(),
            params_schema: EMBED_TEXT_PARAMS_SCHEMA.to_string(),
            result_schema: EMBED_TEXT_RESULT_SCHEMA.to_string(),
        }]
    }

    fn invoke(name: String, args: String) -> Result<String, String> {
        match name.as_str() {
            EMBED_TEXT => {
                let prompt = extract_string_arg(EMBED_TEXT, &args, "prompt")?;
                let vector = VALUES_MODEL.with(|cell| {
                    let mut model = cell.borrow_mut();
                    if model.is_none() {
                        *model = Some(Model::load("embed_text", MODEL)?);
                    }
                    let model = model.as_ref().expect("just loaded");
                    embed(model, &prompt, || {
                        eprintln!(
                            "embed_text: prompt ran past {MAX_TOKENS} tokens and was truncated: {prompt:?}"
                        );
                    })
                })?;
                serde_json::to_string(&vector)
                    .map_err(|e| format!("{EMBED_TEXT}: serializing result: {e}"))
            }
            other => Err(format!(
                "embed does not export {other}; it exports {EMBED_TEXT}"
            )),
        }
    }
}

export!(EmbedText);

struct Embed {
    rows: u32,
    model: Model,
}

impl Node for Embed {
    const NAME: &'static str = "embed";
    const VERSION: &'static str = "0.2.0";
    type Params = NoParams;

    fn shape(_: &NoParams, _: &Bound) -> NodeResult<Shape> {
        Ok(Shape::new()
            .input(Input::rows("rows").clock().schema_json(INPUT_ROWS_SCHEMA))
            .output(Output::rows("vectors").schema_json(OUTPUT_ROWS_SCHEMA))
            .pure())
    }

    fn init(_: NoParams, init: &Init) -> NodeResult<Embed> {
        Ok(Embed {
            rows: init.stream("rows")?.id,
            // Loaded here, so a run missing `-nn embed=<path>` is refused
            // before any row is read.
            model: Model::load("embed", MODEL)?,
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

ffrwd_node::export!(Embed);

#[cfg(test)]
mod tests {
    use super::*;

    // Computed once from the reference tokenizer (transformers'
    // AutoTokenizer over the same tokenizer.json) and pinned here as plain
    // expected values - see the crate's own report for how. `[CLS]` is 101,
    // `[SEP]` is 102.
    const A_DOG_BARKING_OUTSIDE: [i64; 6] = [101, 1037, 3899, 19372, 2648, 102];
    const CARGO_SHIP: [i64; 11] = [
        101, 2053, 2166, 2001, 2179, 7548, 1996, 28839, 6636, 2911, 102,
    ];

    #[test]
    fn known_prompts_tokenize_to_the_reference_ids() {
        for (text, expected) in [
            ("a dog barking outside", &A_DOG_BARKING_OUTSIDE[..]),
            (
                "no life was found aboard the derelict cargo ship",
                &CARGO_SHIP[..],
            ),
        ] {
            let tokenized = tokenize(text).expect("tokenizes");
            assert_eq!(tokenized.ids, expected, "text: {text}");
            assert!(!tokenized.truncated, "text: {text}");
            // No padding: mask and type_ids are all-ones/all-zeros at the
            // true token count, not some fixed shape.
            assert_eq!(tokenized.mask.len(), expected.len());
            assert!(tokenized.mask.iter().all(|&m| m == 1), "text: {text}");
            assert!(tokenized.type_ids.iter().all(|&t| t == 0), "text: {text}");
        }
    }

    #[test]
    fn a_prompt_past_max_tokens_is_reported_truncated_and_capped() {
        let long = "word ".repeat(400);
        let tokenized = tokenize(long.trim()).expect("tokenizes");
        assert!(tokenized.truncated);
        assert_eq!(tokenized.ids.len(), MAX_TOKENS);
        assert_eq!(tokenized.mask.len(), MAX_TOKENS);
    }

    #[test]
    fn a_short_prompt_is_not_reported_truncated() {
        let tokenized = tokenize("hello").expect("tokenizes");
        assert!(!tokenized.truncated);
        assert!(tokenized.ids.len() < MAX_TOKENS);
    }

    #[test]
    fn mean_pool_averages_unmasked_rows() {
        // Two tokens, VECTOR_LEN 3 for the test's own sake would need
        // VECTOR_LEN itself changed, so this drives mean_pool through its
        // real width with a hidden state that is the same value repeated
        // per row, checked against the row count instead.
        let seq_len = 3;
        let mut hidden = vec![0.0f32; seq_len * VECTOR_LEN];
        for row in 0..seq_len {
            for col in 0..VECTOR_LEN {
                hidden[row * VECTOR_LEN + col] = (row + 1) as f32;
            }
        }
        // Row 2 (value 3.0) is masked out.
        let pooled = mean_pool(&hidden, &[1, 1, 0], seq_len);
        // Mean of rows 0 and 1: (1.0 + 2.0) / 2 = 1.5, every column.
        assert!(pooled.iter().all(|&v| (v - 1.5).abs() < 1e-6));
    }

    #[test]
    fn mean_pool_of_an_all_masked_row_set_is_the_zero_vector() {
        let hidden = vec![5.0f32; VECTOR_LEN];
        let pooled = mean_pool(&hidden, &[0], 1);
        assert!(pooled.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn rows_clock_the_node_and_vectors_leave_with_them() {
        use ffrwd_node::{Clock, Kind, Runner};
        let shape = Runner::<Embed>::shape("", &["rows".to_owned()]).expect("a shape");
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
        assert!(Runner::<Embed>::shape("{}", &bound).is_ok());
        let error = Runner::<Embed>::shape(r#"{"lang":"es"}"#, &bound).expect_err("no params");
        assert!(error.contains("lang"), "got: {error}");
    }

    #[test]
    fn take_output_names_what_it_got_when_the_wanted_tensor_is_missing() {
        let error = take_output(
            vec![("other".to_string(), zero_bytes())],
            "last_hidden_state",
            VECTOR_LEN,
        )
        .expect_err("last_hidden_state is not among the outputs");
        assert!(error.contains("last_hidden_state"), "{error}");
        assert!(error.contains("other"), "{error}");
    }

    #[test]
    fn take_output_refuses_a_tensor_of_the_wrong_length() {
        let error = take_output(
            vec![("last_hidden_state".to_string(), zero_bytes())],
            "last_hidden_state",
            VECTOR_LEN,
        )
        .expect_err("only 4 numbers, not VECTOR_LEN");
        assert!(error.contains("last_hidden_state"), "{error}");
    }

    /// 4 zero f32s' little-endian bytes - too short to be a real output,
    /// which is the point in the two tests above. Plain bytes, the way
    /// `run_model` hands `take_output` a compute's answer, rather than the
    /// wit-bindgen `Tensor` resource: `Tensor` is a host resource and
    /// unreachable outside a real wasm import, so nothing in this test
    /// module constructs one.
    fn zero_bytes() -> Vec<u8> {
        vec![0u8; 16]
    }

    #[test]
    fn extract_string_arg_reads_the_named_field() {
        assert_eq!(
            extract_string_arg("embed_text", r#"{"prompt":"a cat"}"#, "prompt").expect("present"),
            "a cat"
        );
    }

    #[test]
    fn extract_string_arg_refuses_a_missing_field() {
        let error =
            extract_string_arg("embed_text", r#"{}"#, "prompt").expect_err("prompt is missing");
        assert!(error.contains("prompt"), "{error}");
    }
}
