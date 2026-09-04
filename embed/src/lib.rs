//! X-CLIP's text tower, reached two ways.
//!
//! `embed_text` is a value function: one prompt in, one 512-d vector out,
//! computed at compile time like any other value call. `embed` is a rows
//! module: it reads whatever produced its `cue[]` argument - sounds's
//! labels, or a transcript's words, either shape - and writes one vector
//! beside each row, no stream involved. Both tokenize their text the same
//! way and run the same graph; the manifest pins the one ONNX file both
//! load, under the export name `embed`.
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

// `generate_all`: the world's interfaces come from two other packages -
// ffrwd:av and wasi:nn - and without it bindgen expects them to have been
// generated somewhere else.
wit_bindgen::generate!({
    path: ["wit", "wit-world"],
    // Fully qualified: three packages are in scope, and each has worlds.
    world: "ffrwd:embed/embed",
    generate_all,
});

use std::cell::RefCell;
use std::sync::OnceLock;

use exports::ffrwd::av::rows_module::{Guest as RowsGuest, RowsModuleMeta};
use exports::ffrwd::av::values::{FunctionMeta, Guest as ValuesGuest};
use ffrwd::av::types::Meta;
use serde::{Deserialize, Serialize};
use tokenizers::{
    PaddingParams, PaddingStrategy, Tokenizer, TruncationDirection, TruncationParams,
    TruncationStrategy,
};
use wasi::nn::graph::{load_by_name, Graph};
use wasi::nn::inference::GraphExecutionContext;
use wasi::nn::tensor::{Tensor, TensorType};

/// The name the host binds the graph to: `-nn embed=<path>`.
const MODEL: &str = "embed";

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
    format!("embed: {what}: {code} ({})", error.data())
}

/// The graph and its execution context, built once per instance and kept
/// for every call after - the first call is what a provider picks its
/// kernels on, and every call after it reuses them. The graph is kept
/// alive alongside its context, which is only valid while the graph is.
struct Nn {
    context: GraphExecutionContext,
    _graph: Graph,
}

thread_local! {
    static NN: RefCell<Option<Nn>> = const { RefCell::new(None) };
}

/// Loads the graph if this instance has not already, then runs `f` against
/// its context.
fn with_context<T>(
    f: impl FnOnce(&GraphExecutionContext) -> Result<T, String>,
) -> Result<T, String> {
    NN.with(|cell| {
        let mut opened = cell.borrow_mut();
        if opened.is_none() {
            let graph =
                load_by_name(MODEL).map_err(|e| failed(&format!("load-by-name({MODEL:?})"), &e))?;
            let context = graph
                .init_execution_context()
                .map_err(|e| failed("init-execution-context", &e))?;
            *opened = Some(Nn {
                context,
                _graph: graph,
            });
        }
        f(&opened.as_ref().expect("just filled").context)
    })
}

/// A tensor's floats, out of the little-endian bytes it arrived as.
fn le_f32s(data: &[u8]) -> Vec<f32> {
    let (whole, _) = data.as_chunks::<4>();
    whole.iter().copied().map(f32::from_le_bytes).collect()
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
            "embed: the graph answered without {name:?} (unclaimed: {unclaimed:?}); \
             this module wants X-CLIP's text tower export"
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

/// One tokenized prompt through the graph: the raw, un-normalized 512-d
/// output.
fn run_tower(ids: &[i64], mask: &[i64]) -> Result<Vec<f32>, String> {
    let dims = [1, MAX_TOKENS as u32];
    let ids_bytes: Vec<u8> = ids.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mask_bytes: Vec<u8> = mask.iter().flat_map(|v| v.to_le_bytes()).collect();

    with_context(|context| {
        let inputs = vec![
            (
                INPUT_IDS.to_string(),
                Tensor::new(&dims, TensorType::I64, &ids_bytes),
            ),
            (
                ATTENTION_MASK.to_string(),
                Tensor::new(&dims, TensorType::I64, &mask_bytes),
            ),
        ];
        let outputs = context.compute(inputs).map_err(|e| failed("compute", &e))?;
        let outputs: Vec<(String, Vec<u8>)> = outputs
            .into_iter()
            .map(|(name, tensor)| (name, tensor.data()))
            .collect();
        take_output(outputs, OUTPUT_NAME, VECTOR_LEN)
    })
}

/// `vector` divided by its own L2 norm, so `cos_similarity` between any two
/// of these is a plain dot product. A zero vector - which a trained tower
/// never actually produces - is returned as is rather than manufacturing a
/// direction from nothing.
fn l2_normalize(mut vector: Vec<f32>) -> Vec<f32> {
    let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut vector {
            *v /= norm;
        }
    }
    vector
}

/// One prompt or row's text, tokenized, run through the tower and
/// normalized. `on_truncated` is called when the text ran past
/// `MAX_TOKENS`, so each caller can say what it was that got cut.
fn embed(text: &str, on_truncated: impl FnOnce()) -> Result<Vec<f32>, String> {
    let tokenized = tokenize(text)?;
    if tokenized.truncated {
        on_truncated();
    }
    let raw = run_tower(&tokenized.ids, &tokenized.mask)?;
    Ok(l2_normalize(raw))
}

const EMBED_TEXT: &str = "embed_text";
const EMBED_TEXT_PARAMS_SCHEMA: &str = r#"{"type":"object","properties":{"prompt":{"type":"string"}},"required":["prompt"],"additionalProperties":false}"#;
const EMBED_TEXT_RESULT_SCHEMA: &str = r#"{"type":"array","items":{"type":"number"},"minItems":512,"maxItems":512,"description":"X-CLIP text tower output, L2-normalized so cos_similarity(a, b) is a plain dot product."}"#;

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

const ROWS_PARAMS_SCHEMA: &str =
    r#"{"type":"object","properties":{},"additionalProperties":false}"#;
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

/// This module takes no parameters: the tower and its tokenizer are fixed.
fn validate_params(params: &str) -> Result<(), String> {
    match params.trim() {
        "" | "{}" => Ok(()),
        other => Err(format!("embed takes no params, got: {other}")),
    }
}

fn process_row(row: &str) -> Result<String, String> {
    let cue: InCue =
        serde_json::from_str(row).map_err(|e| format!("embed: {row}: not a cue: {e}"))?;
    let start_t = cue.start_t;
    let vector = embed(&cue.text, || {
        eprintln!(
            "embed: row at start_t={start_t}: text ran past {MAX_TOKENS} tokens and was truncated"
        );
    })?;
    serde_json::to_string(&OutRow {
        start_t: cue.start_t,
        end_t: cue.end_t,
        vector,
    })
    .map_err(|e| format!("embed: serializing a row: {e}"))
}

struct Embed;

impl ValuesGuest for Embed {
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
                let vector = embed(&prompt, || {
                    eprintln!(
                        "embed_text: prompt ran past {MAX_TOKENS} tokens and was truncated: {prompt:?}"
                    );
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

impl RowsGuest for Embed {
    fn describe() -> RowsModuleMeta {
        RowsModuleMeta {
            meta: Meta {
                name: "embed".to_string(),
                version: "0.1.0".to_string(),
                params_schema: ROWS_PARAMS_SCHEMA.to_string(),
                rows_schema: OUTPUT_ROWS_SCHEMA.to_string(),
                pixel_formats: Vec::new(),
                sample_formats: Vec::new(),
                sample_rates: Vec::new(),
                channel_counts: Vec::new(),
                rows_language: Vec::new(),
            },
            input_rows_schema: INPUT_ROWS_SCHEMA.to_string(),
        }
    }

    fn init(params: String) -> Result<(), String> {
        validate_params(&params)?;
        // Fail fast: a run missing `-nn embed=<path>` is refused here,
        // before any row is read, rather than on the first `process` call.
        with_context(|_| Ok(()))
    }

    fn process(rows: Vec<String>) -> Result<Vec<String>, String> {
        rows.iter().map(|row| process_row(row)).collect()
    }

    fn finish() -> Result<Vec<String>, String> {
        // Every row is embedded and emitted the moment `process` sees it;
        // nothing is held back for a final call to release.
        Ok(Vec::new())
    }
}

export!(Embed);

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

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na = a.iter().map(|v| v * v).sum::<f32>().sqrt();
        let nb = b.iter().map(|v| v * v).sum::<f32>().sqrt();
        dot / (na * nb)
    }

    #[test]
    fn l2_normalize_produces_a_unit_vector() {
        let v = l2_normalize(vec![3.0, 4.0]);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "norm = {norm}");
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn a_zero_vector_stays_zero() {
        assert_eq!(l2_normalize(vec![0.0, 0.0, 0.0]), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn normalizing_does_not_change_direction() {
        let a = l2_normalize(vec![1.0, 2.0, 2.0]);
        let b = l2_normalize(vec![2.0, 4.0, 4.0]);
        assert!(
            (cosine(&a, &b) - 1.0).abs() < 1e-6,
            "same direction, different length, should agree exactly"
        );
    }

    #[test]
    fn no_params_is_accepted_and_anything_else_is_refused() {
        assert!(validate_params("").is_ok());
        assert!(validate_params("{}").is_ok());
        let Err(err) = validate_params(r#"{"lang":"es"}"#) else {
            panic!("embed takes no params and should refuse one");
        };
        assert!(err.contains("no params"), "got: {err}");
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
