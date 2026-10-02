//! What the four describe nodes share: the graph each runs through
//! `wasi:nn`, and the little-endian floats a tensor travels in. The module
//! never opens a file: the host binds the graph to a name with
//! `-nn <name>=<path>`, and the node asks for that name and nothing else.

// `generate_all`: the world's interfaces are wasi:nn's, a package of its own,
// and without it bindgen expects them to have been generated somewhere else.
wit_bindgen::generate!({
    path: "wit-world",
    world: "ffrwd:describe/nn",
    generate_all,
});

use wasi::nn::errors::{Error, ErrorCode};
use wasi::nn::graph::{load_by_name, Graph};
use wasi::nn::inference::GraphExecutionContext;
use wasi::nn::tensor::{Tensor, TensorType};

/// What one input tensor's elements are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Element {
    F32,
    I64,
}

/// One input tensor: the graph's name for it, its shape, and its bytes.
pub struct Feed<'a> {
    pub name: &'a str,
    pub dimensions: &'a [u32],
    pub element: Element,
    pub data: &'a [u8],
}

/// One loaded graph, for the life of the instance.
pub struct Model {
    /// The module, which every message names.
    module: &'static str,
    /// Held for the life of the instance: building it once is what keeps a
    /// provider's kernels from being chosen again per call.
    context: GraphExecutionContext,
    /// Kept alive because the context is only valid while its graph is.
    _graph: Graph,
}

impl Model {
    /// The graph the host bound to `name`, with its execution context built
    /// once: the first call is what a provider picks its kernels on, and
    /// every call after it reuses them.
    pub fn load(module: &'static str, name: &str) -> Result<Model, String> {
        let graph = load_by_name(name)
            .map_err(|e| failed(module, &format!("load-by-name({name:?})"), &e))?;
        let context = graph
            .init_execution_context()
            .map_err(|e| failed(module, "init-execution-context", &e))?;
        Ok(Model {
            module,
            context,
            _graph: graph,
        })
    }

    /// The feeds through the graph, and every tensor it returned as its name
    /// and its bytes, in the graph's own order.
    pub fn run(&self, feeds: &[Feed]) -> Result<Vec<(String, Vec<u8>)>, String> {
        let inputs = feeds
            .iter()
            .map(|feed| {
                let element = match feed.element {
                    Element::F32 => TensorType::Fp32,
                    Element::I64 => TensorType::I64,
                };
                (
                    feed.name.to_string(),
                    Tensor::new(feed.dimensions, element, feed.data),
                )
            })
            .collect();
        let returned = self
            .context
            .compute(inputs)
            .map_err(|e| failed(self.module, "compute", &e))?;
        Ok(returned
            .into_iter()
            .map(|(name, tensor)| (name, tensor.data()))
            .collect())
    }
}

/// The spec's spelling of an error code, so a message says what actually
/// went wrong rather than how this module happens to format things.
fn failed(module: &str, what: &str, error: &Error) -> String {
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
    format!("{module}: {what}: {code} ({})", error.data())
}

/// A tensor's floats, out of the little-endian bytes it arrived as.
pub fn le_f32s(data: &[u8]) -> Vec<f32> {
    let (whole, _) = data.as_chunks::<4>();
    whole.iter().copied().map(f32::from_le_bytes).collect()
}

/// The floats as the little-endian bytes a tensor travels in.
pub fn le_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = vec![0u8; values.len() * 4];
    let (words, _) = bytes.as_chunks_mut::<4>();
    for (word, value) in words.iter_mut().zip(values) {
        *word = value.to_le_bytes();
    }
    bytes
}

/// The vector scaled to unit length, which is what makes a dot product a
/// cosine. A vector of all zeros has no direction and is left alone.
pub fn normalize(mut vector: Vec<f32>) -> Vec<f32> {
    let length = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
    if length > 0.0 {
        for value in &mut vector {
            *value /= length;
        }
    }
    vector
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_survive_the_round_trip_through_a_tensors_bytes() {
        let values = vec![0.0f32, -1.5, 1e-8, 3.25, f32::MIN_POSITIVE];
        assert_eq!(le_f32s(&le_bytes(&values)), values);
    }

    #[test]
    fn a_vector_comes_out_at_unit_length() {
        let vector = normalize(vec![3.0, 4.0]);
        assert!((vector[0] - 0.6).abs() < 1e-6);
        assert!((vector[1] - 0.8).abs() < 1e-6);
        let length: f32 = normalize(vec![1.0; 512]).iter().map(|v| v * v).sum();
        assert!((length - 1.0).abs() < 1e-5);
    }

    #[test]
    fn a_vector_with_no_direction_is_left_alone() {
        assert_eq!(normalize(vec![0.0; 4]), vec![0.0; 4]);
    }

    #[test]
    fn normalizing_does_not_change_direction() {
        let a = normalize(vec![1.0, 2.0, 2.0]);
        let b = normalize(vec![2.0, 4.0, 4.0]);
        let dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        assert!((dot - 1.0).abs() < 1e-6);
    }
}
