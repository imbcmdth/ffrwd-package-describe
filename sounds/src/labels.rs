//! AudioSet's 527 class names, in the id2label order the model's own
//! `config.json` gives them (index 0 is "Speech", 501 is "Sine wave").
//!
//! The module reaches no filesystem at runtime - the `window-module` world
//! imports nothing that would let it read a file the package installed
//! beside the graph - so the names travel as source rather than as a
//! second model file `-nn` never binds. `audioset-labels.txt` is pulled
//! from the model repo's `config.json` (`onnx-community/ast-finetuned-
//! audioset-10-10-0.4593-ONNX`, the revision `ffrwd.json` pins), one label
//! per line in class-index order, and checked into the crate.

use std::sync::OnceLock;

const LABELS_TXT: &str = include_str!("../audioset-labels.txt");

/// How many classes the model's final layer has, and so how long a call's
/// `logits` tensor must be.
pub const COUNT: usize = 527;

fn labels() -> &'static [&'static str] {
    static LABELS: OnceLock<Vec<&'static str>> = OnceLock::new();
    LABELS.get_or_init(|| LABELS_TXT.lines().collect())
}

/// The class name at `index`, or `None` past the end.
pub fn label(index: usize) -> Option<&'static str> {
    labels().get(index).copied()
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// `logits` turned into probabilities, kept at or above `threshold`, and cut
/// to the `top` highest - the label index paired with its own score,
/// highest first. `logits` shorter or longer than [`COUNT`] is a caller
/// error and panics: `sounds` always calls the graph the same way.
pub fn top_labels(logits: &[f32], threshold: f64, top: usize) -> Vec<(usize, f64)> {
    assert_eq!(logits.len(), COUNT, "the graph's own class count");
    let mut scored: Vec<(usize, f64)> = logits
        .iter()
        .enumerate()
        .map(|(index, &logit)| (index, sigmoid(f64::from(logit))))
        .filter(|&(_, score)| score >= threshold)
        .collect();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored.truncate(top);
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn there_are_exactly_527_labels_in_config_json_order() {
        assert_eq!(labels().len(), COUNT);
        assert_eq!(label(0), Some("Speech"));
        assert_eq!(label(1), Some("Male speech, man speaking"));
        assert_eq!(label(501), Some("Sine wave"));
        assert_eq!(label(526), Some("Field recording"));
        assert_eq!(label(527), None);
    }

    #[test]
    fn no_label_repeats() {
        let all = labels();
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert_ne!(a, b, "{a} appears twice");
            }
        }
    }

    /// A logit of 0 is a probability of exactly 0.5, so it is the cleanest
    /// value to pin a threshold test on.
    #[test]
    fn a_class_at_the_threshold_is_kept_and_just_under_it_is_not() {
        let mut logits = [-100.0f32; COUNT];
        logits[501] = 0.0;
        let picked = top_labels(&logits, 0.5, 10);
        assert_eq!(picked, vec![(501, 0.5)], "0.5 is not below 0.5");

        logits[501] = -0.001;
        let picked = top_labels(&logits, 0.5, 10);
        assert!(picked.is_empty(), "just under 0.5 is excluded");
    }

    #[test]
    fn only_the_top_n_scores_survive_highest_first() {
        let mut logits = [-100.0f32; COUNT];
        logits[0] = 1.0; // sigmoid(1) ~= 0.731
        logits[1] = 5.0; // sigmoid(5) ~= 0.993
        logits[2] = 3.0; // sigmoid(3) ~= 0.953
        let picked = top_labels(&logits, 0.0, 2);
        assert_eq!(picked.len(), 2);
        assert_eq!(picked[0].0, 1, "highest score first");
        assert_eq!(picked[1].0, 2);
    }

    #[test]
    fn a_top_of_zero_keeps_nothing() {
        let logits = [100.0f32; COUNT];
        assert!(top_labels(&logits, 0.0, 0).is_empty());
    }

    #[test]
    #[should_panic(expected = "class count")]
    fn a_logits_vector_of_the_wrong_length_panics() {
        top_labels(&[0.0; 10], 0.5, 3);
    }
}
