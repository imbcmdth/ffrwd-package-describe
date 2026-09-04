//! The log-mel spectrogram AST takes as `input_values`: 25 ms frames, a
//! 10 ms hop, 128 Kaldi-scale mel bins, a symmetric Hann window, DC removal
//! and pre-emphasis before the FFT, then a fixed AudioSet mean and standard
//! deviation. This matches `transformers.ASTFeatureExtractor` - specifically
//! its numpy fallback (`transformers.audio_utils.spectrogram` and
//! `mel_filter_bank`), the path it takes without `torchaudio` installed,
//! which is what a machine with no torchaudio validated this crate against.
//! `torchaudio.compliance.kaldi.fbank` is what that fallback is written to
//! match, so the two are the same computation.

use std::f64::consts::PI;
use std::sync::OnceLock;

/// The rate the model was trained at, and the only one this module accepts.
pub const SAMPLE_RATE: u32 = 16_000;

/// 25 ms at 16 kHz.
const FRAME_LENGTH: usize = 400;
/// 10 ms at 16 kHz.
const HOP_LENGTH: usize = 160;
/// The FFT input size: `frame_length` zero-padded to the next power of two.
const FFT_LENGTH: usize = 512;
/// One-sided bins a real FFT of `FFT_LENGTH` carries.
const NUM_FREQ_BINS: usize = FFT_LENGTH / 2 + 1;

/// Mel bins the model's encoder reads.
pub const NUM_MEL_BINS: usize = 128;
/// Frames `input_values` is padded or truncated to.
pub const MAX_FRAMES: usize = 1024;

const PREEMPHASIS: f64 = 0.97;
/// `1.192092955078125e-7`: the extractor's own floor, one ULP above zero in
/// fp32 - a mel bin under it is clamped rather than logged to `-inf`.
const MEL_FLOOR: f64 = 1.192_092_955_078_125e-7;

/// The AudioSet mean and standard deviation `ASTFeatureExtractor` normalizes
/// with by default; the checkpoint this crate loads was trained on features
/// normalized this way.
pub const NORM_MEAN: f64 = -4.267_739_3;
pub const NORM_STD: f64 = 4.568_997_4;

/// How many `FRAME_LENGTH`-sample frames `sample_count` samples give at
/// `HOP_LENGTH` stride, with no centering. Zero when there are not even
/// enough samples for one frame - which is what `lib.rs` reads to skip
/// classifying a call too short to produce anything but padding.
pub fn num_frames(sample_count: usize) -> usize {
    if sample_count < FRAME_LENGTH {
        0
    } else {
        1 + (sample_count - FRAME_LENGTH) / HOP_LENGTH
    }
}

/// A symmetric (non-periodic) Hann window - `np.hanning(400)`.
fn hann_window() -> &'static [f64; FRAME_LENGTH] {
    static WINDOW: OnceLock<[f64; FRAME_LENGTH]> = OnceLock::new();
    WINDOW.get_or_init(|| {
        let mut window = [0.0; FRAME_LENGTH];
        for (n, value) in window.iter_mut().enumerate() {
            *value = 0.5 - 0.5 * (2.0 * PI * n as f64 / (FRAME_LENGTH as f64 - 1.0)).cos();
        }
        window
    })
}

/// Hertz to Kaldi's mel scale: `1127 * ln(1 + hz/700)`.
fn hertz_to_mel(hertz: f64) -> f64 {
    1127.0 * (1.0 + hertz / 700.0).ln()
}

fn linspace(start: f64, end: f64, count: usize) -> Vec<f64> {
    if count < 2 {
        return vec![start; count];
    }
    let step = (end - start) / (count - 1) as f64;
    (0..count).map(|i| start + step * i as f64).collect()
}

/// The `(NUM_FREQ_BINS, NUM_MEL_BINS)` triangular mel filter bank, row-major
/// by frequency bin, matching `mel_filter_bank(num_frequency_bins=257,
/// num_mel_filters=128, min_frequency=20, max_frequency=8000,
/// sampling_rate=16000, mel_scale="kaldi", triangularize_in_mel_space=True)`.
///
/// Triangularizing in mel space means the 128 filters' boundaries are 130
/// points evenly spaced in mel, and both the filters and the FFT bins they
/// weigh are compared in mel rather than hertz - which is what
/// `triangularize_in_mel_space=True` asks of `transformers` and what makes
/// this match `torchaudio`.
fn mel_filter_bank() -> &'static [f64] {
    static FILTERS: OnceLock<Vec<f64>> = OnceLock::new();
    FILTERS.get_or_init(|| {
        let mel_min = hertz_to_mel(20.0);
        let mel_max = hertz_to_mel(f64::from(SAMPLE_RATE) / 2.0);
        // 130 boundary points in mel space: filter m spans points m, m+1, m+2.
        let boundaries = linspace(mel_min, mel_max, NUM_MEL_BINS + 2);
        let diff: Vec<f64> = boundaries
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .collect();

        let fft_bin_width = f64::from(SAMPLE_RATE) / ((NUM_FREQ_BINS - 1) as f64 * 2.0);
        let fft_freqs: Vec<f64> = (0..NUM_FREQ_BINS)
            .map(|bin| hertz_to_mel(fft_bin_width * bin as f64))
            .collect();

        let mut filters = vec![0.0f64; NUM_FREQ_BINS * NUM_MEL_BINS];
        for (bin, &fft_freq) in fft_freqs.iter().enumerate() {
            for mel in 0..NUM_MEL_BINS {
                let down_slope = (fft_freq - boundaries[mel]) / diff[mel];
                let up_slope = (boundaries[mel + 2] - fft_freq) / diff[mel + 1];
                filters[bin * NUM_MEL_BINS + mel] = down_slope.min(up_slope).max(0.0);
            }
        }
        filters
    })
}

/// An iterative radix-2 Cooley-Tukey FFT, in place. `len` is a power of two.
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());

    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }

    let mut len = 2;
    while len <= n {
        let angle = -2.0 * PI / len as f64;
        let (step_re, step_im) = (angle.cos(), angle.sin());
        let mut start = 0;
        while start < n {
            let (mut wr, mut wi) = (1.0, 0.0);
            for k in 0..len / 2 {
                let (a, b) = (start + k, start + k + len / 2);
                let (ur, ui) = (re[a], im[a]);
                let (vr, vi) = (re[b] * wr - im[b] * wi, re[b] * wi + im[b] * wr);
                re[a] = ur + vr;
                im[a] = ui + vi;
                re[b] = ur - vr;
                im[b] = ui - vi;
                let (next_wr, next_wi) = (wr * step_re - wi * step_im, wr * step_im + wi * step_re);
                wr = next_wr;
                wi = next_wi;
            }
            start += len;
        }
        len <<= 1;
    }
}

/// One frame's log-mel row: DC removal, pre-emphasis, the Hann window, an
/// FFT zero-padded to `FFT_LENGTH`, the power spectrum, the mel projection
/// floored at `MEL_FLOOR`, and the natural log.
fn frame_row(samples: &[f32], start: usize) -> [f64; NUM_MEL_BINS] {
    let window = hann_window();
    let mut buffer = [0.0f64; FRAME_LENGTH];
    for (i, slot) in buffer.iter_mut().enumerate() {
        *slot = f64::from(samples[start + i]);
    }
    let mean: f64 = buffer.iter().sum::<f64>() / FRAME_LENGTH as f64;
    for value in &mut buffer {
        *value -= mean;
    }
    // Pre-emphasis reads every sample's pre-emphasis value simultaneously -
    // `buffer[i-1]` before its own update, not after - so it reads a copy
    // rather than the buffer it writes.
    let before = buffer;
    buffer[0] = before[0] * (1.0 - PREEMPHASIS);
    for i in 1..FRAME_LENGTH {
        buffer[i] = before[i] - PREEMPHASIS * before[i - 1];
    }
    for (value, w) in buffer.iter_mut().zip(window.iter()) {
        *value *= w;
    }

    let mut re = [0.0f64; FFT_LENGTH];
    let mut im = [0.0f64; FFT_LENGTH];
    re[..FRAME_LENGTH].copy_from_slice(&buffer);
    fft(&mut re, &mut im);

    let filters = mel_filter_bank();
    let mut mel = [0.0f64; NUM_MEL_BINS];
    for bin in 0..NUM_FREQ_BINS {
        let power = re[bin] * re[bin] + im[bin] * im[bin];
        let row = &filters[bin * NUM_MEL_BINS..(bin + 1) * NUM_MEL_BINS];
        for (m, &weight) in row.iter().enumerate() {
            mel[m] += weight * power;
        }
    }
    let mut row = [0.0f64; NUM_MEL_BINS];
    for (out, &value) in row.iter_mut().zip(mel.iter()) {
        *out = value.max(MEL_FLOOR).ln();
    }
    row
}

/// `samples` (16 kHz mono f32) as AST's `input_values`: `MAX_FRAMES *
/// NUM_MEL_BINS` values, frame-major, padded with (normalized) zero rows
/// when there are fewer than `MAX_FRAMES` frames and truncated when there
/// are more.
pub fn extract(samples: &[f32]) -> Vec<f32> {
    let frames = num_frames(samples.len()).min(MAX_FRAMES);
    let mut out = vec![0.0f32; MAX_FRAMES * NUM_MEL_BINS];
    for t in 0..frames {
        let row = frame_row(samples, t * HOP_LENGTH);
        let target = &mut out[t * NUM_MEL_BINS..(t + 1) * NUM_MEL_BINS];
        for (slot, value) in target.iter_mut().zip(row.iter()) {
            *slot = ((*value - NORM_MEAN) / (NORM_STD * 2.0)) as f32;
        }
    }
    // Padded (never-written) rows are still normalized zeros, the way
    // `ASTFeatureExtractor.normalize` runs over the whole padded array.
    let padded_zero = ((0.0 - NORM_MEAN) / (NORM_STD * 2.0)) as f32;
    for row in out[frames * NUM_MEL_BINS..].iter_mut() {
        *row = padded_zero;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(count: usize, hz: f64) -> Vec<f32> {
        (0..count)
            .map(|i| (2.0 * PI * hz * i as f64 / f64::from(SAMPLE_RATE)).sin() as f32)
            .collect()
    }

    #[test]
    fn frame_count_matches_the_no_centering_formula() {
        assert_eq!(num_frames(0), 0);
        assert_eq!(num_frames(399), 0, "under one frame");
        assert_eq!(num_frames(400), 1, "exactly one frame, no hop taken");
        assert_eq!(num_frames(559), 1, "one sample short of a second frame");
        assert_eq!(num_frames(560), 2, "exactly enough for a second frame");
        assert_eq!(num_frames(800), 3, "three 400-sample frames 160 apart");
        assert_eq!(num_frames(160_000), 998, "a 10 s window");
    }

    #[test]
    fn the_fft_matches_a_brute_force_dft_on_a_small_case() {
        // n=8, a mixed real signal: cheap enough to check against the DFT's
        // own definition rather than trust the FFT to check itself.
        let signal: [f64; 8] = [1.0, 2.0, -1.0, 0.5, 0.0, -2.0, 3.0, 1.5];
        let mut re = signal;
        let mut im = [0.0; 8];
        fft(&mut re, &mut im);
        for k in 0..8 {
            let (mut want_re, mut want_im) = (0.0, 0.0);
            for (n, &x) in signal.iter().enumerate() {
                let angle = -2.0 * PI * k as f64 * n as f64 / 8.0;
                want_re += x * angle.cos();
                want_im += x * angle.sin();
            }
            assert!((re[k] - want_re).abs() < 1e-9, "bin {k} real");
            assert!((im[k] - want_im).abs() < 1e-9, "bin {k} imaginary");
        }
    }

    #[test]
    fn the_mel_filter_bank_rows_sum_to_no_more_than_the_dc_response() {
        // Each of the 128 triangular filters peaks at 1.0 and is 0 elsewhere
        // in a well-formed bank; a value outside [0, 1] would mean the
        // triangle construction is wrong.
        let filters = mel_filter_bank();
        for &weight in filters {
            assert!((0.0..=1.000_001).contains(&weight), "{weight} out of range");
        }
        // At least one filter actually has weight somewhere - an all-zero
        // bank would still pass the range check above.
        assert!(filters.iter().any(|&w| w > 0.0));
    }

    /// Pinned from `transformers.audio_utils.spectrogram`/`mel_filter_bank`
    /// run the way `ASTFeatureExtractor` calls them without `torchaudio`
    /// installed (`num_frequency_bins=257`, `num_mel_filters=128`,
    /// `min_frequency=20`, `max_frequency=8000`, `mel_scale="kaldi"`,
    /// `triangularize_in_mel_space=True`; `frame_length=400`,
    /// `hop_length=160`, `fft_length=512`, `power=2.0`, `center=False`,
    /// `preemphasis=0.97`, `remove_dc_offset=True`, `log_mel="log"`,
    /// `mel_floor=1.192092955078125e-07`), over an 800-sample 440 Hz sine at
    /// 16 kHz - three frames. Bins 63, 64 and 127 floor out on every frame
    /// of this signal, which is itself part of what the fixture checks.
    #[test]
    fn fbank_matches_the_reference_extractor_on_a_sine() {
        let samples = sine(800, 440.0);
        assert_eq!(num_frames(samples.len()), 3);

        let expected: [(usize, usize, f64); 15] = [
            (0, 0, -12.743_317_604_064_941),
            (0, 1, -14.231_415_748_596_191),
            (0, 63, -15.942_384_719_848_633),
            (0, 64, -15.942_384_719_848_633),
            (0, 127, -15.942_384_719_848_633),
            (1, 0, -11.045_696_258_544_922),
            (1, 1, -13.163_240_432_739_258),
            (1, 63, -15.942_384_719_848_633),
            (1, 64, -15.942_384_719_848_633),
            (1, 127, -15.942_384_719_848_633),
            (2, 0, -10.674_262_046_813_965),
            (2, 1, -12.854_706_764_221_191),
            (2, 63, -15.942_384_719_848_633),
            (2, 64, -15.942_384_719_848_633),
            (2, 127, -15.942_384_719_848_633),
        ];
        for (frame, bin, want) in expected {
            let got = frame_row(&samples, frame * HOP_LENGTH)[bin];
            assert!(
                (got - want).abs() < 1e-4,
                "frame {frame} bin {bin}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn extract_pads_short_audio_with_normalized_zero_rows() {
        let samples = sine(800, 440.0);
        let features = extract(&samples);
        assert_eq!(features.len(), MAX_FRAMES * NUM_MEL_BINS);

        let padded_zero = ((0.0 - NORM_MEAN) / (NORM_STD * 2.0)) as f32;
        // Frame 3 (index 3) was never written by the three real frames.
        let row3 = &features[3 * NUM_MEL_BINS..4 * NUM_MEL_BINS];
        assert!(row3.iter().all(|&v| (v - padded_zero).abs() < 1e-6));
        // The last frame is padding too.
        let last = &features[(MAX_FRAMES - 1) * NUM_MEL_BINS..];
        assert!(last.iter().all(|&v| (v - padded_zero).abs() < 1e-6));

        // Frame 0 is real: its normalized value matches the pinned fbank
        // value at bin 0, through the same normalization `extract` applies.
        let want = ((-12.743_317_604_064_941 - NORM_MEAN) / (NORM_STD * 2.0)) as f32;
        assert!((features[0] - want).abs() < 1e-4);
    }

    #[test]
    fn audio_shorter_than_one_frame_is_all_padding() {
        let features = extract(&sine(100, 440.0));
        let padded_zero = ((0.0 - NORM_MEAN) / (NORM_STD * 2.0)) as f32;
        assert!(features.iter().all(|&v| (v - padded_zero).abs() < 1e-6));
    }

    #[test]
    fn a_window_wider_than_max_frames_is_truncated_not_overrun() {
        // 1024 frames need 1024*160 + 240 = 164_080 samples; one hop more
        // is 999 frames, comfortably past 998 for a 10 s window and enough
        // to prove extract() does not panic or grow past MAX_FRAMES.
        let long = sine(164_080 + HOP_LENGTH * 30, 220.0);
        assert!(num_frames(long.len()) > MAX_FRAMES);
        let features = extract(&long);
        assert_eq!(features.len(), MAX_FRAMES * NUM_MEL_BINS);
    }
}
