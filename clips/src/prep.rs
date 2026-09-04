//! One frame turned into the tensor the video tower takes: the short side
//! resized to 224, a 224x224 centre crop out of that, and the ImageNet
//! normalization the checkpoint's own `preprocessor_config.json` names -
//! `size` 224 with `default_to_square` false, `crop_size` 224x224,
//! `resample` 2 (bilinear), mean 0.485/0.456/0.406, std 0.229/0.224/0.225.
//!
//! The resize is the antialiased bilinear the reference pipeline uses: a
//! triangle filter whose support widens with the reduction, so a frame
//! shrunk five-fold averages the pixels it passes over rather than sampling
//! one in five. It runs on 8-bit values and rounds back to 8 bits before the
//! crop, which is where the reference puts its own rescale.

/// The side of the square the tower is exported at.
pub const SIDE: usize = 224;

/// What the short side is resized to before the crop. Equal to `SIDE` here,
/// and the two are different quantities.
const SHORT_EDGE: usize = 224;

const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// One frame's tensor: three planes of `SIDE`x`SIDE`.
const PLANE: usize = SIDE * SIDE;
pub const TENSOR: usize = 3 * PLANE;

/// The pixel format an instance was opened for, fixed for its life.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PixFmt {
    Yuv420p,
    Rgba,
}

impl PixFmt {
    /// The format the host named, or an error naming what it was.
    pub fn parse(named: &str) -> Result<PixFmt, String> {
        match named {
            "yuv420p" => Ok(PixFmt::Yuv420p),
            "rgba" => Ok(PixFmt::Rgba),
            other => Err(format!("clips does not accept pixel format {other}")),
        }
    }
}

/// The size the short-side resize lands on, as (height, width). The short
/// side becomes `SHORT_EDGE` and the long one keeps the shape, truncated -
/// which is what the reference's own integer conversion does.
pub fn resized_size(width: usize, height: usize) -> (usize, usize) {
    let (short, long) = if width <= height {
        (width, height)
    } else {
        (height, width)
    };
    let new_long = (SHORT_EDGE as f64 * long as f64 / short as f64) as usize;
    let new_long = new_long.max(1);
    if width <= height {
        (new_long, SHORT_EDGE)
    } else {
        (SHORT_EDGE, new_long)
    }
}

/// The taps and weights one output sample of an antialiased resize reads.
/// `start` is the first input sample, `weights` the run beginning there,
/// already normalized so they sum to one.
struct Tap {
    start: usize,
    weights: Vec<f32>,
}

/// One axis of the resize: which input samples each output sample mixes.
/// The triangle filter's support is the reduction factor when shrinking and
/// one sample when growing, so shrinking averages and growing interpolates.
fn taps(out_size: usize, in_size: usize) -> Vec<Tap> {
    let scale = in_size as f64 / out_size as f64;
    let (support, inverse) = if scale >= 1.0 {
        (scale, 1.0 / scale)
    } else {
        (1.0, 1.0)
    };
    (0..out_size)
        .map(|i| {
            let center = scale * (i as f64 + 0.5);
            let start = ((center - support + 0.5) as isize).max(0) as usize;
            let end = ((center + support + 0.5) as usize).min(in_size);
            let mut weights: Vec<f32> = (start..end)
                .map(|j| {
                    let x = ((j as f64 + 0.5 - center) * inverse).abs();
                    if x < 1.0 {
                        (1.0 - x) as f32
                    } else {
                        0.0
                    }
                })
                .collect();
            let total: f32 = weights.iter().sum();
            if total != 0.0 {
                for w in &mut weights {
                    *w /= total;
                }
            }
            Tap { start, weights }
        })
        .collect()
}

/// One frame row as red, green and blue, a channel at a time so each is
/// contiguous.
fn row_to_rgb(
    frame: &[u8],
    pix_fmt: PixFmt,
    width: usize,
    height: usize,
    y: usize,
    out: &mut [f32],
) {
    let (red, rest) = out.split_at_mut(width);
    let (green, blue) = rest.split_at_mut(width);
    match pix_fmt {
        PixFmt::Rgba => {
            for (x, pixel) in frame[y * width * 4..(y + 1) * width * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .enumerate()
            {
                red[x] = f32::from(pixel[0]);
                green[x] = f32::from(pixel[1]);
                blue[x] = f32::from(pixel[2]);
            }
        }
        PixFmt::Yuv420p => {
            let pixels = width * height;
            let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
            let chroma = cw * ch;
            let luma = &frame[y * width..(y + 1) * width];
            let crow = (y / 2).min(ch - 1) * cw;
            for x in 0..width {
                let l = f32::from(luma[x]);
                let ci = crow + (x / 2).min(cw - 1);
                let u = f32::from(frame[pixels + ci]) - 128.0;
                let v = f32::from(frame[pixels + chroma + ci]) - 128.0;
                // The usual BT.601 inverse, in full range: the frames a module
                // is handed are what the host decoded, not studio-swing video.
                red[x] = l + 1.402 * v;
                green[x] = l - 0.344_136 * u - 0.714_136 * v;
                blue[x] = l + 1.772 * u;
            }
        }
    }
}

/// The frame resized, cropped and normalized: one `[3, SIDE, SIDE]` tensor,
/// red, green and blue in turn.
pub fn preprocess(frame: &[u8], pix_fmt: PixFmt, width: usize, height: usize) -> Vec<f32> {
    let (out_h, out_w) = resized_size(width, height);
    let columns = taps(out_w, width);
    let rows = taps(out_h, height);

    // Every source row resized horizontally, kept as three planes of
    // `height` x `out_w`: the vertical pass mixes whole rows of it.
    let mut wide = vec![0f32; 3 * height * out_w];
    let mut rgb = vec![0f32; width * 3];
    for y in 0..height {
        row_to_rgb(frame, pix_fmt, width, height, y, &mut rgb);
        for channel in 0..3 {
            let source = &rgb[channel * width..(channel + 1) * width];
            let target = &mut wide[(channel * height + y) * out_w..][..out_w];
            for (sample, tap) in target.iter_mut().zip(&columns) {
                *sample = tap
                    .weights
                    .iter()
                    .enumerate()
                    .map(|(k, w)| source[tap.start + k] * w)
                    .sum();
            }
        }
    }

    // The crop, in the resized picture's own coordinates.
    let top = (out_h - SIDE) / 2;
    let left = (out_w - SIDE) / 2;

    let mut tensor = vec![0f32; TENSOR];
    for channel in 0..3 {
        let plane = &wide[channel * height * out_w..(channel + 1) * height * out_w];
        for y in 0..SIDE {
            let tap = &rows[top + y];
            let target = &mut tensor[channel * PLANE + y * SIDE..][..SIDE];
            for (x, sample) in target.iter_mut().enumerate() {
                let column = left + x;
                let mixed: f32 = tap
                    .weights
                    .iter()
                    .enumerate()
                    .map(|(k, w)| plane[(tap.start + k) * out_w + column] * w)
                    .sum();
                // The reference resizes 8-bit pixels and hands 8-bit pixels
                // to the crop, so the rounding happens here rather than at
                // the end.
                let eight_bit = mixed.round().clamp(0.0, 255.0);
                *sample = (eight_bit / 255.0 - MEAN[channel]) / STD[channel];
            }
        }
    }
    tensor
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `width`x`height` rgba frame filled with one colour.
    fn flat_rgba(width: usize, height: usize, colour: [u8; 3]) -> Vec<u8> {
        std::iter::repeat_n([colour[0], colour[1], colour[2], 255], width * height)
            .flatten()
            .collect()
    }

    #[test]
    fn the_short_side_lands_on_the_edge_and_the_long_one_keeps_its_shape() {
        // 320x240: the short side is 240, and 224 * 320 / 240 is 298.66,
        // truncated to 298.
        assert_eq!(resized_size(320, 240), (224, 298));
        // Portrait is the same the other way up.
        assert_eq!(resized_size(240, 320), (298, 224));
        // A square frame lands square.
        assert_eq!(resized_size(512, 512), (224, 224));
        // 1920x1080: 224 * 1920 / 1080 is 398.22.
        assert_eq!(resized_size(1920, 1080), (224, 398));
    }

    #[test]
    fn one_axis_of_taps_covers_every_input_sample_exactly_once() {
        // Shrinking four to one: each output sample averages its own
        // quarter, and the four quarters together read every sample.
        let axis = taps(1, 4);
        assert_eq!(axis.len(), 1);
        assert_eq!(axis[0].start, 0);
        let total: f32 = axis[0].weights.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "the weights are normalized");
        assert!(
            axis[0].weights.len() >= 4,
            "the support widens with the reduction, so all four samples are read"
        );
    }

    #[test]
    fn an_identity_resize_reads_each_sample_squarely() {
        for tap in taps(4, 4) {
            let biggest = tap.weights.iter().cloned().fold(f32::MIN, f32::max);
            assert!(
                (biggest - 1.0).abs() < 1e-6,
                "an unresized axis takes one sample whole"
            );
        }
    }

    #[test]
    fn a_flat_frame_normalizes_to_one_value_per_channel() {
        // 51/255 is exactly 0.2, so each channel lands on
        // (0.2 - mean) / std with nothing to interpolate.
        let frame = flat_rgba(64, 48, [51, 51, 51]);
        let tensor = preprocess(&frame, PixFmt::Rgba, 64, 48);
        assert_eq!(tensor.len(), TENSOR);
        for channel in 0..3 {
            let expected = (0.2 - MEAN[channel]) / STD[channel];
            for sample in &tensor[channel * PLANE..(channel + 1) * PLANE] {
                assert!(
                    (sample - expected).abs() < 1e-5,
                    "channel {channel}: got {sample}, want {expected}"
                );
            }
        }
    }

    #[test]
    fn the_crop_keeps_the_middle_of_a_wide_frame() {
        // 448x224 halves to 448x224 -> resized (224, 448), cropped to the
        // middle 224 columns, which is source columns 224..672 of 896.
        // A frame black on the left third and white on the right two thirds
        // therefore comes out entirely white in the crop's own right half.
        let (width, height) = (896usize, 448usize);
        let mut frame = flat_rgba(width, height, [0, 0, 0]);
        for y in 0..height {
            for x in width / 2..width {
                let base = (y * width + x) * 4;
                frame[base] = 255;
                frame[base + 1] = 255;
                frame[base + 2] = 255;
            }
        }
        let tensor = preprocess(&frame, PixFmt::Rgba, width, height);
        let white = (1.0 - MEAN[0]) / STD[0];
        let black = (0.0 - MEAN[0]) / STD[0];
        // The crop spans source columns 224..672 of the 448-wide resize,
        // i.e. the middle half of the picture; its own last column is the
        // source's 671st, well inside the white half.
        let row = &tensor[112 * SIDE..][..SIDE];
        assert!(
            (row[SIDE - 1] - white).abs() < 1e-4,
            "got {}",
            row[SIDE - 1]
        );
        assert!((row[0] - black).abs() < 1e-4, "got {}", row[0]);
    }

    #[test]
    fn neutral_chroma_yuv_is_grayscale_rgb() {
        let (width, height) = (4usize, 4usize);
        let mut frame = vec![100u8; width * height];
        frame.extend(vec![128u8; 2 * 2 * 2]);
        let mut rgb = vec![0f32; width * 3];
        row_to_rgb(&frame, PixFmt::Yuv420p, width, height, 0, &mut rgb);
        for x in 0..width {
            assert!((rgb[x] - 100.0).abs() < 0.01, "red is the luma");
            assert!((rgb[width + x] - 100.0).abs() < 0.01);
            assert!((rgb[2 * width + x] - 100.0).abs() < 0.01);
        }
    }

    /// A 32x32 rgba ramp: red is the column, green the row, blue their sum.
    /// Small enough that the reference below is a short list of numbers.
    fn ramp_rgba() -> Vec<u8> {
        let mut frame = Vec::with_capacity(32 * 32 * 4);
        for y in 0..32u32 {
            for x in 0..32u32 {
                frame.push((x * 8) as u8);
                frame.push((y * 8) as u8);
                frame.push(((x + y) * 4) as u8);
                frame.push(255);
            }
        }
        frame
    }

    #[test]
    fn the_ramp_matches_the_reference_preprocessing() {
        // Reference values from the checkpoint's own pipeline, computed with
        // torch: bilinear antialiased resize of the 8-bit frame to the short
        // edge, rounded back to 8 bits, centre-cropped, rescaled by 1/255 and
        // normalized. A 32x32 frame is square, so the resize is 32 -> 224 and
        // the crop takes everything.
        let tensor = preprocess(&ramp_rgba(), PixFmt::Rgba, 32, 32);
        // Corners and centre of each plane: (channel, y, x, value).
        let expected: [(usize, usize, usize, f32); 9] = [
            (0, 0, 0, -2.117_904),
            (0, 0, 223, 2.129_035),
            (0, 223, 0, -2.117_904),
            (1, 0, 0, -2.035_714),
            (1, 111, 111, 0.117_647),
            (1, 223, 223, 2.306_022),
            (2, 0, 0, -1.804_444),
            (2, 111, 111, 0.339_346),
            (2, 223, 223, 2.517_996),
        ];
        for (channel, y, x, want) in expected {
            let got = tensor[channel * PLANE + y * SIDE + x];
            // One 8-bit step is 1/255 before normalization, and the reference
            // rounds to 8 bits as this does, so the two agree to well inside
            // one step.
            assert!(
                (got - want).abs() < 0.01,
                "channel {channel} at ({y}, {x}): got {got}, want {want}"
            );
        }
    }
}
