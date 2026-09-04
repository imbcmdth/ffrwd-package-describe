//! Where one shot ends and the next begins. Each frame's luma is reduced to a
//! small grid of cell averages and compared with the previous frame's: a mean
//! absolute difference above `THRESHOLD` is a cut.
//!
//! The same detector the fleet's `shots` module runs, and the same numbers.
//! It lives here because a package's SQL can name the modules of packages and
//! nothing else, so a package that wants shot bounds has to find them itself.

use crate::prep::PixFmt;

/// Cells a frame is reduced to, per side. Small enough that a moving subject
/// barely moves the average, large enough that a new picture moves all of them.
const GRID: usize = 32;

/// Well clear of the frame-to-frame difference ordinary motion produces, and
/// well below what a hard cut produces, in luma steps.
const THRESHOLD: f64 = 12.0;

/// The luma of one run of one row, totalled. yuv420p carries luma in the
/// first plane; rgba is converted with the standard weights, per pixel, so
/// each one rounds where it always did.
fn row_luma_sum(
    frame: &[u8],
    pix_fmt: PixFmt,
    width: usize,
    y: usize,
    x0: usize,
    x1: usize,
) -> u32 {
    match pix_fmt {
        PixFmt::Yuv420p => frame[y * width + x0..y * width + x1]
            .iter()
            .map(|sample| u32::from(*sample))
            .sum(),
        PixFmt::Rgba => {
            let base = (y * width + x0) * 4;
            frame[base..base + (x1 - x0) * 4]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|pixel| {
                    let r = u32::from(pixel[0]);
                    let g = u32::from(pixel[1]);
                    let b = u32::from(pixel[2]);
                    (r * 299 + g * 587 + b * 114) / 1000
                })
                .sum()
        }
    }
}

/// The frame's luma as `GRID`x`GRID` cell averages, written into `cells`. A
/// frame smaller than the grid repeats pixels rather than leaving cells empty.
///
/// A cell's columns are one contiguous run of each of its rows, so it is
/// totalled a run at a time and the format is decided once per run rather
/// than once per pixel.
pub fn downsample(frame: &[u8], pix_fmt: PixFmt, width: usize, height: usize, cells: &mut Vec<u8>) {
    cells.clear();
    cells.resize(GRID * GRID, 0);
    let columns: Vec<(usize, usize)> = (0..GRID)
        .map(|cx| {
            let x0 = cx * width / GRID;
            (x0, ((cx + 1) * width / GRID).max(x0 + 1))
        })
        .collect();

    for cy in 0..GRID {
        let y0 = cy * height / GRID;
        let y1 = ((cy + 1) * height / GRID).max(y0 + 1);
        for (cx, (x0, x1)) in columns.iter().enumerate() {
            let mut total: u32 = 0;
            for y in y0..y1 {
                total += row_luma_sum(frame, pix_fmt, width, y, *x0, *x1);
            }
            let count = ((y1 - y0) * (x1 - x0)) as u32;
            cells[cy * GRID + cx] = (total / count.max(1)) as u8;
        }
    }
}

/// Mean absolute difference between two frames' cells, in luma steps.
fn mean_abs_diff(a: &[u8], b: &[u8]) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    let total: u32 = a
        .iter()
        .zip(b)
        .map(|(p, q)| u32::from(p.abs_diff(*q)))
        .sum();
    f64::from(total) / a.len() as f64
}

/// Whether the picture changed enough between two frames' cells to be a cut.
pub fn is_cut(previous: &[u8], current: &[u8]) -> bool {
    mean_abs_diff(previous, current) > THRESHOLD
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `width`x`height` rgba frame filled with one grey level.
    fn flat_rgba(width: usize, height: usize, level: u8) -> Vec<u8> {
        std::iter::repeat_n([level, level, level, 255], width * height)
            .flatten()
            .collect()
    }

    fn cells_of(frame: &[u8], width: usize, height: usize) -> Vec<u8> {
        let mut cells = Vec::new();
        downsample(frame, PixFmt::Rgba, width, height, &mut cells);
        cells
    }

    #[test]
    fn a_flat_frame_reduces_to_one_level_in_every_cell() {
        let cells = cells_of(&flat_rgba(64, 64, 90), 64, 64);
        assert_eq!(cells.len(), GRID * GRID);
        assert!(
            cells.iter().all(|c| *c == 90),
            "every cell of a flat frame holds the frame's level"
        );
    }

    #[test]
    fn a_frame_smaller_than_the_grid_still_fills_it() {
        assert_eq!(cells_of(&flat_rgba(8, 8, 40), 8, 8).len(), GRID * GRID);
    }

    #[test]
    fn a_new_picture_is_a_cut_and_a_drift_is_not() {
        let dark = cells_of(&flat_rgba(64, 64, 10), 64, 64);
        let light = cells_of(&flat_rgba(64, 64, 210), 64, 64);
        let drifted = cells_of(&flat_rgba(64, 64, 18), 64, 64);
        assert!(is_cut(&dark, &light), "black to white is a cut");
        assert!(!is_cut(&dark, &drifted), "eight luma steps is not");
        assert!(!is_cut(&dark, &dark), "and a still frame never is");
    }
}
