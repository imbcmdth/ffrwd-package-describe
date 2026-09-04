//! The eight frames one shot is described by, chosen while the shot is still
//! running.
//!
//! A shot's length is not known until it ends, so the reservoir keeps every
//! `stride`-th frame and doubles the stride whenever it holds more than it
//! has room for, dropping every second frame as it does. What it holds is
//! therefore always evenly spread over the shot so far, at a spacing that
//! halves the memory rather than the coverage. Closing it resamples that
//! spread down to the eight the tower takes.

use crate::prep::TENSOR;

/// Frames one clip carries, fixed by the export.
pub const FRAMES: usize = 8;

/// Frames the reservoir holds before it thins. Twice `FRAMES` leaves it never
/// holding fewer than `FRAMES` once a shot is that long, so the resample down
/// to eight always has eight distinct frames to choose from.
const CAPACITY: usize = 2 * FRAMES;

/// One kept frame: where it sat in the shot, and its tensor.
struct Kept {
    index: usize,
    tensor: Vec<f32>,
}

pub struct Reservoir {
    kept: Vec<Kept>,
    /// Frames between the ones kept; doubles as the shot grows.
    stride: usize,
    /// Frames the shot has carried, kept or not.
    seen: usize,
}

impl Default for Reservoir {
    fn default() -> Self {
        Reservoir {
            kept: Vec::with_capacity(CAPACITY + 1),
            stride: 1,
            seen: 0,
        }
    }
}

impl Reservoir {
    /// Offers the shot's next frame. `preprocess` runs only for the frames
    /// the reservoir keeps, so a long shot pays for one frame in `stride`.
    pub fn offer(&mut self, preprocess: impl FnOnce() -> Vec<f32>) {
        if self.seen.is_multiple_of(self.stride) {
            self.kept.push(Kept {
                index: self.seen,
                tensor: preprocess(),
            });
            if self.kept.len() > CAPACITY {
                self.thin();
            }
        }
        self.seen += 1;
    }

    /// Every second frame dropped and the stride doubled, which leaves what
    /// is kept evenly spread at the new spacing.
    fn thin(&mut self) {
        let mut index = 0;
        self.kept.retain(|_| {
            index += 1;
            index % 2 == 1
        });
        self.stride *= 2;
    }

    /// Whether the shot's last frame is one of the kept ones. When it is not,
    /// the caller hands it to `close` so the eight span the whole shot.
    pub fn holds_last(&self) -> bool {
        self.kept.last().map(|k| k.index) == self.seen.checked_sub(1)
    }

    /// The eight frames the tower reads, back to back as one
    /// `[FRAMES, 3, SIDE, SIDE]` tensor, and the reservoir emptied for the
    /// next shot. `last` is the shot's final frame, which `holds_last` says
    /// whether the reservoir needs.
    pub fn close(&mut self, last: Option<Vec<f32>>) -> Vec<f32> {
        if let Some(tensor) = last {
            let index = self.seen.saturating_sub(1);
            self.kept.push(Kept { index, tensor });
        }
        let picked = spread(self.kept.len());
        let mut input = Vec::with_capacity(FRAMES * TENSOR);
        for at in picked {
            input.extend_from_slice(&self.kept[at].tensor);
        }
        *self = Reservoir::default();
        input
    }

    #[cfg(test)]
    fn indices(&self) -> Vec<usize> {
        self.kept.iter().map(|k| k.index).collect()
    }

    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.seen == 0
    }
}

/// Which of `held` frames the eight are taken from: evenly spread over what
/// is there, first and last included. A shot shorter than eight frames
/// repeats some of them, since the tower takes eight either way.
fn spread(held: usize) -> [usize; FRAMES] {
    let mut picked = [0usize; FRAMES];
    if held <= 1 {
        return picked;
    }
    for (slot, at) in picked.iter_mut().enumerate() {
        let along = slot as f64 * (held - 1) as f64 / (FRAMES - 1) as f64;
        *at = along.round() as usize;
    }
    picked
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reservoir fed `count` frames, each tensor one value long standing in
    /// for the real one.
    fn fed(count: usize) -> Reservoir {
        let mut reservoir = Reservoir::default();
        for i in 0..count {
            reservoir.offer(|| vec![i as f32]);
        }
        reservoir
    }

    #[test]
    fn a_short_shot_keeps_every_frame_it_saw() {
        assert_eq!(fed(5).indices(), vec![0, 1, 2, 3, 4]);
        assert_eq!(fed(16).indices(), (0..16).collect::<Vec<_>>());
    }

    #[test]
    fn a_longer_shot_thins_to_an_even_spread() {
        // The seventeenth frame is one too many, so every second frame goes
        // and the stride doubles.
        assert_eq!(fed(17).indices(), vec![0, 2, 4, 6, 8, 10, 12, 14, 16]);
        // And again at the thirty-third.
        assert_eq!(fed(33).indices(), vec![0, 4, 8, 12, 16, 20, 24, 28, 32]);
        assert_eq!(
            fed(60).indices(),
            vec![0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56],
            "sixty frames at a stride of four"
        );
    }

    #[test]
    fn the_spacing_never_leaves_fewer_frames_than_the_tower_takes() {
        for count in 8..200 {
            assert!(
                fed(count).indices().len() >= FRAMES,
                "{count} frames thinned below {FRAMES}"
            );
        }
    }

    #[test]
    fn the_shots_last_frame_is_known_to_be_missing() {
        assert!(fed(16).holds_last(), "a stride of one holds every frame");
        assert!(
            !fed(60).holds_last(),
            "frame 59 is not a multiple of the stride"
        );
        assert!(fed(57).holds_last(), "frame 56 is");
    }

    #[test]
    fn eight_frames_come_out_evenly_spread_over_what_was_kept() {
        assert_eq!(spread(8), [0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(spread(16), [0, 2, 4, 6, 9, 11, 13, 15]);
        assert_eq!(spread(15), [0, 2, 4, 6, 8, 10, 12, 14]);
        assert_eq!(
            spread(3),
            [0, 0, 1, 1, 1, 1, 2, 2],
            "a shot of three frames repeats them"
        );
        assert_eq!(spread(1), [0; FRAMES]);
    }

    #[test]
    fn closing_a_sixty_frame_shot_reads_the_first_frame_the_last_and_six_between() {
        let mut reservoir = fed(60);
        assert!(!reservoir.holds_last());
        // Frame 59 goes in beside the fifteen kept, and eight of the sixteen
        // are taken.
        let input = reservoir.close(Some(vec![59.0]));
        assert_eq!(
            input,
            vec![0.0, 8.0, 16.0, 24.0, 36.0, 44.0, 52.0, 59.0],
            "the first frame, the last, and six evenly between"
        );
        assert!(reservoir.is_empty(), "the reservoir starts the next shot");
    }

    #[test]
    fn closing_a_one_frame_shot_repeats_it_eight_times() {
        let mut reservoir = fed(1);
        assert!(reservoir.holds_last());
        assert_eq!(reservoir.close(None), vec![0.0; FRAMES]);
    }
}
