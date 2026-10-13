//! The GPU build: a node beside `ffrwd:gpu`. The host makes each frame the
//! reservoir keeps into a tensor (`ffrwd:gpu/nn.crops-tensor`), and joins the
//! eight the tower reads into its one input (`ffrwd:gpu/nn.stack`), so the
//! frame is never fetched.
//!
//! The tensor is the host build's preprocessing made in one resampling: the
//! frame's centre square, the square the host build's short-side resize and
//! centre crop keep, resized to 224 with bilinear widened where it shrinks,
//! and normalised with ImageNet's mean and deviation. The host build rounds
//! to eight bits once, after its resize; this does not, and the square's
//! edges are whole pixels where the host build's fall between them.
//!
//! `memory` says where `v` is held. On `host`, the default, the frame
//! arrives in host memory and the host makes the tensors on the CPU. On
//! `gpu`, the frame is held on the device in any format the sidecar holds
//! there, the tensors are made on the device, and where the tower's session
//! reads the device's memory the eight are joined there and bound to it with
//! no copy. Either way the vector comes back to host memory.

// `generate_all`: the world's interfaces come from other packages - ffrwd:gpu,
// wasi:webgpu and its deps - and without it bindgen expects them to have been
// generated somewhere else. wasi:nn is the bindings `describe_common` already
// generates, so the tensors `crops-tensor` and `stack` answer are the ones
// `Model` passes to `compute`. The path is the `clips` crate's, so the
// `clips-gpu` crate reaching this file finds it too.
mod bindings {
    wit_bindgen::generate!({
        path: "../clips/wit-gpu",
        world: "ffrwd:gpu/gpu-nn-module@0.1.0",
        with: {
            "wasi:nn/tensor@0.2.0-rc-2024-10-28": describe_common::wasi::nn::tensor,
            "wasi:nn/inference@0.2.0-rc-2024-10-28": describe_common::wasi::nn::inference,
            "wasi:nn/errors@0.2.0-rc-2024-10-28": describe_common::wasi::nn::errors,
        },
        generate_all,
    });
}

use bindings::exports::ffrwd::gpu::ports::Guest;
use bindings::ffrwd::gpu::nn::{
    crops_tensor, stack, ChannelOrder, Fit, FrameTensorOptions, Rect, ResizeFilter, TensorLayout,
};
use bindings::ffrwd::gpu::types::{FrameMemory, PortMemory};
pub use describe_common::wasi::nn::tensor::Tensor;
use serde::Deserialize;

use crate::prep::{MEAN, SIDE, STD};
use crate::reservoir::FRAMES;
use crate::PARAMS_SCHEMA;

/// Where `v`'s frames are held.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Memory {
    #[default]
    Host,
    Gpu,
}

/// The GPU build's params: where `v` is held.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
pub struct Settings {
    #[serde(default)]
    pub memory: Memory,
}

/// The formats `v` takes: on `gpu`, any the sidecar holds on the device; on
/// `host`, the host build's.
pub fn pixel_formats(settings: &Settings) -> &'static [&'static str] {
    match settings.memory {
        Memory::Gpu => &["yuv420p", "rgba", "yuv422p", "yuv444p"],
        Memory::Host => &["yuv420p", "rgba"],
    }
}

/// The square a `width` x `height` frame keeps: its short side, centred on
/// the long one.
pub fn centre_square(width: u32, height: u32) -> Rect {
    let side = width.min(height);
    Rect {
        x: (width - side) / 2,
        y: (height - side) / 2,
        width: side,
        height: side,
    }
}

/// How the host makes a kept frame into the tower's input: the square
/// stretched to 224 with bilinear widened where it shrinks, planar RGB in
/// 0..1 normalised by ImageNet's mean and deviation.
pub fn options(model: &str) -> FrameTensorOptions {
    let side = SIDE as u32;
    FrameTensorOptions {
        model: model.to_string(),
        width: side,
        height: side,
        layout: TensorLayout::Nchw,
        order: ChannelOrder::Rgb,
        fit: Fit::Stretch,
        pad: 0.0,
        scale: 1.0,
        mean: (MEAN[0], MEAN[1], MEAN[2]),
        std: (STD[0], STD[1], STD[2]),
        filter: ResizeFilter::Bilinear,
        antialias: true,
    }
}

/// What a frame of `v` is made into: the square of the frame, as the host
/// makes it.
pub struct Maker {
    v: u32,
    square: Rect,
    options: FrameTensorOptions,
}

impl Maker {
    pub fn new(v: u32, width: u32, height: u32, model: &str) -> Maker {
        Maker {
            v,
            square: centre_square(width, height),
            options: options(model),
        }
    }

    /// Frame `index` of `v` this tick as a `[1, 3, 224, 224]` tensor, which
    /// the module may keep past the tick.
    pub fn tensor(&self, index: u32) -> Result<Tensor, String> {
        crops_tensor(
            self.v,
            index,
            std::slice::from_ref(&self.square),
            &self.options,
        )
        .map_err(|e| format!("clips: {e}"))
    }
}

/// `picks` of the kept frames' tensors joined, in order, as the tower's
/// `[1, 8, 3, 224, 224]` input.
pub fn clip(kept: &[Tensor], picks: &[usize; FRAMES]) -> Result<Tensor, String> {
    let held: Vec<&Tensor> = picks.iter().map(|&at| &kept[at]).collect();
    let side = SIDE as u32;
    stack(&held, &[1, FRAMES as u32, 3, side, side]).map_err(|e| format!("clips: {e}"))
}

struct Ports;

impl Guest for Ports {
    /// `v`, when `memory` is `gpu`. The shots leave as rows, which are not
    /// frames and are held nowhere.
    fn memory(params: String) -> Vec<PortMemory> {
        match ffrwd_node::read_params::<Settings>(PARAMS_SCHEMA, &params) {
            Ok((settings, _)) if settings.memory == Memory::Gpu => vec![PortMemory {
                port: "v".to_string(),
                memory: FrameMemory::Gpu,
            }],
            _ => Vec::new(),
        }
    }
}

bindings::export!(Ports with_types_in bindings);

#[cfg(test)]
mod tests {
    use super::*;
    use ffrwd_node::read_params;

    #[test]
    fn memory_defaults_to_host() {
        let settings =
            |params: &str| read_params::<Settings>(PARAMS_SCHEMA, params).map(|(s, _)| s);
        assert_eq!(settings("").expect("the default").memory, Memory::Host);
        assert_eq!(
            settings(r#"{"memory":"gpu"}"#).expect("gpu").memory,
            Memory::Gpu
        );
        let error = settings(r#"{"memory":"cpu"}"#).expect_err("not a memory");
        assert!(error.contains("memory"), "{error}");
    }

    #[test]
    fn v_is_held_on_the_gpu_only_when_memory_says_so() {
        let held = <Ports as Guest>::memory(r#"{"memory":"gpu"}"#.to_string());
        assert_eq!(held.len(), 1);
        assert_eq!(held[0].port, "v");
        assert!(matches!(held[0].memory, FrameMemory::Gpu));
        for host in ["", "{}", r#"{"memory":"host"}"#, r#"{"memory":"cpu"}"#] {
            assert!(
                <Ports as Guest>::memory(host.to_string()).is_empty(),
                "{host}"
            );
        }
    }

    #[test]
    fn the_square_is_the_short_side_centred_on_the_long_one() {
        let square = centre_square(1920, 1080);
        assert_eq!(
            (square.x, square.y, square.width, square.height),
            (420, 0, 1080, 1080)
        );
        let square = centre_square(1080, 1920);
        assert_eq!(
            (square.x, square.y, square.width, square.height),
            (0, 420, 1080, 1080)
        );
        let square = centre_square(3840, 2160);
        assert_eq!(
            (square.x, square.y, square.width, square.height),
            (840, 0, 2160, 2160)
        );
        let square = centre_square(64, 64);
        assert_eq!((square.x, square.y, square.width), (0, 0, 64));
    }

    #[test]
    fn the_tensor_is_the_host_builds_resize_and_normalization() {
        let options = options("clips");
        assert_eq!(options.model, "clips");
        assert_eq!((options.width, options.height), (224, 224));
        assert!(matches!(options.layout, TensorLayout::Nchw));
        assert!(matches!(options.order, ChannelOrder::Rgb));
        assert!(matches!(options.fit, Fit::Stretch));
        assert!(matches!(options.filter, ResizeFilter::Bilinear));
        assert!(options.antialias, "the reference widens the kernel");
        assert_eq!(options.scale, 1.0);
        assert_eq!(options.mean, (0.485, 0.456, 0.406));
        assert_eq!(options.std, (0.229, 0.224, 0.225));
    }
}
