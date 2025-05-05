#![feature(float_erf)]
#![feature(iter_array_chunks)]

pub mod math;
pub mod ring;
pub mod spike;
pub mod text;

pub use spike::{ChatSpikeDetector, Event, Phase};
