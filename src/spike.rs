//! Lightweight spike detection over chat streams.
//!
//! This module exposes three layers of abstraction:
//!
//! 1. **`SpikeDetector`** – purely timestamp based burst detector  
//! 2. **`ChatWindow`**  – sliding window + TF-IDF-like weights  
//! 3. **`ChatSpikeDetector`** – combines 1 & 2 into a single API
//!
//! All generics use two const parameters:
//!
//! * **`S`** – “short” horizon (events kept in memory / Ring buffer)  
//! * **`L`** – “long” horizon (used only for exponential decay)
//!
//! ```rust
//! use std::time::Instant;
//! use chat_spike::{ChatSpikeDetector, Event, MemoryDictionary};
//!
//! let mut dict = MemoryDictionary::<2>::default();
//! let mut det = ChatSpikeDetector::<1, 2>::default()
//!     .with_threshold(0.0, f64::INFINITY); // any activity => spike
//!
//! let e = det.update_and_detect("Hello 🌎".into(), Instant::now(), &mut dict);
//! assert!(matches!(e, Event::SpikeBegin { .. }));
//!
//! // Phase can be inspected without advancing time.
//! assert!(matches!(det.current_phase(), chat_spike::Phase::InSpike));
//! ```

use crate::dict::Dictionary;
use crate::math::neg_ln_poisson_tail;
use crate::ring::Ring;
use crate::text::{normalize, unique_char_ngrams};
use std::collections::HashMap;
use std::time::Instant;

/// Detects bursts of activity in a stream of timestamps.
///
/// * `S`: short-term window size  
/// * `L`: long-term  window size
///
/// When the surprise score  
/// `−ln P(X ≥ S | λ = dur_s · L / dur_l)`  
/// rises above `start_t`, a *spike* begins; it ends once the score
/// drops below `end_t`.
///
/// See the module-level examples for a minimal live demo.
pub struct SpikeDetector<const S: usize, const L: usize> {
    dur_s: f64,
    dur_l: f64,
    start_t: f64,
    end_t: f64,
    last_ts: Option<Instant>,
    phase: Phase,
}

#[derive(Clone, Copy, Default, Debug)]
pub enum Phase {
    #[default]
    Idle,
    InSpike,
}

#[derive(Clone, Copy, Default, Debug)]
pub enum SpikeEvent {
    #[default]
    None,
    Begin {
        surprise: f64,
    },
    End {
        surprise: f64,
    },
}

impl<const S: usize, const L: usize> Default for SpikeDetector<S, L> {
    fn default() -> Self {
        Self {
            dur_s: 0.,
            dur_l: 0.,
            start_t: 2.5,
            end_t: 1.25,
            last_ts: None,
            phase: Phase::Idle,
        }
    }
}

impl<const S: usize, const L: usize> SpikeDetector<S, L> {
    pub fn with_threshold(mut self, start_t: f64, end_t: f64) -> Self {
        self.start_t = start_t;
        self.end_t = end_t;
        self
    }
    pub fn current_surprise(&self) -> f64 {
        let λ_null = self.dur_s * (L as f64) / self.dur_l;
        neg_ln_poisson_tail(S as f64, λ_null)
    }
    /// Feed the next timestamp and return a spike event, if any.
    pub fn push(&mut self, ts: Instant) -> SpikeEvent {
        let decay_s = 1. - 1. / (S as f64);
        let decay_l = 1. - 1. / (L as f64);
        let time_gap = self
            .last_ts
            .map_or(0.0, |prev| ts.duration_since(prev).as_secs_f64())
            .max(f64::EPSILON);
        self.dur_s = self.dur_s * decay_s + time_gap;
        self.dur_l = self.dur_l * decay_l + time_gap;
        self.last_ts = Some(ts);
        let surprise = self.current_surprise();
        match self.phase {
            Phase::Idle if surprise > self.start_t => {
                self.phase = Phase::InSpike;
                SpikeEvent::Begin { surprise }
            }
            Phase::InSpike if surprise < self.end_t => {
                self.phase = Phase::Idle;
                SpikeEvent::End { surprise }
            }
            _ => SpikeEvent::None,
        }
    }
}

/// Sliding window of recent chats with TF-IDF-like weighting.
///
/// Short/long horizons reuse the same `S`/`L` parameters as `SpikeDetector`.
#[derive(Clone)]
pub struct ChatWindow<const S: usize, const L: usize, D = ()> {
    recent_chats: Ring<ChatCache<D>, S>,
    ngram_range: (usize, usize),
    last_chat_idx: usize,
}

#[derive(Clone, Default)]
pub struct ChatCache<D> {
    chat: String,
    data: Option<D>,
}

impl<const S: usize, const L: usize, D> Default for ChatWindow<S, L, D> {
    fn default() -> Self {
        Self {
            recent_chats: Ring::default(),
            ngram_range: (1, 4),
            last_chat_idx: 0,
        }
    }
}

impl<const S: usize, const L: usize, D> ChatWindow<S, L, D> {
    /// Insert a chat line, updating token statistics.
    pub fn with_ngram_range(mut self, min: usize, max: usize) -> Self {
        self.ngram_range = (min, max);
        self
    }
    pub fn push(&mut self, chat: String) {
        self.push_with_data(chat, None)
    }
    pub fn push_with_dict<DI: Dictionary>(&mut self, chat: String, dict: &mut DI) {
        self.push_with_data_and_dict(chat, None, dict);
    }
    pub fn push_with_data_and_dict<DI: Dictionary>(
        &mut self,
        chat: String,
        data: Option<D>,
        dict: &mut DI,
    ) {
        self.last_chat_idx += 1;
        let chat = normalize(&chat);
        let tokens = unique_char_ngrams(chat.as_str(), self.ngram_range.0, self.ngram_range.1);
        tokens
            .into_iter()
            .for_each(|token| dict.observe(&token, self.last_chat_idx));
        self.push_with_data(chat, data)
    }
    pub fn push_with_data(&mut self, chat: String, data: Option<D>) {
        self.recent_chats.push(ChatCache { chat, data });
    }

    /// Return `(chat_text, Option<data>, score)` with the highest degree centrality.
    pub fn summary_with_dict<DI: Dictionary>(&self, dict: &DI) -> Option<(&str, Option<&D>, f64)> {
        let mut uv = HashMap::<&str, f64>::new();
        let tokenses: Vec<_> = self
            .recent_chats
            .iter()
            .map(|c| unique_char_ngrams(c.chat.as_str(), self.ngram_range.0, self.ngram_range.1))
            .collect();
        for tokens in tokenses.iter() {
            let norm2: f64 = tokens
                .iter()
                .map(|t| ((L as f64) / dict.count(t).max(1.0)).ln().powi(2))
                .sum::<f64>()
                .sqrt();
            for (id, u) in tokens
                .iter()
                .map(|t| (t, ((L as f64) / dict.count(t).max(1.0)).ln() / norm2))
            {
                uv.entry(id).and_modify(|v| *v += u).or_insert(u);
            }
        }
        self.recent_chats
            .iter()
            .zip(tokenses.iter())
            .map(|(ChatCache { chat, data }, tokens)| {
                let norm2: f64 = tokens
                    .iter()
                    .map(|t| ((L as f64) / dict.count(t).max(1.0)).ln().powi(2))
                    .sum::<f64>()
                    .sqrt();
                let degree_centrality = tokens
                    .iter()
                    .map(|t| (((L as f64) / dict.count(t).max(1.0)).ln() / norm2, t))
                    .map(|(u, t)| u * uv.get(&t.as_str()).unwrap_or(&0.))
                    .sum::<f64>()
                    - 1.0;
                let degree_centrality = if degree_centrality.is_nan() {
                    0.0
                } else {
                    degree_centrality
                };
                (chat.as_str(), data.as_ref(), degree_centrality)
            })
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Less))
    }
}

/// Combines timestamp-based burst detection with content-based summaries.
#[derive(Default)]
pub struct ChatSpikeDetector<const S: usize, const L: usize, D = ()> {
    spike: SpikeDetector<S, L>,
    recent_chats: ChatWindow<S, L, D>,
}

/// High-level event emitted by `ChatSpikeDetector`.
#[derive(Clone, Copy, Default, Debug)]
pub enum Event<'a, D> {
    #[default]
    None,
    SpikeBegin {
        summary: Option<&'a str>,
        data: Option<&'a D>,
        surprise: f64,
    },
    SpikeEnd {
        summary: Option<&'a str>,
        data: Option<&'a D>,
        surprise: f64,
    },
}

impl<const S: usize, const L: usize, D> ChatSpikeDetector<S, L, D> {
    pub fn with_ngram_range(mut self, min: usize, max: usize) -> Self {
        self.recent_chats = self.recent_chats.with_ngram_range(min, max);
        self
    }
    pub fn with_threshold(mut self, start_t: f64, end_t: f64) -> Self {
        self.spike = self.spike.with_threshold(start_t, end_t);
        self
    }

    /// Add a chat message and return an event when a spike starts or ends.
    pub fn update_and_detect<DI: Dictionary>(
        &mut self,
        chat: String,
        ts: Instant,
        dict: &mut DI,
    ) -> Event<D> {
        self.update_and_detect_with_data(chat, ts, None, dict)
    }
    pub fn update_and_detect_with_data<DI: Dictionary>(
        &mut self,
        chat: String,
        ts: Instant,
        data: Option<D>,
        dict: &mut DI,
    ) -> Event<D> {
        self.recent_chats.push_with_data_and_dict(chat, data, dict);
        match self.spike.push(ts) {
            SpikeEvent::Begin { surprise } => {
                let summary = self.recent_chats.summary_with_dict(dict);
                Event::SpikeBegin {
                    summary: summary.map(|s| s.0),
                    data: summary.and_then(|s| s.1),
                    surprise,
                }
            }
            SpikeEvent::End { surprise } => {
                let summary = self.recent_chats.summary_with_dict(dict);
                Event::SpikeEnd {
                    summary: summary.map(|s| s.0),
                    data: summary.and_then(|s| s.1),
                    surprise,
                }
            }
            _ => Event::None,
        }
    }

    pub fn current_surprise(&self) -> f64 {
        self.spike.current_surprise()
    }
    pub fn current_phase(&self) -> Phase {
        self.spike.phase
    }
    pub fn last_updated_at(&self) -> Option<Instant> {
        self.spike.last_ts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dict::MemoryDictionary;

    #[test]
    fn spike_detector_triggers_begin() {
        let mut sd = SpikeDetector::<1, 2>::default().with_threshold(0.0, f64::INFINITY); // extremely low threshold

        // First event should start a spike immediately.
        let now = Instant::now();
        assert!(matches!(sd.push(now), SpikeEvent::Begin { .. }));
        assert!(matches!(sd.phase, Phase::InSpike));
    }

    #[test]
    fn chat_window_summary_nonempty() {
        let mut cw = ChatWindow::<3, 12>::default();
        let mut dict = MemoryDictionary::<12>::default();
        cw.push_with_dict("hello world".into(), &mut dict);
        cw.push_with_dict("hello world".into(), &mut dict);
        cw.push_with_dict("some noises".into(), &mut dict);
        let summary = cw.summary_with_dict(&dict);
        assert!(summary.is_some());
        assert_eq!(summary.unwrap().0, "hello world");
    }

    #[test]
    fn chat_window_summary_with_data() {
        let mut cw = ChatWindow::<3, 12, usize>::default();
        let mut dict = MemoryDictionary::<12>::default();
        cw.push_with_data_and_dict("hello world".into(), Some(1), &mut dict);
        cw.push_with_data_and_dict("hello world".into(), Some(2), &mut dict);
        cw.push_with_data_and_dict("some noises".into(), Some(3), &mut dict);
        let summary = cw.summary_with_dict(&dict);
        assert!(summary.is_some());
        assert_eq!(summary.unwrap().0, "hello world");
        assert_eq!(summary.unwrap().1, Some(&2));
    }

    #[test]
    fn chat_spike_detector_phase_consistency() {
        let mut det = ChatSpikeDetector::<1, 2>::default().with_threshold(0.0, f64::INFINITY);
        let mut dict = MemoryDictionary::<12>::default();
        let t0 = Instant::now();
        let ev = det.update_and_detect("hi".into(), t0, &mut dict);
        assert!(matches!(ev, Event::SpikeBegin { .. }));
        assert!(matches!(det.current_phase(), Phase::InSpike));
        assert!(det.current_surprise() >= 0.0);
    }
}
