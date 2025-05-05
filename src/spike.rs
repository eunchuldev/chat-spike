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
//! use chat_spike::{ChatSpikeDetector, Event};
//!
//! let mut det = ChatSpikeDetector::<1, 2>::default()
//!     .with_threshold(0.0, f64::INFINITY); // any activity => spike
//!
//! let e = det.update_and_detect("Hello 🌎".into(), Instant::now());
//! assert!(matches!(e, Event::SpikeBegin { .. }));
//!
//! // Phase can be inspected without advancing time.
//! assert!(matches!(det.current_phase(), chat_spike::Phase::InSpike));
//! ```

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
pub struct ChatWindow<const S: usize, const L: usize> {
    ngram_range: (usize, usize),
    last_chat_idx: u32,
    recent_chats: Ring<ChatCache, S>,
    next_token_id: usize,
    token_dict: HashMap<String, usize>,
    token_stats: Vec<TokenStats>,
}

#[derive(Clone, Default)]
pub struct TokenStats {
    count_s: f64,
    count_l: f64,
    last_chat_idx: u32,
}

#[derive(Clone, Default)]
pub struct ChatCache {
    token_ids: Vec<usize>,
    chat: String,
}

impl<const S: usize, const L: usize> Default for ChatWindow<S, L> {
    fn default() -> Self {
        Self {
            ngram_range: (1, 4),
            last_chat_idx: 0,
            recent_chats: Ring::default(),
            next_token_id: 0,
            token_dict: HashMap::default(),
            token_stats: Vec::default(),
        }
    }
}

impl<const S: usize, const L: usize> ChatWindow<S, L> {
    pub fn with_ngram_range(mut self, min: usize, max: usize) -> Self {
        self.ngram_range = (min, max);
        self
    }
    /// Insert a chat line, updating token statistics.
    pub fn push(&mut self, chat: String) {
        let decay_s = 1. - 1. / (S as f64);
        let decay_l = 1. - 1. / (L as f64);
        self.last_chat_idx += 1;
        let chat = normalize(&chat);
        let tokens = unique_char_ngrams(chat.as_str(), self.ngram_range.0, self.ngram_range.1);
        let token_ids: Vec<_> = tokens
            .into_iter()
            .map(|token| {
                *self.token_dict.entry(token).or_insert_with(|| {
                    let id = self.next_token_id;
                    self.token_stats.push(TokenStats::default());
                    self.next_token_id += 1;
                    id
                })
            })
            .collect();
        token_ids.iter().for_each(|&id| {
            let stats = &mut self.token_stats[id];
            let num_gap = (self.last_chat_idx - stats.last_chat_idx) as f64;
            if num_gap < 10. * L as f64 {
                stats.count_l = stats.count_l * decay_l.powf(num_gap) + 1.;
                stats.count_s = stats.count_s * decay_s.powf(num_gap) + 1.;
            } else {
                stats.count_l = 1.;
                stats.count_s = 1.;
            }
            stats.last_chat_idx = self.last_chat_idx;
        });
        self.recent_chats.push(ChatCache { token_ids, chat });
    }

    /// Return `(chat_text, score)` with the highest degree centrality.
    pub fn summary(&self) -> Option<(&str, f64)> {
        let mut uv = HashMap::<usize, f64>::new();
        for ChatCache { token_ids, .. } in self.recent_chats.iter() {
            let norm2: f64 = token_ids
                .iter()
                .map(|&t| ((L as f64) / self.token_stats[t].count_l).ln().powi(2))
                .sum::<f64>()
                .sqrt();
            for (id, u) in token_ids
                .iter()
                .map(move |&t| (t, ((L as f64) / self.token_stats[t].count_l).ln() / norm2))
            {
                uv.entry(id).and_modify(|v| *v += u).or_insert(u);
            }
        }
        self.recent_chats
            .iter()
            .map(|ChatCache { token_ids, chat }| {
                let norm2: f64 = token_ids
                    .iter()
                    .map(|&t| ((L as f64) / self.token_stats[t].count_l).ln().powi(2))
                    .sum::<f64>()
                    .sqrt();
                let degree_centrality = token_ids
                    .iter()
                    .map(move |&t| (((L as f64) / self.token_stats[t].count_l).ln() / norm2, t))
                    .map(|(u, t)| u * uv.get(&t).unwrap_or(&0.))
                    .sum::<f64>()
                    - 1.0;
                let degree_centrality = if degree_centrality.is_nan() {
                    0.0
                } else {
                    degree_centrality
                };
                (chat.as_str(), degree_centrality)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Less))
    }
}

/// Combines timestamp-based burst detection with content-based summaries.
#[derive(Default)]
pub struct ChatSpikeDetector<const S: usize, const L: usize> {
    spike: SpikeDetector<S, L>,
    recent_chats: ChatWindow<S, L>,
}

/// High-level event emitted by `ChatSpikeDetector`.
#[derive(Clone, Copy, Default, Debug)]
pub enum Event<'a> {
    #[default]
    None,
    SpikeBegin {
        summary: &'a str,
        surprise: f64,
    },
    SpikeEnd {
        summary: &'a str,
        surprise: f64,
    },
}

impl<const S: usize, const L: usize> ChatSpikeDetector<S, L> {
    pub fn with_ngram_range(mut self, min: usize, max: usize) -> Self {
        self.recent_chats = self.recent_chats.with_ngram_range(min, max);
        self
    }
    pub fn with_threshold(mut self, start_t: f64, end_t: f64) -> Self {
        self.spike = self.spike.with_threshold(start_t, end_t);
        self
    }

    /// Add a chat message and return an event when a spike starts or ends.
    pub fn update_and_detect(&mut self, chat: String, ts: Instant) -> Event {
        self.recent_chats.push(chat);
        match self.spike.push(ts) {
            SpikeEvent::Begin { surprise } => Event::SpikeBegin {
                summary: self.recent_chats.summary().unwrap().0,
                surprise,
            },
            SpikeEvent::End { surprise } => Event::SpikeEnd {
                summary: self.recent_chats.summary().unwrap().0,
                surprise,
            },
            _ => Event::None,
        }
    }
    pub fn current_surprise(&self) -> f64 {
        self.spike.current_surprise()
    }
    pub fn current_phase(&self) -> Phase {
        self.spike.phase
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        cw.push("hello world".into());
        cw.push("hello world".into());
        cw.push("some noises".into());
        let summary = cw.summary();
        assert!(summary.is_some());
        assert_eq!(summary.unwrap().0, "hello world");
    }

    #[test]
    fn chat_spike_detector_phase_consistency() {
        let mut det = ChatSpikeDetector::<1, 2>::default().with_threshold(0.0, f64::INFINITY);
        let t0 = Instant::now();
        let ev = det.update_and_detect("hi".into(), t0);
        assert!(matches!(ev, Event::SpikeBegin { .. }));
        assert!(matches!(det.current_phase(), Phase::InSpike));
        assert!(det.current_surprise() >= 0.0);
    }
}
