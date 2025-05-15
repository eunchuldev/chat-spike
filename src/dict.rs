use std::collections::HashMap;

pub trait Dictionary {
    fn observe(&mut self, token: &str, idx: usize);
    fn count(&self, token: &str) -> f64;
}

#[derive(Clone, Default)]
struct TokenEntry {
    count: f64,
    last_updated: usize,
}

const VACUCUME_SKIP_SIZE: usize = 8;

#[derive(Clone, Default)]
pub struct MemoryDictionary<const L: usize> {
    tokens: HashMap<String, TokenEntry>,
    last_vaccumed_size: usize,
    last_vaccumed_idx: usize,
    idx: usize,
}
impl<const L: usize> MemoryDictionary<L> {
    const DECAY_L: f64 = 1.0 - 1.0 / L as f64;
    const SIGMA: f64 = 0.3679; // ~= decay_l.powf(L as f64);

    fn vaccume(&mut self) {
        self.tokens.retain(|_, v| {
            let num_gap = self.idx.saturating_sub(v.last_updated) as i32;
            v.count *= Self::DECAY_L.powi(num_gap);
            v.last_updated = self.idx;
            v.count > Self::SIGMA
        });
        self.last_vaccumed_idx = self.idx;
    }
    fn vaccume_if_required(&mut self) {
        if self.tokens.len() > VACUCUME_SKIP_SIZE
            && self.tokens.len() * 2 > self.last_vaccumed_size * 3
        {
            self.vaccume();
            self.last_vaccumed_size = self.tokens.len();
        }
    }
}

impl<const L: usize> Dictionary for MemoryDictionary<L> {
    fn observe(&mut self, token: &str, idx: usize) {
        self.idx = idx;
        self.tokens
            .entry(token.to_string())
            .and_modify(|c| {
                c.count += 1.;
                c.last_updated = self.idx;
            })
            .or_insert(TokenEntry {
                last_updated: self.idx,
                count: 1.,
            });
        self.vaccume_if_required();
    }
    fn count(&self, token: &str) -> f64 {
        self.tokens
            .get(token)
            .map(|v| {
                let num_gap = self.idx.saturating_sub(v.last_updated) as i32;
                v.count * Self::DECAY_L.powi(num_gap)
            })
            .unwrap_or(0.0)
    }
}
