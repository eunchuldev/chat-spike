#[derive(Clone, Debug)]
pub struct Ring<T, const S: usize> {
    buf: [Option<T>; S],
    offset: usize,
    size: usize,
}

impl<T, const S: usize> Default for Ring<T, S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const S: usize> Ring<T, S> {
    pub fn new() -> Self {
        Self {
            buf: [const { None }; S],
            offset: 0,
            size: 0,
        }
    }
    pub fn push(&mut self, t: T) -> Option<T> {
        let last = self.buf[self.offset].take();
        self.buf[self.offset] = Some(t);
        self.offset = (self.offset + 1) % S;
        self.size = (self.size + 1).min(S);
        last
    }
    pub fn iter(&self) -> RingIterator<'_, T, S> {
        RingIterator {
            ring: self,
            index: 0,
        }
    }
}

pub struct RingIterator<'a, T, const S: usize> {
    ring: &'a Ring<T, S>,
    index: usize,
}

impl<'a, T, const S: usize> Iterator for RingIterator<'a, T, S> {
    type Item = &'a T;
    fn next(&mut self) -> Option<&'a T> {
        let res = if self.index >= self.ring.size {
            None
        } else {
            self.ring.buf[(self.ring.offset + self.index) % self.ring.size].as_ref()
        };
        self.index += 1;
        res
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_iter_empty() {
        let ring = Ring::<String, 3>::new();
        assert_eq!(ring.iter().next(), None);
    }
    #[test]
    fn ring_iter_full() {
        let mut ring = Ring::<String, 3>::new();
        ring.push("1".to_string());
        ring.push("2".to_string());
        ring.push("3".to_string());
        ring.push("4".to_string());
        assert_eq!(ring.iter().collect::<Vec<_>>(), vec!["2", "3", "4"]);
    }
    #[test]
    fn ring_iter_partial() {
        let mut ring = Ring::<String, 3>::new();
        ring.push("1".to_string());
        ring.push("2".to_string());
        assert_eq!(ring.iter().collect::<Vec<_>>(), vec!["1", "2"]);
    }
}
