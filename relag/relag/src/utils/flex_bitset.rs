use std::sync::atomic::{AtomicU64, Ordering};
use bincode::{Decode, Encode};

const BITS: usize = u64::BITS as usize;

/// A growable bitset whose bits can be set and read concurrently through `&self`.
///
/// `insert` publishes with `Release` and `contains` reads with `Acquire`: a thread that
/// observes a bit as set also observes every write the setting thread made before
/// `insert`. Growing requires exclusive access (`&mut self`).
pub struct FlexBitSet {
    words: Vec<AtomicU64>,
    len: usize,
}

impl FlexBitSet {
    /// Create a bitset of `len` bits, all unset.
    pub fn with_capacity(len: usize) -> Self {
        Self {
            words: (0..len.div_ceil(BITS)).map(|_| AtomicU64::new(0)).collect(),
            len,
        }
    }

    /// Create a bitset of `len` bits, all set.
    pub fn filled(len: usize) -> Self {
        let mut set = Self::with_capacity(0);
        set.grow(len);
        set.insert_range(0..len);
        set
    }

    /// Number of bits.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Grow to `len` bits; new bits are unset. Does nothing if `len` is not larger.
    pub fn grow(&mut self, len: usize) {
        if len > self.len {
            self.words.resize_with(len.div_ceil(BITS), || AtomicU64::new(0));
            self.len = len;
        }
    }

    /// Set bit `idx`. Panics if `idx >= len()`.
    pub fn insert(&self, idx: usize) {
        assert!(idx < self.len, "FlexBitSet::insert: index {idx} out of range ({})", self.len);
        self.words[idx / BITS].fetch_or(1 << (idx % BITS), Ordering::Release);
    }

    /// Set every bit in `range`. Panics if `range.end > len()`.
    pub fn insert_range(&mut self, range: std::ops::Range<usize>) {
        assert!(range.end <= self.len, "FlexBitSet::insert_range: end {} out of range ({})", range.end, self.len);
        for idx in range {
            *self.words[idx / BITS].get_mut() |= 1 << (idx % BITS);
        }
    }

    /// Whether bit `idx` is set. Panics if `idx >= len()`.
    pub fn contains(&self, idx: usize) -> bool {
        assert!(idx < self.len, "FlexBitSet::contains: index {idx} out of range ({})", self.len);
        self.words[idx / BITS].load(Ordering::Acquire) & (1 << (idx % BITS)) != 0
    }
}

impl Clone for FlexBitSet {
    fn clone(&self) -> Self {
        Self {
            words: self.words.iter().map(|w| AtomicU64::new(w.load(Ordering::Acquire))).collect(),
            len: self.len,
        }
    }
}

/// Encoded as the bit count followed by the `len().div_ceil(64)` words.
impl Encode for FlexBitSet {
    fn encode<E: bincode::enc::Encoder>(&self, encoder: &mut E) -> Result<(), bincode::error::EncodeError> {
        self.len.encode(encoder)?;
        for word in &self.words {
            word.load(Ordering::Acquire).encode(encoder)?;
        }
        Ok(())
    }
}

impl<Context> Decode<Context> for FlexBitSet {
    fn decode<D: bincode::de::Decoder<Context = Context>>(decoder: &mut D) -> Result<Self, bincode::error::DecodeError> {
        let len = usize::decode(decoder)?;
        let words = (0..len.div_ceil(BITS))
            .map(|_| u64::decode(decoder).map(AtomicU64::new))
            .collect::<Result<_, _>>()?;
        Ok(Self { words, len })
    }
}

bincode::impl_borrow_decode!(FlexBitSet);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_contains() {
        let set = FlexBitSet::with_capacity(130);
        assert!(!set.contains(0));
        set.insert(0);
        set.insert(64);
        set.insert(129);
        assert!(set.contains(0) && set.contains(64) && set.contains(129));
        assert!(!set.contains(1) && !set.contains(63) && !set.contains(128));
    }

    #[test]
    fn grow_keeps_bits_and_adds_unset_bits() {
        let mut set = FlexBitSet::with_capacity(10);
        set.insert(9);
        set.grow(200);
        assert_eq!(set.len(), 200);
        assert!(set.contains(9));
        assert!(!set.contains(10) && !set.contains(199));
        set.grow(5);
        assert_eq!(set.len(), 200);
    }

    #[test]
    fn filled_sets_exactly_len_bits() {
        let mut set = FlexBitSet::filled(70);
        assert!((0..70).all(|i| set.contains(i)));
        set.grow(80);
        assert!((70..80).all(|i| !set.contains(i)));
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn contains_rejects_out_of_range() {
        FlexBitSet::with_capacity(10).contains(10);
    }

    #[test]
    fn bincode_roundtrip() {
        let set = FlexBitSet::with_capacity(130);
        for i in [0, 63, 64, 129] {
            set.insert(i);
        }
        let cfg = bincode::config::standard();
        let bytes = bincode::encode_to_vec(&set, cfg).unwrap();
        let (decoded, _): (FlexBitSet, usize) = bincode::decode_from_slice(&bytes, cfg).unwrap();
        assert_eq!(decoded.len(), 130);
        assert!((0..130).all(|i| decoded.contains(i) == set.contains(i)));
    }

    #[test]
    fn concurrent_inserts_are_all_visible() {
        use rayon::prelude::*;
        let set = FlexBitSet::with_capacity(10_000);
        (0..10_000).into_par_iter().filter(|i| i % 3 == 0).for_each(|i| set.insert(i));
        assert!((0..10_000).all(|i| set.contains(i) == (i % 3 == 0)));
    }
}
