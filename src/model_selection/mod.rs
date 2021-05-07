pub use std::borrow::Cow;

use rand::{prelude::ThreadRng, seq::SliceRandom};
use std::{borrow::Borrow, fmt};

mod kfold;

#[derive(Debug)]
pub struct Dataset<'a, T: Clone> {
    records: Vec<Cow<'a, T>>,
    generator: ThreadRng,
}

impl<'a, T: Clone> Dataset<'a, T> {
    /// Creates a new `Dataset` from a vector.
    ///
    /// # Parameters
    /// * `records` - A non-empty vector of records to store.
    ///
    /// # Panics
    /// This function will panic if `records` is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use neuronika::model_selection::{Dataset, Cow};
    ///
    /// let dataset = Dataset::new((0..3).collect());
    /// assert_eq!(&dataset[..], &[Cow::Owned(0), Cow::Owned(1), Cow::Owned(2)]);
    /// ```
    pub fn new(mut records: Vec<T>) -> Self {
        if records.is_empty() {
            panic!("cannot build an empty dataset");
        }

        Self {
            records: records.drain(..).map(Cow::Owned).collect(),
            generator: rand::thread_rng(),
        }
    }

    /// Creates a new `Dataset`, sharing the content of `self`.
    ///
    /// # Example
    ///
    /// ```
    /// use neuronika::model_selection::{Dataset, Cow};
    ///
    /// let owned = Dataset::new((0..10).collect());
    /// let first = owned.share();
    /// assert_eq!(
    ///     first.iter().all(|v| matches!(v, Cow::Borrowed(_))),
    ///     true
    /// );
    ///
    /// let second = first.share();
    /// assert_eq!(
    ///     second.iter().all(|v| matches!(v, Cow::Borrowed(_))),
    ///     true
    /// );
    /// ```
    pub fn share<'b: 'a>(&'b self) -> Self {
        Self {
            records: self
                .records
                .iter()
                .map(|v| Cow::Borrowed(v.borrow()))
                .collect(),
            generator: self.generator.clone(),
        }
    }

    /// Returns the number of elements in the dataset, also referred to
    /// as its 'length'.
    ///
    /// # Examples
    ///
    /// ```
    /// use neuronika::model_selection::Dataset;
    ///
    /// let dataset = Dataset::new((0..5).collect());
    /// assert_eq!(dataset.len(), 5);
    /// ```
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns an iterator over the records of the `Dataset`.
    ///
    /// # Example
    /// ```
    /// use neuronika::model_selection::{Dataset, Cow};
    ///
    /// let dataset = Dataset::new((0..5).collect::<Vec<usize>>());
    /// for (i, cow) in dataset.iter().enumerate() {
    ///     assert_eq!(cow, &Cow::Owned(i));
    /// }
    /// ```
    pub fn iter<'b: 'a>(&'b self) -> std::slice::Iter<'b, Cow<'a, T>> {
        self.records.iter()
    }

    /// Returns an iterator that allows modifying the records of the dataset.
    ///
    /// # Example
    /// ```
    /// use neuronika::model_selection::{Dataset, Cow};
    ///
    /// let mut dataset = Dataset::new((0..3).collect());
    /// for cow in dataset.iter_mut() {
    ///     *cow.to_mut() += 1;
    /// }
    /// ```
    pub fn iter_mut<'b: 'a>(&'b mut self) -> std::slice::IterMut<'b, Cow<'a, T>> {
        self.records.iter_mut()
    }

    /// Shuffles the records in the dataset.
    ///
    /// # Example
    /// ```
    /// use neuronika::model_selection::Dataset;
    ///
    /// let mut dataset = Dataset::new((0..10).collect());
    /// assert_eq!(dataset[..].windows(2).all(|w| w[0] <= w[1]), true);
    ///
    /// dataset.shuffle();
    /// assert_eq!(dataset[..].windows(2).all(|w| w[0] <= w[1]), false);
    /// ```
    pub fn shuffle(&mut self) {
        self.records.shuffle(&mut self.generator);
    }
}

impl<'a, T: Clone> IntoIterator for Dataset<'a, T> {
    type Item = Cow<'a, T>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.records.into_iter()
    }
}

impl<'a, 'b, T: Clone> IntoIterator for &'b Dataset<'a, T> {
    type Item = &'b Cow<'a, T>;
    type IntoIter = std::slice::Iter<'b, Cow<'a, T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.records.iter()
    }
}

impl<'a, 'b, T: Clone> IntoIterator for &'b mut Dataset<'a, T> {
    type Item = &'b mut Cow<'a, T>;
    type IntoIter = std::slice::IterMut<'b, Cow<'a, T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.records.iter_mut()
    }
}

impl<'a, T, U> std::ops::Index<U> for Dataset<'a, T>
where
    T: Clone,
    U: std::slice::SliceIndex<[Cow<'a, T>]>,
{
    type Output = U::Output;

    fn index(&self, index: U) -> &Self::Output {
        &self.records[index]
    }
}

impl<'a, T: Clone + fmt::Display> fmt::Display for Dataset<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        let last = self.records.len() - 1;
        for cow in &self.records[..last] {
            write!(f, "{}, ", &*cow)?;
        }
        write!(f, "{}]", self.records[last])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creation() {
        let dataset = Dataset::new((0..10).collect());
        assert_eq!(dataset.records, (0..10).map(Cow::Owned).collect::<Vec<_>>());
    }

    #[test]
    fn share() {
        let owned = Dataset::new((0..10).collect());
        let first = owned.share();
        assert_eq!(
            first.records.iter().all(|v| matches!(v, Cow::Borrowed(_))),
            true
        );

        let second = first.share();
        assert_eq!(
            second.records.iter().all(|v| matches!(v, Cow::Borrowed(_))),
            true
        );
    }

    #[test]
    fn shuffle() {
        let owned = Dataset::new((0..10).collect());
        let mut shared = owned.share();
        assert_eq!(owned.records.windows(2).all(|w| w[0] <= w[1]), true);
        assert_eq!(shared.records.windows(2).all(|w| w[0] <= w[1]), true);

        shared.shuffle();
        assert_eq!(owned.records.windows(2).all(|w| w[0] <= w[1]), true);
        assert_eq!(shared.records.windows(2).all(|w| w[0] <= w[1]), false);
    }

    #[test]
    fn separation() {
        let first = Dataset::new((0..10).collect());
        let mut second = first.share();

        for cow in &mut second {
            let asd = cow.to_mut();
            *asd += 1;
        }

        second.iter().all(|v| matches!(v, Cow::Owned(_)));
        first.iter().all(|v| matches!(v, Cow::Owned(_)));
    }
}
