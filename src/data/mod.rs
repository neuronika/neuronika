use crate::variable::{Tensor, TensorView};
use csv::{ReaderBuilder, StringRecord};
use ndarray::{Axis, Dimension, IntoDimension, RemoveAxis};
use serde::de::DeserializeOwned;
use std::{error::Error, fs::File, io::Read, result};

/// Dataset's result type.
pub type Result<T> = result::Result<T, Box<dyn Error>>;

/// Shorthand for `Dimension::Larger`.
pub type LargerDim<T> = <T as Dimension>::Larger;

/// Computes the correct shape for the stacked records of a dataset.
fn stacked_shape<D: Dimension>(rows: usize, shape: D) -> D::Larger {
    let mut new_shape = D::Larger::zeros(shape.ndim() + 1);
    new_shape[0] = rows;
    new_shape.slice_mut()[1..].clone_from_slice(shape.slice());

    new_shape
}

/// A collection of uniquely owned **unlabeled** records.
///
/// `Dataset` is the basic container type required to interact
/// with any model created with `neuronika`. It can be created easily
/// by the [`DatasetBuilder`] struct, for example loading the content of a
/// `csv` file with the method [`.from_csv()`].
///
/// The `Dataset<D>` struct is generic upon [dimensionality] `D` and is organized
/// as a tensor in which the first axis has the same length as the number of records,
/// while the other `D - 1` represent the shape of each one.
///
/// [`.from_csv()`]: crate::data::DatasetBuilder::from_csv
/// [dimensionality]: ndarray::Dimension
pub struct Dataset<D> {
    records: Tensor<D>,
}

impl<D: RemoveAxis> Dataset<D> {
    /// Creates a new `Dataset` from a [`Tensor`].
    ///
    /// ## Arguments
    /// - `records`: Records to store into the dataset
    fn new(records: Tensor<D>) -> Self {
        Self { records }
    }

    /// Provides non-mutable access to the stored records.
    ///
    /// ## Examples
    /// ```rust
    /// use neuronika::data::{Dataset, DatasetBuilder};
    /// use ndarray::Array;
    ///
    /// let csv_content = "\
    ///    0,1,2\n\
    ///    3,4,5\n\
    ///    6,7,8";
    ///
    /// let dataset = DatasetBuilder::new()
    ///     .without_headers()
    ///     .from_reader(csv_content.as_bytes(), 3)
    ///     .unwrap();
    /// assert_eq!(dataset.records(), Array::from(vec![[0., 1., 2.],
    ///                                                [3., 4., 5.],
    ///                                                [6., 7., 8.]]));
    /// ```
    pub fn records(&self) -> &Tensor<D> {
        &self.records
    }

    /// Provides a [`KFold`] iterator.
    ///
    /// The dataset is split **without shuffling** into `k` consecutive folds.
    /// Each fold is used exactly once as a validation set, while the remaining `k - 1`
    /// form the training set.
    ///
    /// ## Arguments
    /// - `k`: Number of folds to perform
    ///
    /// ## Panics
    /// This function panics if `k < 2`.
    ///
    /// ## Examples
    /// ```rust
    /// use neuronika::data::{Dataset, DatasetBuilder};
    /// use ndarray::Array;
    ///
    /// let csv_content = "\
    ///    0,1,2\n\
    ///    3,4,5\n\
    ///    6,7,8";
    ///
    /// let dataset = DatasetBuilder::new()
    ///     .without_headers()
    ///     .from_reader(csv_content.as_bytes(), 3)
    ///     .unwrap();
    ///
    /// let mut kfold = dataset.kfold(2);
    ///
    /// let (training_set, test_set) = kfold.next().unwrap();
    /// assert_eq!(training_set, Array::from(vec![[6., 7., 8.]]));
    /// assert_eq!(test_set, Array::from(vec![[0., 1., 2.],
    ///                                       [3., 4., 5.]]));
    ///
    /// let (training_set, test_set) = kfold.next().unwrap();
    /// assert_eq!(training_set, Array::from(vec![[0., 1., 2.],
    ///                                           [3., 4., 5.]]));
    /// assert_eq!(test_set, Array::from(vec![[6., 7., 8.]]));
    ///
    /// assert_eq!(kfold.next(), None);
    /// ```
    pub fn kfold(&self, k: usize) -> KFold<D> {
        KFold::new(self.records.view(), k)
    }
}

/// [`Dataset`]'s builder.
///
/// The base configuration considers the first row of the source as an header
/// and expects a `,` as field delimiter.
pub struct DatasetBuilder {
    r_builder: ReaderBuilder,
}

impl DatasetBuilder {
    /// Creates a new builder for configuring a new [`Dataset`].
    pub fn new() -> Self {
        Self {
            r_builder: ReaderBuilder::new(),
        }
    }

    /// Configures the indexes on the records in which the labels are located.
    ///
    /// ## Arguments
    /// - `labels`: Indexes of the labels for each record
    ///
    /// ## Panics
    /// This function panics if `labels.is_empty() == true` or if `labels` contains duplicates.
    pub fn with_labels(self, labels: &[usize]) -> LabeledDatasetBuilder {
        LabeledDatasetBuilder::new(self, labels)
    }

    /// Configures the source from which to load the `Dataset` as without an header row.
    pub fn without_headers(&mut self) -> &mut Self {
        self.r_builder.has_headers(false);

        self
    }

    /// Configures the delimiter of each record's field.
    pub fn with_delimiter(&mut self, delimiter: u8) -> &mut Self {
        self.r_builder.delimiter(delimiter);

        self
    }

    /// Creates the configured `Dataset` loading its content from a `csv` file.
    ///
    /// ## Arguments
    /// - `src`: String representing the path of the source file
    /// - `shape`: Record's shape
    ///
    /// ## Errors
    /// All the errors returned by [`File::open`] and [`csv::Error`].
    ///
    /// ## Panics
    /// This function panics if `shape` would generate an empty record.
    pub fn from_csv<S>(
        &mut self,
        src: &str,
        shape: S,
    ) -> Result<Dataset<<S::Dim as Dimension>::Larger>>
    where
        S: IntoDimension,
    {
        self.from_reader_fn(File::open(src)?, shape, |r| r)
    }

    /// Creates the configured `Dataset` loading its content from a reader.
    ///
    /// ## Arguments
    /// - `src`: Reader from which to load the records
    /// - `shape`: Record's shape
    ///
    /// ## Errors
    /// In the event of a deserialization error, a [`csv::Error`] is returned.
    ///
    /// ## Panics
    /// This function panics if `shape` would generate an empty record.
    pub fn from_reader<R, S>(
        &mut self,
        src: R,
        shape: S,
    ) -> Result<Dataset<<S::Dim as Dimension>::Larger>>
    where
        R: Read,
        S: IntoDimension,
    {
        self.from_reader_fn(src, shape, |r| r)
    }

    /// Creates a `Dataset` by applying a function to the result of the deserialization
    /// of each line of a `csv` file
    ///
    /// ## Arguments
    /// - `src`: String representing the path of the source file
    /// - `shape`: Record's shape
    /// - `f`: Closure to apply to each deserialized object
    ///
    /// ## Errors
    /// All the errors returned by [`File::open`] and [`csv::Error`].
    ///
    /// ## Panics
    /// This function panics if `shape` would generate an empty record.
    pub fn from_csv_fn<S, T, F>(
        &mut self,
        src: &str,
        shape: S,
        f: F,
    ) -> Result<Dataset<<S::Dim as Dimension>::Larger>>
    where
        S: IntoDimension,
        T: DeserializeOwned,
        F: Fn(T) -> Vec<f32>,
    {
        self.from_reader_fn(File::open(src)?, shape, f)
    }

    /// Creates a `Dataset` by applying a function to the result of the deserialization
    /// of the content of a reader.
    ///
    /// ## Arguments
    /// - `src`: Reader from which to load the records
    /// - `shape`: Record's shape
    /// - `f`: Closure to apply to each deserialized object
    ///
    /// ## Errors
    /// All the errors returned by [`File::open`] and [`csv::Error`].
    ///
    /// ## Panics
    /// This function panics if `shape` would generate an empty record.
    pub fn from_reader_fn<R, S, T, F>(
        &mut self,
        src: R,
        shape: S,
        f: F,
    ) -> Result<Dataset<<S::Dim as Dimension>::Larger>>
    where
        R: Read,
        S: IntoDimension,
        T: DeserializeOwned,
        F: Fn(T) -> Vec<f32>,
    {
        let shape = shape.into_dimension();
        if shape.size() == 0 {
            panic!("cannot handle empty records")
        }

        let mut records = Vec::new();
        let mut rows = 0;
        for record in self.r_builder.from_reader(src).deserialize() {
            let record = f(record?);
            records.extend(record);
            rows += 1;
        }

        Ok(Dataset::new(Tensor::from_shape_vec(
            stacked_shape(rows, shape),
            records,
        )?))
    }
}

impl Default for DatasetBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// A collection of uniquely owned **labeled** records.
///
/// `LabeledDataset` can be created easily by the [`LabeledDatasetBuilder`] struct,
/// for example loading the content of a `csv` file with the method [`.from_csv()`].
///
/// The `LabeledDataset<D1, D2>` struct is generic upon both the [dimensionality] of
/// the records `D1` and labels `D2` and it's organized as a pair of tensors in which the first
/// axis has the same length as the number of records, while the other represent the shape
/// of each one.
///
/// [`.from_csv()`]: crate::data::LabeledDatasetBuilder::from_csv
/// [dimensionality]: ndarray::Dimension
pub struct LabeledDataset<D1, D2> {
    records: Tensor<D1>,
    labels: Tensor<D2>,
}

impl<D1: RemoveAxis, D2: RemoveAxis> LabeledDataset<D1, D2> {
    /// Creates a new `LabeledDataset` from a pair of [`Tensor`]s.
    ///
    /// ## Arguments
    /// - `records`: Records to be stored
    /// - `labels`: Records' labels to be stored
    fn new(records: Tensor<D1>, labels: Tensor<D2>) -> Self {
        Self { records, labels }
    }

    /// Provides non-mutable access to the stored records.
    ///
    /// ## Examples
    /// ```rust
    /// use neuronika::data::{Dataset, DatasetBuilder};
    /// use ndarray::Array;
    ///
    /// let csv_content = "\
    ///    0,1,2,0\n\
    ///    3,4,5,1\n\
    ///    6,7,8,0";
    ///
    /// let dataset = DatasetBuilder::new()
    ///     .with_labels(&[3])
    ///     .without_headers()
    ///     .from_reader(csv_content.as_bytes(), 3, 1)
    ///     .unwrap();
    /// assert_eq!(dataset.records(), Array::from(vec![
    ///     [0., 1., 2.],
    ///     [3., 4., 5.],
    ///     [6., 7., 8.]
    /// ]));
    /// ```
    pub fn records(&self) -> &Tensor<D1> {
        &self.records
    }

    /// Provides non-mutable access to the stored labels.
    ///
    /// ## Examples
    /// ```rust
    /// use neuronika::data::{Dataset, DatasetBuilder};
    /// use ndarray::Array;
    ///
    /// let csv_content = "\
    ///    0,1,2,0\n\
    ///    3,4,5,1\n\
    ///    6,7,8,0";
    ///
    /// let dataset = DatasetBuilder::new()
    ///     .with_labels(&[3])
    ///     .without_headers()
    ///     .from_reader(csv_content.as_bytes(), 3, 1)
    ///     .unwrap();
    /// assert_eq!(dataset.labels(), Array::from(vec![[0.], [1.], [0.]]));
    /// ```
    pub fn labels(&self) -> &Tensor<D2> {
        &self.labels
    }

    /// Provides a [`LabeledKFold`] iterator.
    ///
    /// The dataset is split **without shuffling** into `k` consecutive folds.
    /// Each fold is used exactly once as a validation set, while the remaining `k - 1`
    /// form the training set.
    ///
    /// ## Arguments
    /// - `k`: Number of fold to perform
    ///
    /// ## Panics
    /// This function panics if `k < 2`.
    pub fn kfold(&self, k: usize) -> LabeledKFold<D1, D2> {
        LabeledKFold::new(self.records.view(), self.labels.view(), k)
    }
}

/// [`LabeledDataset`]'s builder.
///
/// The base configuration considers the first row of the source as an header
/// and expects a `,` as field delimiter.
pub struct LabeledDatasetBuilder {
    r_builder: ReaderBuilder,
    labels: Vec<usize>,
}

impl LabeledDatasetBuilder {
    fn deserialize_record<T, U>(&self, record: StringRecord) -> Result<(T, U)>
    where
        T: DeserializeOwned,
        U: DeserializeOwned,
    {
        let mut input = StringRecord::new();
        let mut label = StringRecord::new();
        for (id, value) in record.iter().enumerate() {
            match self.labels.binary_search(&id) {
                Ok(_) => label.push_field(value),
                Err(_) => input.push_field(value),
            }
        }

        Ok((input.deserialize(None)?, label.deserialize(None)?))
    }

    fn new(builder: DatasetBuilder, labels: &[usize]) -> Self {
        if labels.is_empty() {
            panic!("labels not provided");
        }

        let mut labels = labels.to_vec();
        labels.sort_unstable();
        if labels.windows(2).any(|w| w[0] == w[1]) {
            panic!("duplicated labels");
        }

        Self {
            r_builder: builder.r_builder,
            labels,
        }
    }

    /// Configures the source from which to load the `LabeledDataset` as without an header row.
    pub fn without_headers(&mut self) -> &mut Self {
        self.r_builder.has_headers(false);

        self
    }

    /// Configures the delimiter of each record's field.
    pub fn with_delimiter(&mut self, delimiter: u8) -> &mut Self {
        self.r_builder.delimiter(delimiter);

        self
    }

    /// Creates the configured `LabeledDataset` loading its content from a `csv` file.
    ///
    /// ## Arguments
    /// - `src`: String representing the path of the source file
    /// - `sh1`: Record's shape
    /// - `sh2`: Label's shape
    ///
    /// ## Errors
    /// All the errors returned by [`File::open`] and [`csv::Error`].
    ///
    /// ## Panics
    /// This function panics if `sh1` or `sh2` would generate an empty tensor.
    pub fn from_csv<S1, S2>(
        &mut self,
        src: &str,
        sh1: S1,
        sh2: S2,
    ) -> Result<LabeledDataset<LargerDim<S1::Dim>, LargerDim<S2::Dim>>>
    where
        S1: IntoDimension,
        S2: IntoDimension,
    {
        self.from_reader_fn(File::open(src)?, sh1, sh2, |r| r)
    }

    /// Creates the configured `LabeledDataset` loading its content from a reader.
    ///
    /// ## Arguments
    /// - `src`: Reader from which to load the data
    /// - `sh1`: Record's shape
    /// - `sh2`: Label's shape
    ///
    /// ## Errors
    /// All the errors returned by [`File::open`] and [`csv::Error`].
    ///
    /// ## Panics
    /// This function panics if `sh1` or `sh2` would generate an empty tensor.
    pub fn from_reader<R, S1, S2>(
        &mut self,
        src: R,
        sh1: S1,
        sh2: S2,
    ) -> Result<LabeledDataset<LargerDim<S1::Dim>, LargerDim<S2::Dim>>>
    where
        R: Read,
        S1: IntoDimension,
        S2: IntoDimension,
    {
        self.from_reader_fn(src, sh1, sh2, |r| r)
    }

    /// Creates the configured `LabeledDataset` loading its content from a `csv` file.
    ///
    /// ## Arguments
    /// - `src`: String representing the path of the source file
    /// - `sh1`: Record's shape
    /// - `sh2`: Label's shape
    /// - `f`: Closure to apply to each deserialized object
    ///
    /// ## Errors
    /// All the errors returned by [`File::open`] and [`csv::Error`].
    ///
    /// ## Panics
    /// This function panics if `sh1` or `sh2` would generate an empty tensor.
    pub fn from_csv_fn<S1, S2, T, U, F>(
        &mut self,
        src: &str,
        sh1: S1,
        sh2: S2,
        f: F,
    ) -> Result<LabeledDataset<LargerDim<S1::Dim>, LargerDim<S2::Dim>>>
    where
        S1: IntoDimension,
        S2: IntoDimension,
        T: DeserializeOwned,
        U: DeserializeOwned,
        F: Fn((T, U)) -> (Vec<f32>, Vec<f32>),
    {
        self.from_reader_fn(File::open(src)?, sh1, sh2, f)
    }

    /// Creates a `Dataset` by applying a function to the result of the deserialization
    /// of the content of a reader.
    ///
    /// ## Arguments
    /// - `src`: Reader from which to load the data
    /// - `sh1`: Record's shape
    /// - `sh2`: Label's shape
    /// - `f`: Closure to apply to each deserialized object
    ///
    /// ## Errors
    /// All the errors returned by [`File::open`] and [`csv::Error`].
    ///
    /// ## Panics
    /// This function panics if `sh1` or `sh2` would generate an empty tensor.
    pub fn from_reader_fn<R, S1, S2, T, U, F>(
        &mut self,
        src: R,
        sh1: S1,
        sh2: S2,
        f: F,
    ) -> Result<LabeledDataset<LargerDim<S1::Dim>, LargerDim<S2::Dim>>>
    where
        R: Read,
        S1: IntoDimension,
        S2: IntoDimension,
        T: DeserializeOwned,
        U: DeserializeOwned,
        F: Fn((T, U)) -> (Vec<f32>, Vec<f32>),
    {
        let sh1 = sh1.into_dimension();
        let sh2 = sh2.into_dimension();
        if sh1.size() == 0 || sh2.size() == 0 {
            panic!("cannot handle empty records")
        }

        let mut records = Vec::new();
        let mut labels = Vec::new();
        let mut rows = 0;
        for record in self.r_builder.from_reader(src).records() {
            let (record, label) = f(self.deserialize_record(record?)?);
            records.extend(record);
            labels.extend(label);
            rows += 1;
        }

        Ok(LabeledDataset::new(
            Tensor::from_shape_vec(stacked_shape(rows, sh1), records)?,
            Tensor::from_shape_vec(stacked_shape(rows, sh2), labels)?,
        ))
    }
}

struct SetKFold<'a, D> {
    source: TensorView<'a, D>,
    step: usize,
    axis_len: usize,
}

impl<'a, D: RemoveAxis> SetKFold<'a, D> {
    pub fn new(source: TensorView<'a, D>, k: usize) -> Self {
        if k < 2 {
            panic!("meaningless fold number");
        }

        let axis_len = source.len_of(Axis(0));
        debug_assert_ne!(axis_len, 0, "no record provided");

        Self {
            source,
            step: 1 + (axis_len - 1) / k,
            axis_len,
        }
    }

    pub fn compute_fold(&mut self, i: usize) -> (Tensor<D>, Tensor<D>) {
        let start = self.step * i;
        let stop = self.axis_len.min(start + self.step);

        let train_ids: Vec<usize> = (0..start).chain(stop..self.axis_len).collect();
        let test_ids: Vec<usize> = (start..stop).collect();

        (
            self.source.select(Axis(0), &train_ids),
            self.source.select(Axis(0), &test_ids),
        )
    }
}

pub struct LabeledKFold<'a, D1, D2> {
    records: SetKFold<'a, D1>,
    labels: SetKFold<'a, D2>,
    iteration: usize,
    k: usize,
}

impl<'a, D1, D2> LabeledKFold<'a, D1, D2>
where
    D1: RemoveAxis,
    D2: RemoveAxis,
{
    pub fn new(records: TensorView<'a, D1>, labels: TensorView<'a, D2>, k: usize) -> Self {
        assert_eq!(records.len_of(Axis(0)), labels.len_of(Axis(0)));

        Self {
            records: SetKFold::new(records, k),
            labels: SetKFold::new(labels, k),
            iteration: 0,
            k,
        }
    }
}

impl<'a, D1, D2> Iterator for LabeledKFold<'a, D1, D2>
where
    D1: RemoveAxis,
    D2: RemoveAxis,
{
    type Item = ((Tensor<D1>, Tensor<D1>), (Tensor<D2>, Tensor<D2>));

    fn next(&mut self) -> Option<Self::Item> {
        if self.iteration >= self.k {
            return None;
        }

        let training_part = self.records.compute_fold(self.iteration);
        let test_part = self.labels.compute_fold(self.iteration);
        self.iteration += 1;

        Some((training_part, test_part))
    }
}

pub struct KFold<'a, D> {
    records: SetKFold<'a, D>,
    iteration: usize,
    k: usize,
}

impl<'a, D: RemoveAxis> KFold<'a, D> {
    pub fn new(records: TensorView<'a, D>, k: usize) -> Self {
        Self {
            records: SetKFold::new(records, k),
            iteration: 0,
            k,
        }
    }
}

impl<'a, D: RemoveAxis> Iterator for KFold<'a, D> {
    type Item = (Tensor<D>, Tensor<D>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.iteration >= self.k {
            return None;
        }

        let (records_in, records_out) = self.records.compute_fold(self.iteration);
        self.iteration += 1;

        Some((records_in, records_out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod dataset {
        use super::*;
        use crate::variable::Tensor;

        static DATASET: &str = "\
            0,1,2,3,4,5,6,7,8,9\n\
            9,8,7,6,5,4,3,2,1,0\n\
            0,1,2,3,4,5,6,7,8,9\n\
            9,8,7,6,5,4,3,2,1,0\n\
            0,1,2,3,4,5,6,7,8,9";

        #[test]
        fn from_reader() {
            let dataset = DatasetBuilder::new()
                .without_headers()
                .from_reader(DATASET.as_bytes(), 10)
                .unwrap();

            assert_eq!(
                dataset.records(),
                Tensor::from_shape_vec(
                    (5, 10),
                    vec![
                        0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 9., 8., 7., 6., 5., 4., 3., 2., 1.,
                        0., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 9., 8., 7., 6., 5., 4., 3., 2.,
                        1., 0., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
                    ]
                )
                .unwrap()
            );
        }

        #[test]
        fn kfold() {
            let dataset = DatasetBuilder::new()
                .without_headers()
                .from_reader(DATASET.as_bytes(), 10)
                .unwrap();
            let mut kfold = dataset.kfold(2);

            let (train, test) = kfold.next().unwrap();
            assert_eq!(
                train,
                Tensor::from_shape_vec(
                    (2, 10),
                    vec![
                        9., 8., 7., 6., 5., 4., 3., 2., 1., 0., 0., 1., 2., 3., 4., 5., 6., 7., 8.,
                        9.,
                    ]
                )
                .unwrap()
            );
            assert_eq!(
                test,
                Tensor::from_shape_vec(
                    (3, 10),
                    vec![
                        0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 9., 8., 7., 6., 5., 4., 3., 2., 1.,
                        0., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
                    ]
                )
                .unwrap()
            );

            let (train, test) = kfold.next().unwrap();
            assert_eq!(
                train,
                Tensor::from_shape_vec(
                    (3, 10),
                    vec![
                        0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 9., 8., 7., 6., 5., 4., 3., 2., 1.,
                        0., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
                    ]
                )
                .unwrap()
            );
            assert_eq!(
                test,
                Tensor::from_shape_vec(
                    (2, 10),
                    vec![
                        9., 8., 7., 6., 5., 4., 3., 2., 1., 0., 0., 1., 2., 3., 4., 5., 6., 7., 8.,
                        9.,
                    ]
                )
                .unwrap()
            );

            assert_eq!(kfold.next(), None);
        }
    }

    mod labeled_dataset {
        use super::*;
        use crate::variable::Tensor;

        static DATASET: &str = "\
            0,1,2,1,3,4,5,6,0,7,8,9\n\
            9,8,7,0,6,5,4,3,1,2,1,0\n\
            0,1,2,1,3,4,5,6,0,7,8,9\n\
            9,8,7,0,6,5,4,3,1,2,1,0\n\
            0,1,2,1,3,4,5,6,0,7,8,9";

        #[test]
        fn from_reader() {
            let dataset = DatasetBuilder::new()
                .with_labels(&[3, 8])
                .without_headers()
                .from_reader(DATASET.as_bytes(), 10, 2)
                .unwrap();

            assert_eq!(
                dataset.records(),
                Tensor::from_shape_vec(
                    (5, 10),
                    vec![
                        0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 9., 8., 7., 6., 5., 4., 3., 2., 1.,
                        0., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 9., 8., 7., 6., 5., 4., 3., 2.,
                        1., 0., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
                    ]
                )
                .unwrap()
            );

            assert_eq!(
                dataset.labels(),
                Tensor::from_shape_vec((5, 2), vec![1., 0., 0., 1., 1., 0., 0., 1., 1., 0.])
                    .unwrap()
            );
        }

        #[test]
        fn kfold() {
            let dataset = DatasetBuilder::new()
                .with_labels(&[3, 8])
                .without_headers()
                .from_reader(DATASET.as_bytes(), 10, 2)
                .unwrap();
            let mut kfold = dataset.kfold(2);

            let ((train_in, train_out), (test_in, test_out)) = kfold.next().unwrap();
            assert_eq!(
                train_in,
                Tensor::from_shape_vec(
                    (2, 10),
                    vec![
                        9., 8., 7., 6., 5., 4., 3., 2., 1., 0., 0., 1., 2., 3., 4., 5., 6., 7., 8.,
                        9.,
                    ]
                )
                .unwrap()
            );
            assert_eq!(
                train_out,
                Tensor::from_shape_vec(
                    (3, 10),
                    vec![
                        0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 9., 8., 7., 6., 5., 4., 3., 2., 1.,
                        0., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
                    ]
                )
                .unwrap()
            );
            assert_eq!(
                test_in,
                Tensor::from_shape_vec((2, 2), vec![0., 1., 1., 0.]).unwrap()
            );
            assert_eq!(
                test_out,
                Tensor::from_shape_vec((3, 2), vec![1., 0., 0., 1., 1., 0.]).unwrap()
            );

            let ((train_in, train_out), (test_in, test_out)) = kfold.next().unwrap();
            assert_eq!(
                train_in,
                Tensor::from_shape_vec(
                    (3, 10),
                    vec![
                        0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 9., 8., 7., 6., 5., 4., 3., 2., 1.,
                        0., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
                    ]
                )
                .unwrap()
            );
            assert_eq!(
                train_out,
                Tensor::from_shape_vec(
                    (2, 10),
                    vec![
                        9., 8., 7., 6., 5., 4., 3., 2., 1., 0., 0., 1., 2., 3., 4., 5., 6., 7., 8.,
                        9.,
                    ]
                )
                .unwrap()
            );
            assert_eq!(
                test_in,
                Tensor::from_shape_vec((3, 2), vec![1., 0., 0., 1., 1., 0.]).unwrap()
            );
            assert_eq!(
                test_out,
                Tensor::from_shape_vec((2, 2), vec![0., 1., 1., 0.]).unwrap()
            );

            assert_eq!(kfold.next(), None);
        }
    }
}
