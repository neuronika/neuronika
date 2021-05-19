use crate::variable::{Tensor, TensorView};
use csv::{ReaderBuilder, StringRecord};
use ndarray::{Axis, Dimension, IntoDimension, RemoveAxis};
use serde::de::DeserializeOwned;
use std::{error::Error, fs::File, io::Read};

pub struct Dataset<D> {
    records: Tensor<D>,
}

impl<D: Dimension> Dataset<D> {
    fn new(records: Tensor<D>) -> Self {
        Self { records }
    }

    pub fn records(&self) -> &Tensor<D> {
        &self.records
    }
}

impl<D: RemoveAxis> Dataset<D> {
    fn kfold(&self, k: usize) -> KFold<D> {
        KFold::new(self.records.view(), k)
    }
}

pub struct DatasetBuilder {
    // All configuration settings
}

impl DatasetBuilder {
    pub fn new() -> Self {
        Self {}
    }

    pub fn with_labels(self, labels: &[usize]) -> LabeledDatasetBuilder {
        LabeledDatasetBuilder::new(labels)
    }

    pub fn load_from_csv<S>(self, src: &str, shape: S) -> Result<Dataset<S::Dim>, Box<dyn Error>>
    where
        S: IntoDimension,
    {
        self.load_from_reader(File::open(src)?, shape)
    }

    pub fn load_from_reader<R, S>(self, src: R, shape: S) -> Result<Dataset<S::Dim>, Box<dyn Error>>
    where
        R: Read,
        S: IntoDimension,
    {
        let mut reader = ReaderBuilder::new().has_headers(false).from_reader(src);
        let mut records = Vec::new();
        for record in reader.deserialize() {
            let record: Vec<f32> = record?;
            records.extend(record);
        }

        Ok(Dataset::new(Tensor::from_shape_vec(shape, records)?))
    }
}

impl Default for DatasetBuilder {
    fn default() -> Self {
        Self::new()
    }
}

pub struct LabeledDataset<D1, D2> {
    inputs: Tensor<D1>,
    labels: Tensor<D2>,
}

impl<D1: Dimension, D2: Dimension> LabeledDataset<D1, D2> {
    fn new(inputs: Tensor<D1>, labels: Tensor<D2>) -> Self {
        Self { inputs, labels }
    }

    pub fn inputs(&self) -> &Tensor<D1> {
        &self.inputs
    }

    pub fn labels(&self) -> &Tensor<D2> {
        &self.labels
    }
}

impl<D1: RemoveAxis, D2: RemoveAxis> LabeledDataset<D1, D2> {
    pub fn kfold(&self, k: usize) -> LabeledKFold<D1, D2> {
        LabeledKFold::new(self.inputs.view(), self.labels.view(), k)
    }
}

pub struct LabeledDatasetBuilder {
    labels: Vec<usize>,
}

impl LabeledDatasetBuilder {
    fn deserialize_record<T, U>(&self, record: StringRecord) -> Result<(T, U), Box<dyn Error>>
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

    fn new(labels: &[usize]) -> Self {
        let mut labels = labels.to_vec();
        labels.sort_unstable();

        assert_eq!(
            labels.windows(2).all(|w| w[0] != w[1]),
            true,
            "duplicated labels"
        );

        Self { labels }
    }

    pub fn load_from_csv<S1, S2>(
        self,
        src: &str,
        sh1: S1,
        sh2: S2,
    ) -> Result<LabeledDataset<S1::Dim, S2::Dim>, Box<dyn Error>>
    where
        S1: IntoDimension,
        S2: IntoDimension,
    {
        self.load_from_reader(File::open(src)?, sh1, sh2)
    }

    pub fn load_from_reader<S1, S2, R>(
        self,
        src: R,
        sh1: S1,
        sh2: S2,
    ) -> Result<LabeledDataset<S1::Dim, S2::Dim>, Box<dyn Error>>
    where
        S1: IntoDimension,
        S2: IntoDimension,
        R: Read,
    {
        let mut reader = ReaderBuilder::new().has_headers(false).from_reader(src);

        let mut inputs = Vec::new();
        let mut labels = Vec::new();
        for record in reader.records() {
            let (input, label): (Vec<f32>, Vec<f32>) = self.deserialize_record(record?)?;
            inputs.extend(input);
            labels.extend(label);
        }

        Ok(LabeledDataset::new(
            Tensor::from_shape_vec(sh1, inputs)?,
            Tensor::from_shape_vec(sh2, labels)?,
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
        let axis_len = source.len_of(Axis(0));
        if axis_len == 0 {
            panic!("cannot handle 0 length axis");
        }

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
    inputs: SetKFold<'a, D1>,
    labels: SetKFold<'a, D2>,
    iteration: usize,
    k: usize,
}

impl<'a, D1, D2> LabeledKFold<'a, D1, D2>
where
    D1: RemoveAxis,
    D2: RemoveAxis,
{
    pub fn new(inputs: TensorView<'a, D1>, labels: TensorView<'a, D2>, k: usize) -> Self {
        assert_eq!(inputs.len_of(Axis(0)), labels.len_of(Axis(0)));

        Self {
            inputs: SetKFold::new(inputs, k),
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

        let training_part = self.inputs.compute_fold(self.iteration);
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
        fn load_from_reader() {
            let dataset = DatasetBuilder::new()
                .load_from_reader(DATASET.as_bytes(), (5, 10))
                .unwrap();

            assert_eq!(
                dataset.records,
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
                .load_from_reader(DATASET.as_bytes(), (5, 10))
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
        fn load_from_reader() {
            let dataset = DatasetBuilder::new()
                .with_labels(&[3, 8])
                .load_from_reader(DATASET.as_bytes(), (5, 10), (5, 2))
                .unwrap();

            assert_eq!(
                dataset.inputs(),
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
                .load_from_reader(DATASET.as_bytes(), (5, 10), (5, 2))
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
