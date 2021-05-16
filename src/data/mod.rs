use crate::variable::Tensor;
use csv::{DeserializeRecordsIter, Reader, ReaderBuilder};
use ndarray::{Array, IntoDimension};
use serde::de::DeserializeOwned;
use std::{
    borrow::Borrow, collections::HashMap, error::Error, fmt::Debug, fs::File, hash::Hash, io::Read,
    str::FromStr,
};
/// Basic Dataset struct.
///
/// Please note that if the data and the labels to be parsed differ in the type `AnnotatedDataset`
/// must be used instead, as the `Dataset` struct assumes the data and the labels to be typed
/// **homogeneously**.
pub struct Dataset<T> {
    data: Vec<T>,
    num_records: usize,
}

impl<'a, T: DeserializeOwned + Copy> Dataset<T> {
    /// Creates an empty Dataset.
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            num_records: 0,
        }
    }

    /// Separates the data from the labels. Consumes `self` and creates an `AnnotatedDataset`
    /// instance.
    ///
    /// # Arguments
    ///
    /// * `labels_colums` - the columns to be considered labels
    pub fn with_annotations(self, labels_columns: &[usize]) -> AnnotatedDataset<T, T> {
        let row_len = self.num_records / self.data.len();
        let (enumerated_data, enumerated_labels): (Vec<(usize, &T)>, Vec<(usize, &T)>) =
            self.data.iter().enumerate().partition(|(i, _)| {
                labels_columns
                    .iter()
                    .any(|col| *col == i.rem_euclid(row_len))
            });

        let (data, labels, num_records) = {
            (
                enumerated_data.iter().map(|(_, &el)| el).collect(),
                enumerated_labels.iter().map(|(_, &el)| el).collect(),
                self.num_records,
            )
        };

        AnnotatedDataset {
            data,
            labels,
            num_records,
        }
    }

    /// Loads the data from a **.csv** file into `self`.
    ///
    /// # Arguments
    ///
    /// * `file_path` - the path to the *csv* file
    /// * `has_headers` - whether the *csv* file has headers that must be skipped
    pub fn load_csv(&mut self, file_path: &str, has_headers: bool) {
        let file = match File::open(file_path) {
            Ok(file) => file,
            Err(e) => panic!("error: neuronika couldn't open {}, {}", file_path, e),
        };

        let mut reader = ReaderBuilder::new()
            .has_headers(has_headers)
            .from_reader(file);

        self.from_reader(&mut reader);
    }

    /// Loads the data from a **reader** into `self`.
    ///
    /// # Arguments
    ///
    /// * `reader` - the reader to read the data from
    pub fn from_reader<R: Read>(&mut self, reader: &mut Reader<R>) {
        let results: DeserializeRecordsIter<'_, R, Vec<T>> = reader.deserialize();

        for result in results {
            let content = result.unwrap();
            self.num_records += 1;
            self.data.extend(content);
        }
    }

    /// Consumes `self` and returns the data as a **ndarray's array**.
    pub fn into_ndarray<Sh: IntoDimension>(self, shape: Sh) -> Array<T, Sh::Dim> {
        let v = self.data;
        Array::from_shape_vec(shape, v).unwrap()
    }
}

/// An annotated Dataset.
pub struct AnnotatedDataset<T, U> {
    data: Vec<T>,
    labels: Vec<U>,
    num_records: usize,
}

impl<T, U> AnnotatedDataset<T, U>
where
    T: DeserializeOwned + FromStr,
    T::Err: Debug,
    U: DeserializeOwned + FromStr,
    U::Err: Debug,
{
    /// Creates an empty AnnotatedDataset.
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            labels: Vec::new(),
            num_records: 0,
        }
    }
    /// Consumes `self` and returns the data and the labels as a **ndarray's array** tuple.
    pub fn into_ndarray<ShD: IntoDimension, ShL: IntoDimension>(
        self,
        data_shape: ShD,
        labels_shape: ShL,
    ) -> (Array<T, ShD::Dim>, Array<U, ShL::Dim>) {
        let (data, labels) = (self.data, self.labels);
        (
            Array::from_shape_vec(data_shape, data).unwrap(),
            Array::from_shape_vec(labels_shape, labels).unwrap(),
        )
    }

    /// Loads the data from a **.csv** file into `self`.
    /// # Arguments
    ///
    /// * `file_path` - the path to the *csv* file
    /// * `has_headers` - whether the *csv* file has headers that must be skipped
    /// * `label_columns` - the columns to be considered labels
    pub fn load_csv(&mut self, file_path: &str, has_headers: bool, labels_columns: &[usize]) {
        let file = match File::open(file_path) {
            Ok(file) => file,
            Err(e) => panic!("error: neuronika couldn't open {}, {}", file_path, e),
        };

        let mut reader = ReaderBuilder::new()
            .has_headers(has_headers)
            .from_reader(file);

        self.from_reader(&mut reader, labels_columns);
    }

    /// Loads the data from a **reader** into `self`.
    ///
    /// # Arguments
    ///
    /// * `reader` - the reader to read the data from
    /// * `label_columns` - the columns to be considered labels
    pub fn from_reader<R: Read>(&mut self, reader: &mut Reader<R>, labels_columns: &[usize]) {
        for result in reader.records() {
            let record = match result {
                Ok(res) => res,
                Err(e) => panic!("error: while parsing records, {}", e),
            };
            for i in 0..record.len() {
                if labels_columns.iter().any(|col| *col == i) {
                    self.labels.push(record.get(i).unwrap().parse().unwrap())
                } else {
                    self.data.push(record.get(i).unwrap().parse().unwrap())
                }
            }
            self.num_records += 1;
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fn read_from_csv<R: Read>(
    reader: &mut Reader<R>,
    split: Option<usize>,
) -> Result<(Vec<f32>, Vec<f32>), Box<dyn Error>> {
    let mut results = reader.deserialize();

    let mut content: Vec<f32> = results.next().unwrap()?;
    let split = match split {
        Some(id) => id,
        None => content.len(),
    };

    let mut inputs = content[..split].to_vec();
    let mut targets = content[split..].to_vec();
    for result in results {
        content = result?;
        inputs.extend(&content[..split]); // what if the label columns are not contiguous?
        targets.extend(&content[split..]);
    }

    Ok((inputs, targets))
}

fn from_csv<R, Sh>(reader: &mut Reader<R>, shape: Sh) -> Result<Tensor<Sh::Dim>, Box<dyn Error>>
where
    R: Read,
    Sh: IntoDimension,
{
    let (inputs, _) = read_from_csv(reader, None)?;

    Ok(Tensor::from_shape_vec(shape, inputs)?)
}

fn from_csv_with_targets<R, S1, S2>(
    reader: &mut Reader<R>,
    input_shape: S1,
    target_shape: S2,
    split: usize,
) -> Result<(Tensor<S1::Dim>, Tensor<S2::Dim>), Box<dyn Error>>
where
    R: Read,
    S1: IntoDimension,
    S2: IntoDimension,
{
    let (inputs, targets) = read_from_csv(reader, Some(split))?;

    Ok((
        Tensor::from_shape_vec(input_shape, inputs)?,
        Tensor::from_shape_vec(target_shape, targets)?,
    ))
}
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pub struct KFold<'a, T> {
    dataset: &'a [T],
    splits: usize,
    size: usize,
    iteration: usize,
}

impl<'a, T> KFold<'a, T> {
    pub fn new(dataset: &'a [T], splits: usize) -> Self {
        Self {
            dataset,
            splits,
            size: 1 + (dataset.len() - 1) / splits,
            iteration: 0,
        }
    }
}

impl<'a, T> Iterator for KFold<'a, T> {
    type Item = (Vec<&'a T>, Vec<&'a T>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.iteration >= self.splits {
            return None;
        }

        // Compute starting point of the `training` dataset
        let begin = self.iteration * self.size;

        // Update the iterator state now, to avoid the same computation below
        self.iteration += 1;

        // Compute ending point of the `training` dataset
        let end = std::cmp::min(self.dataset.len(), self.iteration * self.size);

        Some((
            self.dataset[begin..end].iter().collect(),
            self.dataset[..begin]
                .iter()
                .chain(self.dataset[end..].iter())
                .collect(),
        ))
    }
}

pub struct StratifiedKFold<'a, T> {
    dataset_len: usize,
    set_len: usize,
    left: usize,
    splits: usize,
    iterations: usize,
    classes: Vec<Vec<&'a T>>,
    read_infos: Vec<(usize, usize)>,
}

impl<'a, T> StratifiedKFold<'a, T> {
    pub fn new<U>(dataset: &'a [T], labels: &'a [U], splits: usize) -> Self
    where
        U: Hash + Eq + Borrow<U>,
    {
        assert_eq!(dataset.len(), labels.len());

        let mut label_key_map = HashMap::new();
        let mut classes: Vec<Vec<&'a T>> = Vec::new();
        for (record, label) in dataset.iter().zip(labels) {
            match label_key_map.get(label) {
                None => {
                    label_key_map.insert(label, classes.len());
                    classes.push(vec![record]);
                }
                Some(&id) => classes.get_mut::<usize>(id).unwrap().push(record),
            }
        }

        let dataset_len = dataset.len();
        let set_len = dataset_len / splits;
        let ratio = set_len as f32 / dataset_len as f32;
        let mut read_infos = Vec::with_capacity(classes.len());
        for class in &classes {
            read_infos.push((0, (class.len() as f32 * ratio).ceil() as usize));
        }

        Self {
            dataset_len,
            set_len,
            left: dataset_len % splits,
            splits,
            iterations: 0,
            classes,
            read_infos,
        }
    }
}

impl<'a, T> Iterator for StratifiedKFold<'a, T> {
    type Item = (Vec<&'a T>, Vec<&'a T>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.iterations >= self.splits {
            return None;
        }

        self.iterations += 1;
        let mut remaining = if self.left > 0 {
            self.left -= 1;
            self.set_len + 1
        } else {
            self.set_len
        };

        let mut training_set = Vec::with_capacity(self.dataset_len - remaining);
        let mut test_set = Vec::with_capacity(remaining);
        let mut iter = self.classes.iter().zip(self.read_infos.iter_mut());
        while remaining > 0 {
            let (class, (begin, window_size)) = iter.next().unwrap();
            if *begin < class.len() {
                // Compute the amount of records to read and the ending point
                // in order to save a bit of time
                let to_read = remaining.min(*window_size);
                let end = *begin + to_read;

                // Merge the sets in the appropriate way
                test_set.extend(&class[*begin..end]);
                training_set.extend(class[..*begin].iter().chain(&class[end..]));

                // Update the state of the computation
                *begin = end;
                remaining -= to_read;
            } else {
                // We may encounter classes that have been read completely,
                // so we must ensure that those won't be missed
                training_set.extend(&class[..]);
            }
        }
        for (class, _) in iter {
            // If we already put all the required records into `test_set`,
            // but some classes are left, we put them into `training_set`
            training_set.extend(&class[..]);
        }

        Some((training_set, test_set))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod kfold {
        use super::*;

        #[test]
        fn creation() {
            let dataset: Vec<_> = (0..10).collect();

            let kfold = KFold::new(&dataset, 3);
            assert_eq!(kfold.splits, 3);
            assert_eq!(kfold.size, 4);
            assert_eq!(kfold.iteration, 0);
        }

        #[test]
        fn state_transition() {
            let dataset: Vec<_> = (0..10).collect();
            let mut kfold = KFold::new(&dataset, 3);

            let (training, test) = kfold.next().unwrap();
            assert_eq!(training, vec![&0, &1, &2, &3]);
            assert_eq!(test, vec![&4, &5, &6, &7, &8, &9]);
            assert_eq!(kfold.splits, 3);
            assert_eq!(kfold.size, 4);
            assert_eq!(kfold.iteration, 1);

            let (training, test) = kfold.next().unwrap();
            assert_eq!(training, vec![&4, &5, &6, &7]);
            assert_eq!(test, vec![&0, &1, &2, &3, &8, &9]);
            assert_eq!(kfold.splits, 3);
            assert_eq!(kfold.size, 4);
            assert_eq!(kfold.iteration, 2);

            let (training, test) = kfold.next().unwrap();
            assert_eq!(training, vec![&8, &9]);
            assert_eq!(test, vec![&0, &1, &2, &3, &4, &5, &6, &7]);
            assert_eq!(kfold.splits, 3);
            assert_eq!(kfold.size, 4);
            assert_eq!(kfold.iteration, 3);

            assert_eq!(kfold.next(), None);
            assert_eq!(kfold.next(), None);
        }
    }

    mod stratified_kfold {
        use super::*;

        #[test]
        fn creation_imperfect_slice() {
            let dataset: Vec<_> = (0..11).collect();
            let labels = vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2];
            let skf = StratifiedKFold::new(&dataset, &labels, 3);

            assert_eq!(skf.dataset_len, 11);
            assert_eq!(skf.set_len, 3);
            assert_eq!(skf.left, 2);
            assert_eq!(skf.splits, 3);
            assert_eq!(skf.iterations, 0);
            assert_eq!(skf.classes[0].len(), 6);
            assert_eq!(skf.classes[1].len(), 3);
            assert_eq!(skf.classes[2].len(), 2);
            assert_eq!(skf.read_infos[0], (0, 2));
            assert_eq!(skf.read_infos[1], (0, 1));
            assert_eq!(skf.read_infos[2], (0, 1));
        }

        #[test]
        fn creation_perfect_slice() {
            let dataset: Vec<_> = (0..9).collect();
            let labels = vec![0, 0, 0, 0, 0, 0, 1, 1, 2];
            let skf = StratifiedKFold::new(&dataset, &labels, 3);

            assert_eq!(skf.dataset_len, 9);
            assert_eq!(skf.set_len, 3);
            assert_eq!(skf.left, 0);
            assert_eq!(skf.splits, 3);
            assert_eq!(skf.iterations, 0);
            assert_eq!(skf.classes[0].len(), 6);
            assert_eq!(skf.classes[1].len(), 2);
            assert_eq!(skf.classes[2].len(), 1);
            assert_eq!(skf.read_infos[0], (0, 2));
            assert_eq!(skf.read_infos[1], (0, 1));
            assert_eq!(skf.read_infos[2], (0, 1));
        }

        #[test]
        fn state_transition_imperfect_slice() {
            let dataset: Vec<_> = (0..11).collect();
            let labels = vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2];
            let mut skf = StratifiedKFold::new(&dataset, &labels, 3);

            skf.next().unwrap();
            assert_eq!(skf.dataset_len, 11);
            assert_eq!(skf.set_len, 3);
            assert_eq!(skf.left, 1);
            assert_eq!(skf.splits, 3);
            assert_eq!(skf.iterations, 1);
            assert_eq!(skf.classes[0].len(), 6);
            assert_eq!(skf.classes[1].len(), 3);
            assert_eq!(skf.classes[2].len(), 2);
            assert_eq!(skf.read_infos[0], (2, 2));
            assert_eq!(skf.read_infos[1], (1, 1));
            assert_eq!(skf.read_infos[2], (1, 1));

            skf.next().unwrap();
            assert_eq!(skf.dataset_len, 11);
            assert_eq!(skf.set_len, 3);
            assert_eq!(skf.left, 0);
            assert_eq!(skf.splits, 3);
            assert_eq!(skf.iterations, 2);
            assert_eq!(skf.classes[0].len(), 6);
            assert_eq!(skf.classes[1].len(), 3);
            assert_eq!(skf.classes[2].len(), 2);
            assert_eq!(skf.read_infos[0], (4, 2));
            assert_eq!(skf.read_infos[1], (2, 1));
            assert_eq!(skf.read_infos[2], (2, 1));

            skf.next().unwrap();
            assert_eq!(skf.dataset_len, 11);
            assert_eq!(skf.set_len, 3);
            assert_eq!(skf.left, 0);
            assert_eq!(skf.splits, 3);
            assert_eq!(skf.iterations, 3);
            assert_eq!(skf.classes[0].len(), 6);
            assert_eq!(skf.classes[1].len(), 3);
            assert_eq!(skf.classes[2].len(), 2);
            assert_eq!(skf.read_infos[0], (6, 2));
            assert_eq!(skf.read_infos[1], (3, 1));
            assert_eq!(skf.read_infos[2], (2, 1));

            assert_eq!(skf.next(), None);
            assert_eq!(skf.next(), None);
        }

        #[test]
        fn state_transition_perfect_slice() {
            let dataset: Vec<_> = (0..9).collect();
            let labels = vec![0, 0, 0, 0, 0, 0, 1, 1, 2];
            let mut skf = StratifiedKFold::new(&dataset, &labels, 3);

            skf.next().unwrap();
            assert_eq!(skf.dataset_len, 9);
            assert_eq!(skf.set_len, 3);
            assert_eq!(skf.left, 0);
            assert_eq!(skf.splits, 3);
            assert_eq!(skf.iterations, 1);
            assert_eq!(skf.classes[0].len(), 6);
            assert_eq!(skf.classes[1].len(), 2);
            assert_eq!(skf.classes[2].len(), 1);
            assert_eq!(skf.read_infos[0], (2, 2));
            assert_eq!(skf.read_infos[1], (1, 1));
            assert_eq!(skf.read_infos[2], (0, 1));

            skf.next().unwrap();
            assert_eq!(skf.dataset_len, 9);
            assert_eq!(skf.set_len, 3);
            assert_eq!(skf.left, 0);
            assert_eq!(skf.splits, 3);
            assert_eq!(skf.iterations, 2);
            assert_eq!(skf.classes[0].len(), 6);
            assert_eq!(skf.classes[1].len(), 2);
            assert_eq!(skf.classes[2].len(), 1);
            assert_eq!(skf.read_infos[0], (4, 2));
            assert_eq!(skf.read_infos[1], (2, 1));
            assert_eq!(skf.read_infos[2], (0, 1));

            skf.next().unwrap();
            assert_eq!(skf.dataset_len, 9);
            assert_eq!(skf.set_len, 3);
            assert_eq!(skf.left, 0);
            assert_eq!(skf.splits, 3);
            assert_eq!(skf.iterations, 3);
            assert_eq!(skf.classes[0].len(), 6);
            assert_eq!(skf.classes[1].len(), 2);
            assert_eq!(skf.classes[2].len(), 1);
            assert_eq!(skf.read_infos[0], (6, 2));
            assert_eq!(skf.read_infos[1], (2, 1));
            assert_eq!(skf.read_infos[2], (1, 1));

            assert_eq!(skf.next(), None);
            assert_eq!(skf.next(), None);
        }

        #[test]
        fn imperfect_slice() {
            let dataset: Vec<_> = (0..11).collect();
            let labels = vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2];
            let mut skf = StratifiedKFold::new(&dataset, &labels, 3);

            let (training, test) = skf.next().unwrap();
            assert_eq!(training, vec![&2, &3, &4, &5, &7, &8, &10]);
            assert_eq!(test, vec![&0, &1, &6, &9]);

            let (training, test) = skf.next().unwrap();
            assert_eq!(training, vec![&0, &1, &4, &5, &6, &8, &9]);
            assert_eq!(test, vec![&2, &3, &7, &10]);

            let (training, test) = skf.next().unwrap();
            assert_eq!(training, vec![&0, &1, &2, &3, &6, &7, &9, &10]);
            assert_eq!(test, vec![&4, &5, &8]);

            assert_eq!(skf.next(), None);
            assert_eq!(skf.next(), None);
        }

        #[test]
        fn perfect_slice() {
            let dataset: Vec<_> = (0..9).collect();
            let labels = vec![0, 0, 0, 0, 0, 0, 1, 1, 2];
            let mut skf = StratifiedKFold::new(&dataset, &labels, 3);

            let (training, test) = skf.next().unwrap();
            assert_eq!(training, vec![&2, &3, &4, &5, &7, &8]);
            assert_eq!(test, vec![&0, &1, &6]);

            let (training, test) = skf.next().unwrap();
            assert_eq!(training, vec![&0, &1, &4, &5, &6, &8]);
            assert_eq!(test, vec![&2, &3, &7]);

            let (training, test) = skf.next().unwrap();
            assert_eq!(training, vec![&0, &1, &2, &3, &6, &7]);
            assert_eq!(test, vec![&4, &5, &8]);

            assert_eq!(skf.next(), None);
            assert_eq!(skf.next(), None);
        }
    }
}

#[cfg(test)]
mod data_tests {
    use super::*;

    static CSV_CONTENT: &str = "\
        1,2,3,4,5,6,7,8,9,10,1\n\
        10,9,8,7,6,5,4,3,2,1,0\n\
        1,2,3,4,5,6,7,8,9,10,1\n\
        10,9,8,7,6,5,4,3,2,1,0\n\
        1,2,3,4,5,6,7,8,9,10,1";

    #[test]
    fn read_from_csv() {
        let mut reader = ReaderBuilder::new()
            .has_headers(false)
            .delimiter(b',')
            .from_reader(CSV_CONTENT.as_bytes());

        assert_eq!(
            from_csv(&mut reader, (5, 11)).unwrap(),
            Tensor::from_shape_vec(
                (5, 11),
                vec![
                    1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 1., 10., 9., 8., 7., 6., 5., 4., 3.,
                    2., 1., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 1., 10., 9., 8., 7., 6.,
                    5., 4., 3., 2., 1., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 1.,
                ]
            )
            .unwrap()
        );
    }

    #[test]
    fn read_from_csv_with_target() {
        let mut reader = csv::ReaderBuilder::new()
            .has_headers(false)
            .delimiter(b',')
            .from_reader(CSV_CONTENT.as_bytes());

        let (inputs, targets) = from_csv_with_targets(&mut reader, (5, 10), 5, 10).unwrap();
        assert_eq!(
            inputs,
            Tensor::from_shape_vec(
                (5, 10),
                vec![
                    1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 10., 9., 8., 7., 6., 5., 4., 3., 2.,
                    1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 10., 9., 8., 7., 6., 5., 4., 3.,
                    2., 1., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.,
                ]
            )
            .unwrap()
        );

        assert_eq!(
            targets,
            Tensor::from_shape_vec(5, vec![1., 0., 1., 0., 1.,]).unwrap()
        )
    }
}
