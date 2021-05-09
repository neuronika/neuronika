use std::{borrow::Borrow, collections::HashMap, hash::Hash};

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
