use std::{collections::HashMap, hash::Hash};

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

// pub struct StratifiedKFold<'a, 'b, T, K>
// where
//     'b: 'a,
//     T: Clone,
//     K: Hash + Eq,
// {
//     splits: usize,
//     iteration: usize,
//     key_class_map: HashMap<&'b K, usize>,
//     classes: Vec<Vec<&'b Cow<'a, T>>>,
//     windows: Vec<usize>,
// }

// impl<'a, 'b, T, K> StratifiedKFold<'a, 'b, T, K>
// where
//     'b: 'a,
//     T: Clone,
//     K: Hash + Eq,
// {
//     pub fn new<F: Fn(&T) -> &K>(dataset: &'b Dataset<'a, T>, splits: usize, key_fun: F) -> Self {
//         let mut key_class_map = HashMap::new();
//         let mut classes = Vec::new();
//         for cow in dataset {
//             let record = match cow {
//                 Cow::Owned(r) => r,
//                 Cow::Borrowed(r) => *r,
//             };

//             let key = key_fun(record);
//             if let Some(&id) = key_class_map.get(key) {
//                 // The class has already been encountered

//                 let class: &mut Vec<&'b Cow<'a, T>> = &mut classes[id];
//                 class.push(cow);
//             } else {
//                 // The class has never been seen before

//                 key_class_map.insert(key, classes.len());
//                 classes.push(vec![cow]);
//             }
//         }

//         // Compute windows size concerning the proportions
//         let ratio = (1 + (dataset.len() - 1) / splits) as f32 / dataset.len() as f32;
//         let mut windows = Vec::with_capacity(classes.len());
//         for class in &classes {
//             windows.push((class.len() as f32 * ratio).ceil() as usize);
//         }

//         Self {
//             splits,
//             iteration: 0,
//             key_class_map,
//             classes,
//             windows,
//         }
//     }
// }

// impl<'a, 'b: 'a, T: Clone, K: Hash + Eq> Iterator for StratifiedKFold<'a, 'b, T, K> {
//     type Item = (Dataset<'a, T>, Dataset<'a, T>);

//     fn next(&mut self) -> Option<Self::Item> {
//         if self.iteration >= self.splits {
//             return None;
//         }

//         let begin = self.iteration * self.windows[0];
//         let end = std::cmp::min(
//             self.classes[0].len(),
//             (self.iteration + 1) * self.windows[0],
//         );

//         let training_set = self.classes[0][begin..end].iter().collect::<Vec<_>>();
//         let test_set = self.classes[0][..begin]
//             .iter()
//             .chain(self.classes[0][end..].iter())
//             .collect();
//         for i in 1..self.classes.len() {
//             let begin = self.iteration * self.windows[i];
//             let end = std::cmp::min(
//                 self.classes[i].len(),
//                 (self.iteration + 1) * self.windows[i],
//             );
//             training_set.extend(self.classes[i][begin..end].iter());
//         }

//         Some((
//             Dataset {
//                 records: self.dataset[begin..end]
//                     .iter()
//                     .map(|r| Cow::Borrowed(r.borrow()))
//                     .collect(),
//                 generator: self.dataset.generator.clone(),
//             },
//             Dataset {
//                 records: self.dataset[..begin]
//                     .iter()
//                     .chain(self.dataset[end..].iter())
//                     .map(|r| Cow::Borrowed(r.borrow()))
//                     .collect(),
//                 generator: self.dataset.generator.clone(),
//             },
//         ))
//     }
// }

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

            assert_eq!(kfold.next().is_none(), true);
            assert_eq!(kfold.next().is_none(), true);
        }
    }

    // mod stratified_kfold {
    //     use super::*;

    //     #[test]
    //     fn creation() {
    //         let dataset = vec![(1, 1), (1, 2), (2, 1), (2, 2), (3, 1)];
    //         let skf = StratifiedKFold::new(&dataset, 3, |v| &v.0);

    //         assert_eq!(skf.classes[0].len(), 2);
    //         assert_eq!(skf.classes[1].len(), 2);
    //         assert_eq!(skf.classes[2].len(), 1);

    //         assert_eq!(skf.windows[0], 1);
    //         assert_eq!(skf.windows[1], 1);
    //         assert_eq!(skf.windows[2], 1);
    //     }
    // }
}
