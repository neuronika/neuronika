use super::DataLoader;

mod dataset {

    use ndarray::Array;

    use super::*;

    static DATASET: &str = "\
            0,1,2,3,4,5,6,7,8,9\n\
            9,8,7,6,5,4,3,2,1,0\n\
            0,1,2,3,4,5,6,7,8,9\n\
            9,8,7,6,5,4,3,2,1,0\n\
            0,1,2,3,4,5,6,7,8,9";

    #[test]
    fn from_reader() {
        let dataset = DataLoader::default()
            .without_headers()
            .from_reader(DATASET.as_bytes(), 10);

        assert_eq!(
            dataset.records(),
            Array::from_shape_vec(
                (5, 10),
                vec![
                    0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0.,
                    0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0.,
                    0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
                ]
            )
            .unwrap()
        );
    }

    #[test]
    fn kfold() {
        let dataset = DataLoader::default()
            .without_headers()
            .from_reader(DATASET.as_bytes(), 10);
        let mut kfold = dataset.kfold(2);

        let (train, test) = kfold.next().unwrap();
        assert_eq!(
            train.records(),
            Array::from_shape_vec(
                (2, 10),
                vec![
                    9., 8., 7., 6., 5., 4., 3., 2., 1., 0., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
                ]
            )
            .unwrap()
        );
        assert_eq!(
            test.records(),
            Array::from_shape_vec(
                (3, 10),
                vec![
                    0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0.,
                    0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
                ]
            )
            .unwrap()
        );

        let (train, test) = kfold.next().unwrap();
        assert_eq!(
            train.records(),
            Array::from_shape_vec(
                (3, 10),
                vec![
                    0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0.,
                    0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
                ]
            )
            .unwrap()
        );
        assert_eq!(
            test.records(),
            Array::from_shape_vec(
                (2, 10),
                vec![
                    9., 8., 7., 6., 5., 4., 3., 2., 1., 0., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
                ]
            )
            .unwrap()
        );

        assert!(kfold.next().is_none());
    }

    #[test]
    fn batch() {
        let dataset = DataLoader::default()
            .without_headers()
            .from_reader(DATASET.as_bytes(), 10);

        let mut batch = dataset.batch(3);
        assert_eq!(
            batch.next().unwrap(),
            Array::from_shape_vec(
                (3, 10),
                vec![
                    0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0.,
                    0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
                ]
            )
            .unwrap()
        );

        assert_eq!(
            batch.next().unwrap(),
            Array::from_shape_vec(
                (2, 10),
                vec![
                    9., 8., 7., 6., 5., 4., 3., 2., 1., 0., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
                ]
            )
            .unwrap()
        );

        assert!(batch.next().is_none());
    }

    #[test]
    fn split() {
        let dataset = DataLoader::default()
            .without_headers()
            .from_reader(DATASET.as_bytes(), 10);

        let datasets = dataset.split(&[1, 1, 1, 2]);

        assert_eq!(
            datasets[0].records(),
            Array::from_shape_vec((1, 10), vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,]).unwrap()
        );

        assert_eq!(
            datasets[1].records(),
            Array::from_shape_vec((1, 10), vec![9., 8., 7., 6., 5., 4., 3., 2., 1., 0.]).unwrap()
        );

        assert_eq!(
            datasets[2].records(),
            Array::from_shape_vec((1, 10), vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,]).unwrap()
        );

        assert_eq!(
            datasets[3].records(),
            Array::from_shape_vec(
                (2, 10),
                vec![
                    9., 8., 7., 6., 5., 4., 3., 2., 1., 0., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
                ]
            )
            .unwrap()
        );
    }

    #[test]
    fn shuffle() {
        let mut dataset = DataLoader::default()
            .without_headers()
            .from_reader(DATASET.as_bytes(), 10);

        dataset.shuffle();

        assert_ne!(
            dataset.records(),
            Array::from_shape_vec(
                (5, 10),
                vec![
                    0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0.,
                    0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0.,
                    0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
                ]
            )
            .unwrap()
        );
    }

    #[test]
    fn drop_last() {
        let dataset = DataLoader::default()
            .without_headers()
            .from_reader(DATASET.as_bytes(), 10);

        let mut batch = dataset.batch(3).drop_last();
        assert_eq!(
            batch.next().unwrap(),
            Array::from_shape_vec(
                (3, 10),
                vec![
                    0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0.,
                    0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
                ]
            )
            .unwrap()
        );

        assert!(batch.next().is_none());
        assert!(batch.next().is_none());
    }
}

mod labeled_dataset {

    use ndarray::Array;

    use super::*;

    static DATASET: &str = "\
            0,1,2,1,3,4,5,6,0,7,8,9\n\
            9,8,7,0,6,5,4,3,1,2,1,0\n\
            0,1,2,1,3,4,5,6,0,7,8,9\n\
            9,8,7,0,6,5,4,3,1,2,1,0\n\
            0,1,2,1,3,4,5,6,0,7,8,9";

    #[test]
    fn from_reader() {
        let dataset = DataLoader::default()
            .with_labels(&[3, 8])
            .without_headers()
            .from_reader(DATASET.as_bytes(), 10, 2);

        assert_eq!(
            dataset.records(),
            Array::from_shape_vec(
                (5, 10),
                vec![
                    0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0.,
                    0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0.,
                    0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
                ]
            )
            .unwrap()
        );

        assert_eq!(
            dataset.labels(),
            Array::from_shape_vec((5, 2), vec![1., 0., 0., 1., 1., 0., 0., 1., 1., 0.]).unwrap()
        );
    }

    #[test]
    fn kfold() {
        let dataset = DataLoader::default()
            .with_labels(&[3, 8])
            .without_headers()
            .from_reader(DATASET.as_bytes(), 10, 2);
        let mut kfold = dataset.kfold(2);

        let (train, test) = kfold.next().unwrap();
        assert_eq!(
            train.records(),
            Array::from_shape_vec(
                (2, 10),
                vec![
                    9., 8., 7., 6., 5., 4., 3., 2., 1., 0., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
                ]
            )
            .unwrap()
        );
        assert_eq!(
            train.labels(),
            Array::from_shape_vec((2, 2), vec![0., 1., 1., 0.]).unwrap()
        );
        assert_eq!(
            test.records(),
            Array::from_shape_vec(
                (3, 10),
                vec![
                    0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0.,
                    0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
                ]
            )
            .unwrap()
        );
        assert_eq!(
            test.labels(),
            Array::from_shape_vec((3, 2), vec![1., 0., 0., 1., 1., 0.]).unwrap()
        );

        let (train, test) = kfold.next().unwrap();
        assert_eq!(
            train.records(),
            Array::from_shape_vec(
                (3, 10),
                vec![
                    0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0.,
                    0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
                ]
            )
            .unwrap()
        );
        assert_eq!(
            train.labels(),
            Array::from_shape_vec((3, 2), vec![1., 0., 0., 1., 1., 0.]).unwrap()
        );
        assert_eq!(
            test.records(),
            Array::from_shape_vec(
                (2, 10),
                vec![
                    9., 8., 7., 6., 5., 4., 3., 2., 1., 0., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
                ]
            )
            .unwrap()
        );
        assert_eq!(
            test.labels(),
            Array::from_shape_vec((2, 2), vec![0., 1., 1., 0.]).unwrap()
        );

        assert!(kfold.next().is_none());
    }

    #[test]
    fn batch() {
        let dataset = DataLoader::default()
            .with_labels(&[3, 8])
            .without_headers()
            .from_reader(DATASET.as_bytes(), 10, 2);
        let mut batch = dataset.batch(3);

        let (records, labels) = batch.next().unwrap();
        assert_eq!(
            records,
            Array::from_shape_vec(
                (3, 10),
                vec![
                    0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0.,
                    0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
                ]
            )
            .unwrap()
        );
        assert_eq!(
            labels,
            Array::from_shape_vec((3, 2), vec![1., 0., 0., 1., 1., 0.]).unwrap()
        );

        let (records, labels) = batch.next().unwrap();
        assert_eq!(
            records,
            Array::from_shape_vec(
                (2, 10),
                vec![
                    9., 8., 7., 6., 5., 4., 3., 2., 1., 0., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
                ]
            )
            .unwrap()
        );
        assert_eq!(
            labels,
            Array::from_shape_vec((2, 2), vec![0., 1., 1., 0.]).unwrap()
        );

        assert!(batch.next().is_none());
    }

    #[test]
    fn split() {
        let dataset = DataLoader::default()
            .with_labels(&[3, 8])
            .without_headers()
            .from_reader(DATASET.as_bytes(), 10, 2);

        let datasets = dataset.split(&[1, 1, 1, 2]);
        assert_eq!(
            datasets[0].records(),
            Array::from_shape_vec((1, 10), vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,]).unwrap()
        );
        assert_eq!(
            datasets[0].labels(),
            Array::from_shape_vec((1, 2), vec![1., 0.]).unwrap()
        );

        assert_eq!(
            datasets[1].records(),
            Array::from_shape_vec((1, 10), vec![9., 8., 7., 6., 5., 4., 3., 2., 1., 0.,]).unwrap()
        );
        assert_eq!(
            datasets[1].labels(),
            Array::from_shape_vec((1, 2), vec![0., 1.]).unwrap()
        );

        assert_eq!(
            datasets[2].records(),
            Array::from_shape_vec((1, 10), vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,]).unwrap()
        );
        assert_eq!(
            datasets[2].labels(),
            Array::from_shape_vec((1, 2), vec![1., 0.]).unwrap()
        );

        assert_eq!(
            datasets[3].records(),
            Array::from_shape_vec(
                (2, 10),
                vec![
                    9., 8., 7., 6., 5., 4., 3., 2., 1., 0., 0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
                ]
            )
            .unwrap()
        );
        assert_eq!(
            datasets[3].labels(),
            Array::from_shape_vec((2, 2), vec![0., 1., 1., 0.]).unwrap()
        );
    }

    #[test]
    fn shuffle() {
        let mut dataset = DataLoader::default()
            .with_labels(&[3, 8])
            .without_headers()
            .from_reader(DATASET.as_bytes(), 10, 2);
        dataset.shuffle();

        assert_ne!(
            dataset.records(),
            Array::from_shape_vec(
                (5, 10),
                vec![
                    0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0.,
                    0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0.,
                    0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
                ]
            )
            .unwrap()
        );
    }

    #[test]
    fn drop_last() {
        let dataset = DataLoader::default()
            .with_labels(&[3, 8])
            .without_headers()
            .from_reader(DATASET.as_bytes(), 10, 2);
        let mut batch = dataset.batch(3).drop_last();

        let (records, labels) = batch.next().unwrap();
        assert_eq!(
            records,
            Array::from_shape_vec(
                (3, 10),
                vec![
                    0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0.,
                    0., 1., 2., 3., 4., 5., 6., 7., 8., 9.,
                ]
            )
            .unwrap()
        );
        assert_eq!(
            labels,
            Array::from_shape_vec((3, 2), vec![1., 0., 0., 1., 1., 0.]).unwrap()
        );

        assert!(batch.next().is_none());
        assert!(batch.next().is_none());
    }
}
