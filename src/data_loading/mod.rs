use crate::variable::Tensor;
use csv::Reader;
use ndarray::IntoDimension;
use std::{error::Error, io::Read};

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
        inputs.extend(&content[..split]);
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

#[cfg(test)]
mod tests {
    use super::*;

    static CSV_CONTENT: &str = "\
        1,2,3,4,5,6,7,8,9,10,1\n\
        10,9,8,7,6,5,4,3,2,1,0\n\
        1,2,3,4,5,6,7,8,9,10,1\n\
        10,9,8,7,6,5,4,3,2,1,0\n\
        1,2,3,4,5,6,7,8,9,10,1";

    #[test]
    fn read_from_csv() {
        let mut reader = csv::ReaderBuilder::new()
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
