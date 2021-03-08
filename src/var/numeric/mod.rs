use ndarray::{
    arr1, concatenate, Array, Array2, ArrayView, Axis, Dimension, Ix0, Ix1, Ix2, Ix3, Ix4, Ix5,
    Ix6, IxDyn, RemoveAxis, ShapeBuilder, Zip,
};
use std::fmt;
use std::ops;

// ============================================= Types Relations =============================================

/// Relation among two [Dimension] items.
///
/// This trait is needed in order to use the *broadcasting* semantic of the standard algebraic operations
/// among [tensor]s.
///
/// [tensor]: prova::Tensor
pub trait Res<R>
where
    Self: Dimension,
    R: Dimension,
{
    type Output: Dimension;

    fn max(self, rhs: R) -> Result<Self, R>;
}

/// The [Res] of two [Dimension] items.
///
/// [Res]: crate::Res
/// [Dimension]: ndarray::Dimension
pub type Result<L, R> = <L as Res<R>>::Output;

impl Res<IxDyn> for IxDyn {
    type Output = IxDyn;

    fn max(self, rhs: IxDyn) -> IxDyn {
        if self.ndim() > rhs.ndim() {
            self
        } else {
            rhs
        }
    }
}

/// Automatically implements all the trivial cases for the [Res] relation.
///
/// [Res]: Prova::Res
macro_rules! impl_unary_res {
    ($($dim: ty),+ $(,)?) => {
        $(
            impl Res<IxDyn> for $dim {
                type Output = IxDyn;

                fn max(self, rhs: IxDyn) -> IxDyn {
                    if self.ndim() > rhs.ndim() {
                        self.into_dyn()
                    } else {
                        rhs.into_dyn()
                    }
                }
            }

            impl Res<$dim> for IxDyn {
                type Output = IxDyn;

                fn max(self, rhs: $dim) -> IxDyn {
                    if self.ndim() > rhs.ndim() {
                        self.into_dyn()
                    } else {
                        rhs.into_dyn()
                    }
                }
            }

            impl Res<$dim> for $dim {
                type Output = $dim;

                fn max(self, _: $dim) -> $dim {
                    self
                }
            }
        )*
    };
}

/// Automatically implements all the cases for the [Res] relation accordingly.
///
/// [Res]: Prova::Res
macro_rules! impl_binary_res {
    ($small: ty, $big: ty) => {
        impl Res<$small> for $big {
            type Output = $big;

            fn max(self, _: $small) -> $big { self }
        }

        impl Res<$big> for $small {
            type Output = $big;

            fn max(self, rhs: $big) -> $big { rhs }
        }
    };

    ($(($small: ty, $big: ty)),+ $(,)?) => {
        $(impl_binary_res!{$small, $big })*
    };
}

impl_unary_res!(Ix0, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6);

impl_binary_res!(
    (Ix0, Ix6),
    (Ix0, Ix5),
    (Ix0, Ix4),
    (Ix0, Ix3),
    (Ix0, Ix2),
    (Ix0, Ix1),
    (Ix1, Ix6),
    (Ix1, Ix5),
    (Ix1, Ix4),
    (Ix1, Ix3),
    (Ix1, Ix2),
    (Ix2, Ix6),
    (Ix2, Ix5),
    (Ix2, Ix4),
    (Ix2, Ix3),
    (Ix3, Ix6),
    (Ix3, Ix5),
    (Ix3, Ix4),
    (Ix4, Ix6),
    (Ix4, Ix5),
    (Ix5, Ix6)
);

// =========================================== Operators Overload ===========================================

/// Automatically implements the overload of the `+`, `-`, `*` and `/` binary algebraic operators for
/// [tensor]s.
///
/// [tensor]: crate::Tensor
macro_rules! impl_arithmetic_ops {
    ($(($trait: ident, $fun: ident, $op: tt)),+ $(,)?) => {
        $(
            impl<L, R> ops::$trait<&Tensor<R>> for &Tensor<L>
            where
                L: Dimension + Res<R>,
                R: Dimension,
            {
                type Output = Tensor<Result<L, R>>;

                fn $fun(self, rhs: &Tensor<R>) -> Self::Output {
                    let shape = self.data.raw_dim().max(rhs.data.raw_dim());
                    let mut data = Array::<f32, Result<L, R>>::zeros(shape);
                    Zip::from(&mut data)
                        .and_broadcast(&self.data)
                        .and_broadcast(&rhs.data)
                        .par_apply(|res, l, r| *res = l $op r);

                    Self::Output { data }
                }
            }
        )*
    };
}

impl_arithmetic_ops!((Add, add, +), (Sub, sub, -), (Mul, mul, *), (Div, div, /));

impl<D> ops::Neg for &Tensor<D>
where
    D: Dimension,
{
    type Output = Tensor<D>;

    fn neg(self) -> Self::Output {
        Self::Output { data: -self.data }
    }
}

// =============================================== Tensor Type ===============================================

/// A *n*-dimensional [tensor] of *real* values that support efficient [broadcasting].
///
/// All the standard mathematic binary operators like `+`, `-`, `*` and `/`, exploit **SIMD** computation
/// and are also executed in multiple threads whenever possible.
///
/// [tensor]: https://en.wikipedia.org/wiki/Tensor
/// [broadcasting]: https://numpy.org/devdocs/user/theory.broadcasting.html
#[derive(Debug, PartialEq)]
struct Tensor<D>
where
    D: Dimension,
{
    /// Content of the tensor
    data: Array<f32, D>,
}

impl<D> Tensor<D>
where
    D: Dimension,
{
    pub fn zeros<S>(shape: S) -> Self
    where
        S: ShapeBuilder<Dim = D>,
    {
        Tensor {
            data: Array::zeros(shape),
        }
    }
}

impl<D> fmt::Display for Tensor<D>
where
    D: Dimension,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.data)
    }
}

// Ora seguono una serie di funzioni che servono per fare l'accumulazione dei
//gradienti.

// Questa funzione si occupa del caso in cui dest
// sia un singoletto, ovvero : [el], [[el]], [[[el]]] ecc...
//
// Restituisce true nel caso in cui lo sia in modo tale da segnalare
// che l'operazione si è conlcusa con successo.
pub fn singleton<T: Dimension, D: Dimension>(
    dest: &mut Array<f32, T>,
    src: &Array<f32, D>,
) -> bool {
    // Se dentro dest c'è solo un elemento...
    if dest.len() == 1 {
        let reduced_src = src.sum();
        Zip::from(dest).apply(|dest_el| *dest_el += reduced_src);
        true
    } else {
        false
    }
}

// Il risultato ha sempre la forma di `dest`. Se hanno la dimensione diversa, comanda sempre `dest`.
// Se la dimensione e` uguale, ma la shape cambia, ed  la lunghezza di una di queste sia `

// Questa funzione di occupa del caso in cui la dimensione di dest
// sia maggiore o uguale della dimensione di src, anche lei restituisce
// un booleano per la stessa ragione di prima.
pub fn geq_dim<T: Dimension, D: Dimension>(
    dest: &mut Array<f32, T>,
    src: ArrayView<f32, D>,
) -> bool {
    // Nel caso in cui la dimensione di dest e di src coincidano...
    if dest.ndim() == src.ndim() {
        let mut axis_of_len_one = false;
        // Questo codice gestisce i casi in cui una o più delle assi di
        // dest abbia lunghezza 1, in tal caso bisogna sommare gli elementi
        // della corrispondente asse di src dentro l'unico presente
        // nell'asse di dest
        for i in 0..dest.ndim() {
            let size = dest.len_of(Axis(i));
            if size == 1_usize {
                axis_of_len_one = true;
                dest.lanes_mut(Axis(i))
                    .into_iter()
                    .zip(src.lanes(Axis(i)))
                    .for_each(|(dest_lane, src_lane)| {
                        Zip::from(dest_lane).apply(|dest_view_el| *dest_view_el += src_lane.sum());
                    });
            }
        }
        // Se nessuna delle assi aveva lunghezza uno...
        if !axis_of_len_one {
            Zip::from(dest)
                .and_broadcast(src)
                .apply(|dest_el, src_el| *dest_el += *src_el);
        }
        true
    } else if dest.ndim() > src.ndim() {
        // Se la dimensione di dest è maggiore di quella di src ndarray se la cava da solo.
        Zip::from(dest)
            .and_broadcast(src)
            .apply(|dest_el, src_el| *dest_el += *src_el);
        true
    } else {
        false
    }
}

impl Tensor<Ix1> {
    fn assign<D: Dimension + RemoveAxis>(&mut self, src: &Tensor<D>) {
        // Se dest non è un singoletto controllo le dimensioni di dest e src
        if !singleton(&mut self.data, &src.data) {
            // Se la dimensione di dest non è maggiore o uguale...
            if !geq_dim(&mut self.data, src.data.view()) {
                //... effettuo un'opportuna riduzione
                for lane in src.data.lanes(Axis(src.data.ndim() - 1)) {
                    geq_dim(&mut self.data, lane);
                }
            }
        }
    }

    fn softmax(&self) -> Self {
        let max = self.data.fold(std::f32::MIN, |x, y| x.max(*y));
        let num = self.data.map(|el| (el - max).exp());
        let den = num.sum();
        Self { data: num / den }
    }
}

impl Tensor<Ix2> {
    // Lui si occupa del caso in cui la dim di dest è >= della dim di src, ovviamente
    // ritorna un booleano, è necessario sapere se l'operazione è riuscita
    fn assign<D: Dimension + RemoveAxis>(&mut self, src: &Tensor<D>) {
        if !singleton(&mut self.data, &src.data) {
            if !geq_dim(&mut self.data, src.data.view()) {
                for view in src.data.axis_iter(Axis(0)) {
                    // Questa è l'opportuna riduzione
                    geq_dim(&mut self.data, view);
                }
            }
        }
    }

    fn softmax(&self, axis: usize) -> Self {
        let mut new = <Array2<f32>>::zeros(self.data.raw_dim());
        match axis {
            0 => {
                Zip::from(self.data.gencolumns())
                    .and(new.gencolumns_mut())
                    .apply(|col_op, mut col_new| {
                        let max = col_op.fold(std::f32::MIN, |x, y| x.max(*y));
                        let num = &col_op.map(|el| (el - max).exp());
                        let den = num.sum();
                        col_new.assign(&(num / den))
                    });
            }
            1 => {
                Zip::from(self.data.genrows()).and(new.genrows_mut()).apply(
                    |row_op, mut row_new| {
                        let max = row_op.fold(std::f32::MIN, |x, y| x.max(*y));
                        let num = &row_op.map(|el| (el - max).exp());
                        let den = num.sum();
                        row_new.assign(&(num / den))
                    },
                );
            }
            _ => panic!("error: a two dimensional tensor has only 2 axes."),
        }
        Self { data: new }
    }
}

impl Tensor<Ix3> {
    // Per Tensor 3d abbiamo solo il caso in cui dest dim >= src dim
    // perchè di più di tre dimensioni non ce ne facciamo di nulla.
    fn assign<D: Dimension + RemoveAxis>(&mut self, src: &Tensor<D>) {
        if !singleton(&mut self.data, &src.data) {
            geq_dim(&mut self.data, src.data.view());
        }
    }
}

impl<D> Tensor<D>
where
    D: Dimension + RemoveAxis,
{
    pub fn data(&self) -> &Array<f32, D> {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut Array<f32, D> {
        &mut self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    pub fn t(&self) -> Self {
        Self {
            data: self.data.t().to_owned(),
        }
    }

    pub fn sum(&self) -> Tensor<Ix1> {
        Tensor {
            data: arr1(&[self.data.sum()]),
        }
    }

    pub fn set_zero(&mut self) {
        self.data.map_inplace(|el| *el = 0.0)
    }

    pub fn map<F>(&self, f: F) -> Self
    where
        F: Fn(f32) -> f32,
    {
        Self {
            data: self.data.map(|el| f(*el)),
        }
    }

    pub fn map_inplace<F: Sync + Send>(&mut self, f: F)
    where
        F: Fn(f32) -> f32,
    {
        self.data.map_inplace(|el| *el = f(*el))
    }

    pub fn cat(tensors: &[&Self], axis: usize) -> Self {
        let data: Vec<ArrayView<f32, D>> = tensors.iter().map(|t| t.data().view()).collect();
        Self {
            data: concatenate(Axis(axis), &data).ok().unwrap(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod creation {
        use ndarray::arr0;

        use super::*;

        #[test]
        fn scalar() {
            let node = Tensor::zeros(());
            assert_eq!(node, Tensor { data: arr0(0.) });
        }
    }

    mod manipulation {
        use ndarray::{arr0, arr2};

        use super::*;

        #[test]
        fn broadcast() {
            let node1 = Tensor { data: arr0(10.0) };
            let node2 = Tensor {
                data: arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
            };
            let result = Tensor {
                data: arr2(&[[11.0, 12.0, 13.0], [14.0, 15.0, 16.0], [17.0, 18.0, 19.0]]),
            };
            assert_eq!(&node1 + &node2, result);
            assert_eq!(&node2 + &node1, result);
        }
    }
}
