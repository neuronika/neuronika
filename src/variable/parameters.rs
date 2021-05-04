use crate::{variable::Tensor, Input, InputBackward};
use itertools::Itertools;
use ndarray::{Dimension, Ix1, Ix2, Ix3, Ix4, Ix5, Ix6, IxDyn};
use std::{
    cell::{Ref, RefMut},
    rc::Rc,
};

use super::node::{Data, Gradient};

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ParamDim Trait ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub trait ParamDim: Dimension + 'static {
    fn insert(item: Param<Self>, dest: &mut Parameters);
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ParamDim Implementations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
impl ParamDim for Ix1 {
    fn insert(item: Param<Self>, dest: &mut Parameters) {
        dest.oned.push(item);
    }
}

impl ParamDim for Ix2 {
    fn insert(item: Param<Self>, dest: &mut Parameters) {
        dest.twod.push(item);
    }
}

impl ParamDim for Ix3 {
    fn insert(item: Param<Self>, dest: &mut Parameters) {
        dest.threed.push(item);
    }
}

impl ParamDim for Ix4 {
    fn insert(item: Param<Self>, dest: &mut Parameters) {
        dest.fourd.push(item);
    }
}

impl ParamDim for Ix5 {
    fn insert(item: Param<Self>, dest: &mut Parameters) {
        dest.fived.push(item);
    }
}

impl ParamDim for Ix6 {
    fn insert(item: Param<Self>, dest: &mut Parameters) {
        dest.sixd.push(item);
    }
}

impl ParamDim for IxDyn {
    fn insert(item: Param<Self>, dest: &mut Parameters) {
        dest.dynd.push(item);
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Param Struct ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub struct Param<D>
where
    D: Dimension,
{
    input: Rc<Input<D>>,
    input_diff: Rc<InputBackward<D>>,
}

impl<D> Param<D>
where
    D: Dimension,
{
    pub(crate) fn new(input: Rc<Input<D>>, input_diff: Rc<InputBackward<D>>) -> Self {
        Self { input, input_diff }
    }

    pub fn data(&self) -> Ref<Tensor<D>> {
        self.input.data()
    }

    pub fn data_mut(&self) -> RefMut<Tensor<D>> {
        self.input.data_mut()
    }

    pub fn grad(&self) -> Ref<Tensor<D>> {
        self.input_diff.gradient()
    }

    pub(crate) fn as_ptr(&self) -> *const InputBackward<D> {
        std::rc::Rc::as_ptr(&self.input_diff)
    }

    pub fn get(self) -> (Rc<Input<D>>, Rc<InputBackward<D>>) {
        (self.input, self.input_diff)
    }
}

impl<D> Clone for Param<D>
where
    D: Dimension,
{
    fn clone(&self) -> Self {
        Self {
            input: self.input.clone(),
            input_diff: self.input_diff.clone(),
        }
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Parameters struct ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#[derive(Clone)]
/// Contains the learnable ancestors of the node.
pub struct Parameters {
    // Contains the one dimensional learnable ancestors
    oned: Vec<Param<Ix1>>,
    // Contains the two dimensional learnable ancestors
    twod: Vec<Param<Ix2>>,
    // Contains the three dimensional learnable ancestors
    threed: Vec<Param<Ix3>>,
    // Contains the four dimensional learnable ancestors
    fourd: Vec<Param<Ix4>>,
    // Contains the five dimensional learnable ancestors
    fived: Vec<Param<Ix5>>,
    // Contains the six dimensional learnable ancestors
    sixd: Vec<Param<Ix6>>,
    // Contains the dynamic dimensional learnable ancestors
    dynd: Vec<Param<IxDyn>>,
}

impl Parameters {
    pub fn new() -> Parameters {
        Parameters {
            oned: Vec::new(),
            twod: Vec::new(),
            threed: Vec::new(),
            fourd: Vec::new(),
            fived: Vec::new(),
            sixd: Vec::new(),
            dynd: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.oned.len()
            + self.twod.len()
            + self.threed.len()
            + self.fourd.len()
            + self.fived.len()
            + self.sixd.len()
            + self.dynd.len()
    }

    pub fn is_empty(&self) -> bool {
        self.oned.is_empty()
            && self.twod.is_empty()
            && self.threed.is_empty()
            && self.fourd.is_empty()
            && self.fived.is_empty()
            && self.sixd.is_empty()
            && self.dynd.is_empty()
    }

    pub(crate) fn get(
        self,
    ) -> (
        Vec<Param<Ix1>>,
        Vec<Param<Ix2>>,
        Vec<Param<Ix3>>,
        Vec<Param<Ix4>>,
        Vec<Param<Ix5>>,
        Vec<Param<Ix6>>,
        Vec<Param<IxDyn>>,
    ) {
        (
            self.oned,
            self.twod,
            self.threed,
            self.fourd,
            self.fived,
            self.sixd,
            self.dynd,
        )
    }
}

impl Default for Parameters {
    fn default() -> Self {
        Self::new()
    }
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~ Functions to keep track of differentiable history ~~~~~~~~~~~~~~~~~~~~~~~
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
pub(crate) fn merge_parameters(lhs_params: Parameters, rhs_params: Parameters) -> Parameters {
    Parameters {
        oned: merge(&lhs_params.oned, &rhs_params.oned),
        twod: merge(&lhs_params.twod, &rhs_params.twod),
        threed: merge(&lhs_params.threed, &rhs_params.threed),
        fourd: merge(&lhs_params.fourd, &rhs_params.fourd),
        fived: merge(&lhs_params.fived, &rhs_params.fived),
        sixd: merge(&lhs_params.sixd, &rhs_params.sixd),
        dynd: merge(&lhs_params.dynd, &rhs_params.dynd),
    }
}

pub(crate) fn merge<D: ParamDim>(lhs_up: &[Param<D>], rhs_up: &[Param<D>]) -> Vec<Param<D>> {
    lhs_up
        .iter()
        .merge_join_by(rhs_up.iter(), |lhs_par, rhs_par| {
            lhs_par.as_ptr().cmp(&rhs_par.as_ptr())
        })
        .map(|choice| match choice {
            itertools::EitherOrBoth::Left(lhs_par) => lhs_par,
            itertools::EitherOrBoth::Right(rhs_par) => rhs_par,
            itertools::EitherOrBoth::Both(lhs_par, _) => lhs_par,
        })
        .cloned()
        .collect()
}
