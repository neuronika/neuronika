# Changelog 

## Unreleased

* Add dynamical typing for Var and VarDiff [#94](https://github.com/neuronika/neuronika/pull/94).
  - The Forward trait has been split in the Forward and Cache traits.
  - Forward and Backward bounds have been removed from the respective methods of Var and VarDiff.
  - The new method `.into_dyn()` available for both Var and VarDiff allows for dynamical typing.

* Remove GradientOverwite trait as it is redundant [#92](https://github.com/neuronika/neuronika/pull/92).
  - The `GradientOverwrite` trait is equal to `Gradient: Overwrite`.
  - Implementors of the `Gradient` trait are now required to implement the `Overwrite` trait.

* Remove Forward trait from `Input` node [#91](https://github.com/neuronika/neuronika/pull/91).

* Change scalar operands dimension to ndarray's `Ix0` [#90](https://github.com/neuronika/neuronika/pull/90).

* Fix co-broadcasting and reshape bias in convolutions operations [#87](https://github.com/neuronika/neuronika/pull/87).

* Remove `Debug` and `Display` bounds from convolution trait bounds [#85](https://github.com/neuronika/neuronika/pull/85).

* Expose the `MatVecMul` trait.
