<p align="center">
  <img width="750" src="./misc/neuronika_logo.png" alt="Neuronika Logo" />
</p>

<hr/>

<p align="center">
<a href="https://app.circleci.com/pipelines/github/neuronika">
  <img src="https://circleci.com/gh/neuronika/neuronika.svg?style=svg&circle-token=a4dc29e597fde3872a02c582dc42c058f41f7869"/>
</a>

<a href="https://codecov.io/gh/neuronika/neuronika">
  <img src="https://codecov.io/gh/neuronika/neuronika/branch/main/graph/badge.svg?token=H7J7TF511B"/>
</a>

<a href="https://opensource.org/licenses/MPL-2.0">
  <img src="https://img.shields.io/badge/License-MPL%202.0-ff69b4.svg"/>
</a>

<a href="https://docs.rs/neuronika">
  <img src="https://img.shields.io/badge/API-reference-brightgreen"/>
</a>
</p>

Neuronika is a machine learning framework written in pure Rust, built with a focus on ease of
use, fast prototyping and performance.

## Dynamic neural networks and auto-differentiation

At the core of Neuronika lies a mechanism called reverse-mode automatic differentiation, that allows you
to define dynamically changing neural networks with very low effort and no overhead by using a lean, fully imperative and define by run API.

![](./misc/neuronika_ad.gif)

## The power of Rust

The Rust language allows for an intuitive, light and easy to use interface while achieving incredible performance.
There's no need for a FFI, everything happens in front of your eyes.

## Crate Feature Flags

The following crate feature flags are available. They configure the [`ndarray`](https://github.com/rust-ndarray/ndarray) backend.

* `serialize` 
  * Enables serialization support for [`serde`](https://github.com/serde-rs/serde) 1.x.

* `blas`
  * Enables transparent BLAS support for matrix multiplication. Uses `blas-src` for pluggable backend, which needs to be configured separately. See [`here`](https://github.com/rust-ndarray/ndarray#how-to-enable-blas-integration) for more informations.

* `matrixmultiply-threading`
  * Enables the `threading` feature in the [`matrixmultiply`](https://github.com/bluss/matrixmultiply) package.

## Contributing

We appreciate and welcome all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion.

If you plan to contribute new features, utility functions, or extensions to the framework, please first open a discussion and discuss the feature with us. Sending a PR without discussion might end up resulting in a rejected PR because we might be taking the project in a different direction than you might be aware of.

To learn more about making a contribution to Neuronika, please see our [contribution page](CONTRIBUTING.md).

## Project Status

Neuronika is very young and rapidly evolving, we are continuously developing the project and breaking changes are expected during transitions from version to version. We adopt the newest stable Rust's features if we need them.