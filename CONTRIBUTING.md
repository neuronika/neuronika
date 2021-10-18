# Contributing to neuronika

Thank you for your interest in contributing to Neuronika! Before you begin writing code, it is important that you share your intention to contribute with the team, based on the type of contribution:

1. You want to propose a new feature and implement it.
    - Post about your intended feature opening a [discussion](https://github.com/neuronika/neuronika/discussions), and we shall discuss the design and implementation. Once we agree that the plan looks good, we will turn it into an issue and you can go ahead and implement it.
2. You want to implement a feature or bug-fix for an outstanding issue.
    - Search for your issue in the Neuronika [issue list](https://github.com/neuronika/neuronika/issues).
    - Pick an issue and comment that you'd like to work on the feature or bug-fix.
    - If you need more context on a particular issue, please ask and we shall provide.

Once you implement and test your feature or bug-fix, please submit a Pull Request to [neuronika](https://github.com/neuronika/neuronika). We will gladly help you through all the process.

## Codebase structure

* [data](https://github.com/neuronika/neuronika/tree/main/src/data) Data parsing and manipulation utilities.
* [nn](https://github.com/neuronika/neuronika/tree/main/src/nn) Implementation of neural networks' building blocks and loss functions handles.
* [optim](https://github.com/neuronika/neuronika/tree/main/src/optim) Optimization algorithms and learning rate schedulers.
* [variable](https://github.com/neuronika/neuronika/tree/main/src/variable) Core modules of Neuronika. Here are found the implementation of the reverse mode auto-differentiation mechanism and the nodes of the computational graph.
    - [node](https://github.com/neuronika/neuronika/tree/main/src/variable/node) This module contains the implementations of differentiable operators and the chain-rule for auto-differentiation. Files are organized accordingly to the operator's arity.
        - [unary](https://github.com/neuronika/neuronika/tree/main/src/variable/node/unary) Unary operations: tensor reduction functions, reshaping operators and element-wise functions.
        - [binary](https://github.com/neuronika/neuronika/tree/main/src/variable/node/binary) Binary operations: fundamental arithmetic operations, linear algebra, convolutions, losses and binary concatenation and stack operations.
         - [nary](https://github.com/neuronika/neuronika/tree/main/src/variable/node/nary) n-ary functions: tensor concatenation and stack operations with variable length arguments.
    - [var](https://github.com/neuronika/neuronika/blob/main/src/variable/var.rs) Implementation of the handle to a non differentiable computational graph's node and core functionalities for the forward evaluation of the computational graph. It's a core component of Neuronika's API.
    - [vardiff](https://github.com/neuronika/neuronika/blob/main/src/variable/vardiff.rs) Implementation fo the handle to a differentiable computational graph's node and core functionalities for the backward evaluation of the computational graph. It's a core component of Neuronika's API.
