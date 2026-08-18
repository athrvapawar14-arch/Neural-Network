# Neural Network From Scratch

A neural network framework being built from scratch in Python to understand the foundations of neural networks, numerical computing, and AI engineering.

The goal of this project is **not** to recreate an existing framework like PyTorch. Instead, the project is being developed as a learning exercise to understand what happens underneath high-level machine-learning libraries while practicing clean, modular Python design.

---

## 🎯 Project Goals

The long-term goal is to build a small neural network capable of performing training and inference without relying on high-level machine-learning frameworks.

The project is being used to develop understanding of:

* Python software engineering
* Object-oriented design
* NumPy and numerical computing
* Linear algebra
* Neural network mathematics
* Forward propagation
* Backpropagation
* Gradient descent
* Machine-learning fundamentals
* Reusable AI components

---

## 🏗️ Current Architecture

The project is being developed incrementally, with individual mathematical and neural-network components kept in separate modules.

```text
neural-network/
│
├── core/
│   ├── vector.py
│   └── matrix.py
│
├── functions/
│   ├── activation.py
│   └── loss.py
│
├── layers/
│   └── dense.py
│
└── ...
```

The structure will continue to evolve as additional components are introduced.

---

# ✅ Current Features

## Vector

A reusable `Vector` abstraction built on top of NumPy.

Currently supports:

* Variable-length vector construction
* Floating-point storage
* Vector addition
* Vector subtraction
* Scalar multiplication
* Dot product
* Cross product
* Magnitude
* Unit vector
* Indexing
* Assignment
* Iteration
* Equality comparison
* Vector shape information
* Operator overloading

---

## Matrix

A reusable `Matrix` abstraction built on top of NumPy.

Currently supports:

* Matrix construction from nested data
* Vector input handling
* Floating-point storage
* Matrix shape information
* Matrix addition
* Matrix subtraction
* Scalar multiplication
* Matrix multiplication
* Matrix × Vector operations
* Transpose
* Determinant
* Inverse
* Indexing
* Assignment
* Iteration
* Equality comparison

---

## Activation Functions

An abstract activation-function interface is used to provide a common API for different activation functions.

Currently implemented:

### ReLU

Used primarily for hidden layers because of its simplicity and fast computation.

### Sigmoid

Produces values between `0` and `1` and is useful for binary classification.

### Tanh

Produces values between `-1` and `1` and provides a zero-centered activation.

### Leaky ReLU

A variation of ReLU that allows a small gradient for negative inputs, helping address the dying-ReLU problem.

### Softmax

Converts a set of values into a probability distribution whose values sum to `1`.

It is intended primarily for multiclass classification outputs.

---

## Loss Functions

A common abstract `Loss` interface has been created with:

* `forward()`
* `backward()`

Currently implemented:

### Mean Squared Error (MSE)

Used to measure the squared difference between predicted and target values.

### Binary Cross-Entropy (BCE)

Designed for binary classification and commonly used together with Sigmoid.

### Softmax + Categorical Cross-Entropy

A combined implementation for multiclass classification.

The Softmax and Categorical Cross-Entropy components are intentionally coupled here because their gradients simplify significantly when used together.

---

## Dense Layer

An initial implementation of a fully connected (`Dense`) layer has been started.

Current implementation includes:

* Weight initialization
* Bias initialization
* Forward propagation
* Gradient storage
* Input caching
* Initial backward-pass structure

The Dense layer is currently **experimental/incomplete** and will be reviewed and refined as the project progresses.

---

# 🚧 Current Limitations

This project is still under active development and should **not** be considered a production-ready neural-network framework.

Current limitations include:

* Heavy reliance on NumPy
* Limited input validation
* API design is still evolving
* Some mathematical operations have not been independently implemented
* No complete training pipeline
* Backpropagation is not yet fully integrated
* No optimizer system
* No complete dataset abstraction
* No batching/training infrastructure
* No production-grade numerical stability guarantees
* Automated testing is still limited
* Documentation is still being developed

The project intentionally prioritizes **understanding and experimentation** over performance and production readiness.

---

# 🧠 Design Philosophy

The project follows a few principles:

### Learn the mathematics

Whenever possible, mathematical operations are understood before being hidden behind library functions.

### Build reusable components

Each component should have a clear responsibility and be usable by other parts of the system.

### Keep components modular

Vectors, matrices, activation functions, losses, and layers are separated into different modules rather than being placed into one large file.

### Use NumPy intelligently

NumPy is used for numerical operations and vectorization rather than manually implementing every low-level numerical operation.

### Design APIs deliberately

Operator overloading, properties, abstract classes, and reusable interfaces are used to make mathematical objects behave naturally in Python.

---

# 🗺️ Roadmap

## Phase 1 — Python Engineering

* [x] Modules
* [x] Packages
* [x] Imports
* [x] Object-oriented design
* [x] Abstract base classes
* [ ] Type hints throughout the project
* [ ] Improved documentation
* [ ] Logging
* [ ] Automated testing

## Phase 2 — Numerical Foundations

* [x] Vector
* [x] Matrix
* [x] NumPy integration
* [x] Matrix multiplication
* [x] Basic linear algebra operations
* [ ] Improved validation
* [ ] More robust numerical handling

## Phase 3 — Activation & Loss Functions

* [x] ReLU
* [x] Sigmoid
* [x] Tanh
* [x] Leaky ReLU
* [x] Softmax
* [x] MSE
* [x] Binary Cross-Entropy
* [x] Softmax + Categorical Cross-Entropy

## Phase 4 — Neural Network Components

* [x] Initial Dense Layer
* [ ] Review and refine Dense Layer
* [ ] Forward propagation
* [ ] Parameter management
* [ ] Proper weight initialization
* [ ] Backpropagation

## Phase 5 — Training

* [ ] Gradient descent
* [ ] Optimizers
* [ ] Learning rate
* [ ] Training loop
* [ ] Epochs
* [ ] Mini-batches
* [ ] Model evaluation

## Phase 6 — Final Neural Network

* [ ] Build a complete multi-layer network
* [ ] Train on a small dataset
* [ ] Evaluate predictions
* [ ] Experiment with different architectures
* [ ] Compare different activation functions
* [ ] Document the complete learning process

---

# 🔮 Future Plans

Once the basic neural network is functional, the project may be extended with:

* Different optimizers
* Better initialization strategies
* Dataset utilities
* Batch processing
* Model serialization
* Training metrics
* Additional activation functions
* Additional loss functions
* More sophisticated layer types
* Numerical gradient checking
* Performance experiments
* Visualization of training progress

---

# 📚 Learning Outcome

The final objective is to have a working neural network **and** understand how its major components work.

By the end of the project, the goal is to be comfortable implementing and reasoning about:

```text
Input
  ↓
Dense Layer
  ↓
Activation
  ↓
Dense Layer
  ↓
Activation
  ↓
Output
  ↓
Loss
  ↓
Gradient
  ↓
Backpropagation
  ↓
Weight Updates
  ↺
```

The project is therefore both a software project and a learning journey into the foundations of machine learning.

---

## ⚠️ Project Status

**Status: In Development**

The project is currently paused temporarily while academic coursework and examinations take priority.

The current development checkpoint is the **initial Dense Layer implementation**. Further development will continue with Dense Layer refinement, backpropagation, and training once development resumes.
