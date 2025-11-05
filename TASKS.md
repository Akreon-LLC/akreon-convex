# Project Management

Core development tasklist for MVP (alpha release) which can move and is flexible. Implicitly all functions need to include tests and mostly all core functionality will be constexpr safe.

TODO to scope:
- dynamic CRTP wrapper
- python UX
- GUI block diag & plots
- ** simulator visualization **

## Math
### Linear Algebra
** All of the below are constexpr safe **
- [x] vectors
- [x] matrices
- [x] linear algebra and vector math
- [x] householder reflections
- [x] hessenberg matrices
- [x] qr decomposition via householder & hessenberg
- [x] lsq via QR & back-solve with R
- [x] svd via QR & back-solve (U \Sigma V)
- [x] inverse operation (QR + backsolve for I)
- [x] jordan form (eig decomp)
- [x] matrix exponential
- [] LDL^T / Cholesky Decomposition

### Convex Analysis

#### Affine Sets & maps
- [x] implicit affine sets
- [x] parametric affine sets
- [x] affine maps
- [x] affine transforms [R | t]
#### Convex Sets
- [x] convex set base
- [x] ellipsoid
- [x] halfspace
- [x] norm ball
- [x] norm cone
- [x] psd cone
- [x] simplex
- [x] polyhedron
- [x] nonnegative orthant (x >= 0)
- [x] neg orthant (x <= 0)
- [x] LMI
- [x] rotated soc cone
- [x] exp cone
- [x] power cone
- [x] log det cone
- [x] conic view (via inv perspective func)
- [x] affine transform
- [x] equality operators
- [x] linear fractional
- [] dual cone (wait till duality)
- [] separating hyperplane
- [] supporting hyperplane
#### Convex Functions
- [x] convex func base
- [x] concave func base
- [x] affine func
- [x] max func
- [x] norm func
- [x] quad / linear func
- [x] indicator func
- [x] log sum exp func
- [x] log det func
- [x] geometric mean func
- [] atomic funcs (exp, log, abs, square, sum_squares, inv)
- [x] epigraph
- [x] hypograph
- [x] convex fn -> concave fn (negation) + RT
- [] non-neg weighted sum
- [] scalar composition
- [] vector compositions
- [] conjugate function
- [] quasi-convex functions
- [] log-convex & log-concave functions

#### Convex Problems
- [x] form cone blocks (slacks)
- [x] cone program base (min. c^T x s.t. F z + g \in K)
- [x] convex program base (min. f_0(x) s.t. f_i(x) <= 0, A x = b)

#### Duality
- [] duality & KKT

#### Convex Optimization
- [] convex problem formulations & modeling
- [] convex opt newton solver
- [] convex opt interior point solver

### Operators & Morphisms
- [] linear map abstraction
- [] composition, identity, inverse
- [] adjoint + dual map
- [] kronecker product
- [] block structures
### Norms and inner products
- [x] l1/l2/l-inf vector norms
- [x] frobenius matrix norm
- [] induced matrix norm
- [] operator norm bounds
### Functional analysis
- [] lyapnuov eq solver
- [x] spectral radius
- [] hurwitz stability classification
- [] system gain computations
### Polynomials
- [] polynomial class
- [] rational function
- [] arithmetic on polynomials
- [] evaluation (horner's rule)

### Differentiable Optimization

** TODO **

## Controls
** All of the below are constexpr safe **

## Modern MIMO Control

- [x] state space structure with A & B + C & D
- [] minimal realization
- [x] gramians (sort of done with SVD A^T A... PBH test vs. Wr / Wo mats too)
- [x] reachability analysis
- [] pole placement state feedback
- [] lqr
- [] integral action
- [x] observability analysis
- [] luenberger observer
- [] kalman filtering
- [] lqg

## Classic SISO Control

- [] add transfer functions in s-domain
- [] step response
- [] freq response
- [] convolution operator
- [] laplace transforms
- [] block diagrams
- [] bode plots (gain & phase... minimum phase)
- [] nyquist criterion and plots
- [] stability margins
- [] PID
- [] sensitivity functions
- [] feedfoward control
- [] loop shaping
** Future **
Robust Control

## Physics (core)
** All of the below are constexpr safe **
- [x] DEL operator
- [x] Dynamical System CRTP
- [x] VI (central differencing)
- [x] tests for various systems (1DoF to 2DoF Pendulum and SMD)
- [] form A & B matrices (linearized from jacobian via DEL) -- Use controls state space ??
- [] \dot{x} = A x test reconstruction
- [] add forced system and test \dot{x} = A x + B u
** Future **
- [] autodiff to replace central differencing?

## UX
** TODO **
