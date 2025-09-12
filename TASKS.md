# Project Management (Spiral)

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
- [x] matrix exponential
- [] LDL^T / Cholesky Decomposition
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

## Controls 
** All of the below are constexpr safe **
Modern MIMO Control
- [] state space structure with A & B 
- [] minimal realization
- [] gramians
- [] reachability analysis
- [] pole placement state feedback
- [] lqr 
- [] integral action
- [] observability analysis
- [] luenberger observer
- [] kalman filtering
- [] lqg
Classic SISO Control
- [] add transfer functions in s-domain
- [] step response
- [] freq response
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
