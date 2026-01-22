"""
State space model interface for RHMC.

A state space model defines:
- Dynamics: x_t = f(x_{t-1}, θ) + ε_t where ε_t ~ N(0, Q)
- Observations: y_t ~ ExpFam(η_t) where η_t = η(x_t, θ)

Models are defined as subtypes of `StateSpaceModel{Dx, Dy, Dp}` where:
- Dx: state dimension
- Dy: observation dimension
- Dp: parameter dimension

Each model must implement methods for the dynamics function f, observation natural
parameter function η, and their derivatives up to third order (for observed Hessian RHMC).
"""

export StateSpaceModel
export f, Df_x, Df_θ, D2f_xx, D2f_xθ, D2f_θθ, DDf_x_θ
export D3f_xxx, D3f_xxθ, D3f_xθθ, D3f_θθθ
export η, Dη_x, Dη_θ, D2η_xx, D2η_xθ, D2η_θθ, DDη_x_θ
export D3η_xxx, D3η_xxθ
export dynamics_covariance, dynamics_covariance_inv, initial_prior, observation_family
export state_dim, obs_dim, param_dim

# =============================================================================
# Abstract Type
# =============================================================================

"""
    StateSpaceModel{Dx, Dy, Dp}

Abstract type for state space models with:
- `Dx`-dimensional state
- `Dy`-dimensional observations
- `Dp`-dimensional parameters

Subtypes must implement the dynamics, observation, and configuration methods below.
"""
abstract type StateSpaceModel{Dx,Dy,Dp} end

state_dim(::StateSpaceModel{Dx}) where {Dx} = Dx
obs_dim(::StateSpaceModel{Dx,Dy}) where {Dx,Dy} = Dy
param_dim(::StateSpaceModel{Dx,Dy,Dp}) where {Dx,Dy,Dp} = Dp

# =============================================================================
# Dynamics Interface
# =============================================================================

"""
    f(model, x::SVector{Dx}, θ::SVector{Dp}) → SVector{Dx}

Dynamics mean function: E[x_t | x_{t-1}, θ] = f(x_{t-1}, θ)
"""
function f end

"""
    Df_x(model, x::SVector{Dx}, θ::SVector{Dp}) → SMatrix{Dx, Dx}

Jacobian of dynamics w.r.t. state: ∂f/∂x
"""
function Df_x end

"""
    Df_θ(model, x::SVector{Dx}, θ::SVector{Dp}) → SMatrix{Dx, Dp}

Jacobian of dynamics w.r.t. parameters: ∂f/∂θ
"""
function Df_θ end

"""
    D2f_xx(model, x::SVector{Dx}, θ::SVector{Dp}) → NTuple{Dx, SMatrix{Dx, Dx}}

Derivatives of dynamics Jacobian columns w.r.t. state: ∂(Df_x[:, d])/∂x for d = 1, ..., Dx
Returns Dx matrices, each Dx × Dx.

Note: Entry [i, j] of the d-th matrix is ∂²f_i/(∂x_d ∂x_j).
"""
function D2f_xx end

"""
    D2f_xθ(model, x::SVector{Dx}, θ::SVector{Dp}) → NTuple{Dx, SMatrix{Dx, Dp}}

Derivatives of dynamics Jacobian columns w.r.t. parameters: ∂(Df_x[:, d])/∂θ for d = 1, ..., Dx
Returns Dx matrices, each Dx × Dp.

Note: Entry [i, p] of the d-th matrix is ∂²f_i/(∂x_d ∂θ_p).
"""
function D2f_xθ end

"""
    D2f_θθ(model, x::SVector{Dx}, θ::SVector{Dp}) → NTuple{Dp, SMatrix{Dx, Dp}}

Second parameter derivatives: ∂²f/∂θ_p∂θ for p = 1, ..., Dp
Returns Dp matrices, each Dx × Dp.
"""
function D2f_θθ end

"""
    DDf_x_θ(model, x::SVector{Dx}, θ::SVector{Dp}) → NTuple{Dp, SMatrix{Dx, Dx}}

Derivatives of state Jacobian w.r.t. parameters: ∂(Df_x)/∂θ_p for p = 1, ..., Dp
Returns Dp matrices, each Dx × Dx.
"""
function DDf_x_θ end

# =============================================================================
# Third-Order Dynamics Derivatives (for Observed Hessian RHMC)
# =============================================================================

"""
    D3f_xxx(model, x::SVector{Dx}, θ::SVector{Dp}) → NTuple{Dx, NTuple{Dx, SMatrix{Dx, Dx}}}

Third derivatives of dynamics w.r.t. state: ∂³f/∂x_c∂x_d∂x_j for all c, d, j.

Returns Dx tuples (one per input component c), each containing Dx matrices (one per d).
Entry [i, j] of matrix (c, d) is ∂³f_i/(∂x_c ∂x_d ∂x_j).

These are used in observed Hessian RHMC for computing ∂(∇²f_output)/∂x.
"""
function D3f_xxx end

"""
    D3f_xxθ(model, x::SVector{Dx}, θ::SVector{Dp}) → NTuple{Dx, NTuple{Dx, SMatrix{Dx, Dp}}}

Third derivatives: ∂³f/∂x_c∂x_d∂θ_p for all c, d, p.

Returns Dx tuples (one per input component c), each containing Dx matrices (one per d).
Entry [i, p] of matrix (c, d) is ∂³f_i/(∂x_c ∂x_d ∂θ_p).

These are used in observed Hessian RHMC for computing ∂(∇²_{x,θ}f)/∂x.
"""
function D3f_xxθ end

"""
    D3f_xθθ(model, x::SVector{Dx}, θ::SVector{Dp}) → NTuple{Dx, NTuple{Dp, SMatrix{Dx, Dp}}}

Third derivatives: ∂³f/∂x_c∂θ_p∂θ_q for all c, p, q.

Returns Dx tuples (one per input component c), each containing Dp matrices (one per p).
Entry [i, q] of matrix (c, p) is ∂³f_i/(∂x_c ∂θ_p ∂θ_q).

These are used in observed Hessian RHMC for computing ∂(f_θθ)/∂x.
"""
function D3f_xθθ end

"""
    D3f_θθθ(model, x::SVector{Dx}, θ::SVector{Dp}) → NTuple{Dp, NTuple{Dp, SMatrix{Dx, Dp}}}

Third derivatives: ∂³f/∂θ_p∂θ_q∂θ_r for all p, q, r.

Returns Dp tuples (one per p), each containing Dp matrices (one per q).
Entry [i, r] of matrix (p, q) is ∂³f_i/(∂θ_p ∂θ_q ∂θ_r).

These are used in observed Hessian RHMC for computing ∂(f_θθ)/∂θ.
"""
function D3f_θθθ end

# =============================================================================
# Observation Interface
# =============================================================================

"""
    η(model, x::SVector{Dx}, θ::SVector{Dp}) → SVector{Dy}

Natural parameter function for observations: η_t = η(x_t, θ)
"""
function η end

"""
    Dη_x(model, x::SVector{Dx}, θ::SVector{Dp}) → SMatrix{Dy, Dx}

Jacobian of natural parameter w.r.t. state: ∂η/∂x
"""
function Dη_x end

"""
    Dη_θ(model, x::SVector{Dx}, θ::SVector{Dp}) → SMatrix{Dy, Dp}

Jacobian of natural parameter w.r.t. parameters: ∂η/∂θ
"""
function Dη_θ end

"""
    D2η_xx(model, x::SVector{Dx}, θ::SVector{Dp}) → NTuple{Dx, SMatrix{Dy, Dx}}

Derivatives of observation Jacobian columns w.r.t. state: ∂(Dη_x[:, d])/∂x for d = 1, ..., Dx
Returns Dx matrices, each Dy × Dx.

Note: Entry [m, j] of the d-th matrix is ∂²η_m/(∂x_d ∂x_j).
"""
function D2η_xx end

"""
    D2η_xθ(model, x::SVector{Dx}, θ::SVector{Dp}) → NTuple{Dx, SMatrix{Dy, Dp}}

Derivatives of observation Jacobian columns w.r.t. parameters: ∂(Dη_x[:, d])/∂θ for d = 1, ..., Dx
Returns Dx matrices, each Dy × Dp.

Note: Entry [m, p] of the d-th matrix is ∂²η_m/(∂x_d ∂θ_p).
"""
function D2η_xθ end

"""
    D2η_θθ(model, x::SVector{Dx}, θ::SVector{Dp}) → NTuple{Dp, SMatrix{Dy, Dp}}

Second parameter derivatives: ∂²η/∂θ_p∂θ for p = 1, ..., Dp
Returns Dp matrices, each Dy × Dp.
"""
function D2η_θθ end

"""
    DDη_x_θ(model, x::SVector{Dx}, θ::SVector{Dp}) → NTuple{Dp, SMatrix{Dy, Dx}}

Derivatives of state Jacobian w.r.t. parameters: ∂(Dη_x)/∂θ_p for p = 1, ..., Dp
Returns Dp matrices, each Dy × Dx.
"""
function DDη_x_θ end

# =============================================================================
# Third-Order Observation Derivatives (for Observed Hessian RHMC)
# =============================================================================

"""
    D3η_xxx(model, x::SVector{Dx}, θ::SVector{Dp}) → NTuple{Dx, NTuple{Dx, SMatrix{Dy, Dx}}}

Third derivatives of observation natural parameter w.r.t. state: ∂³η/∂x_c∂x_d∂x_j.

Returns Dx tuples (one per input component c), each containing Dx matrices (one per d).
Entry [m, j] of matrix (c, d) is ∂³η_m/(∂x_c ∂x_d ∂x_j).

For linear observation functions (η linear in x), this is all zeros.
"""
function D3η_xxx end

"""
    D3η_xxθ(model, x::SVector{Dx}, θ::SVector{Dp}) → NTuple{Dx, NTuple{Dx, SMatrix{Dy, Dp}}}

Third derivatives: ∂³η/∂x_c∂x_d∂θ_p for all c, d, p.

Returns Dx tuples (one per input component c), each containing Dx matrices (one per d).
Entry [m, p] of matrix (c, d) is ∂³η_m/(∂x_c ∂x_d ∂θ_p).

For observation functions that are linear in x and θ, this is all zeros.
"""
function D3η_xxθ end

# =============================================================================
# Configuration Methods
# =============================================================================

"""
    dynamics_covariance(model) → AbstractMatrix{T}

Process noise covariance Q. Should return a Diagonal or SMatrix for efficiency.
"""
function dynamics_covariance end

"""
    dynamics_covariance_inv(model) → AbstractMatrix{T}

Inverse of process noise covariance Q⁻¹.
Default implementation inverts `dynamics_covariance(model)`.
"""
function dynamics_covariance_inv(model::StateSpaceModel)
    return inv(dynamics_covariance(model))
end

"""
    initial_prior(model) → Distribution

Prior distribution for initial state x_1. Typically MvNormal.
"""
function initial_prior end

"""
    observation_family(model) → ExponentialFamily or ProductExponentialFamily

The exponential family distribution for observations.
"""
function observation_family end
