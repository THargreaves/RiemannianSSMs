"""
Bivariate stochastic volatility model.

State: h = [h_1, h_2] where h_i is the log-volatility of asset i
Parameters: θ = [atanh(ϕ), log(κ)] where:
    - ϕ ∈ (-1, 1) controls persistence (near 1 = highly persistent)
    - κ > 0 controls nonlinear mean reversion strength

Dynamics (nonlinear AR with cubic damping):
    h_{t+1} = ϕ·h_t - κ·h_t³ + σ·η_t,  η_t ~ N(0, I)

This combines:
    - Linear persistence (ϕ close to 1 for realistic volatility)
    - Cubic damping (-κh³) for symmetric stabilization around zero
    - Small process noise σ for near-deterministic dynamics

The cubic term provides symmetric mean reversion:
    - When h > 0 and large: -κh³ < 0 (pushes down)
    - When h < 0 and large: -κh³ > 0 (pushes up)

Observations (zero-mean Gaussian with stochastic variance):
    y_t | h_t ~ N(0, diag(exp(h_{1t}), exp(h_{2t})))

The observation natural parameter is precision τ = exp(-h), so:
    η(h, θ) = exp(-h)  (element-wise)

This model is designed to create challenging geometry for HMC:
    - Near-unit-root dynamics create long-range correlations
    - Small process noise creates narrow posterior tubes
    - State-dependent observation variance couples states and data strongly
    - Nonlinear dynamics create position-dependent curvature
"""

export BivariateSVModel

struct BivariateSVModel{T} <: StateSpaceModel{2,2,2}
    σ::T                          # Process noise std (fixed)
    h0::SVector{2,T}              # Initial state mean
    Σ0::Diagonal{T,SVector{2,T}}  # Initial state covariance
end

function BivariateSVModel(;
    σ::T=0.05,
    h0::SVector{2,T}=SVector{2,T}(0.0, 0.0),
    Σ0_diag::SVector{2,T}=SVector{2,T}(1.0, 1.0),
) where {T}
    Σ0 = Diagonal(Σ0_diag)
    return BivariateSVModel{T}(σ, h0, Σ0)
end

# =============================================================================
# Configuration Methods
# =============================================================================

function dynamics_covariance(model::BivariateSVModel{T}) where {T}
    return Diagonal(SVector{2,T}(model.σ^2, model.σ^2))
end

function dynamics_covariance_inv(model::BivariateSVModel{T}) where {T}
    return Diagonal(SVector{2,T}(1 / model.σ^2, 1 / model.σ^2))
end

function initial_prior(model::BivariateSVModel)
    return MvNormal(model.h0, model.Σ0)
end

function observation_family(::BivariateSVModel)
    return ProductExponentialFamily{2}(GaussianPrecision())
end

# =============================================================================
# Dynamics: f(h, θ) = ϕ·h - κ·(h ⊙ h)
# =============================================================================

"""
Extract parameters from unconstrained θ.
    ϕ = tanh(θ[1]) ∈ (-1, 1)
    κ = exp(θ[2]) > 0
"""
@inline function _sv_params(θ::SVector{2,T}) where {T}
    ϕ = tanh(θ[1])
    κ = exp(θ[2])
    return ϕ, κ
end

function f(::BivariateSVModel{T}, h::SVector{2,T}, θ::SVector{2,T}) where {T}
    ϕ, κ = _sv_params(θ)
    return SVector{2,T}(ϕ * h[1] - κ * h[1]^3, ϕ * h[2] - κ * h[2]^3)
end

"""
Jacobian ∂f/∂h (2×2 diagonal matrix).
    ∂f_i/∂h_i = ϕ - 3κh_i²
    ∂f_i/∂h_j = 0  for i ≠ j
"""
function Df_x(::BivariateSVModel{T}, h::SVector{2,T}, θ::SVector{2,T}) where {T}
    ϕ, κ = _sv_params(θ)
    return @SMatrix [
        ϕ-3κ*h[1]^2 zero(T)
        zero(T) ϕ-3κ*h[2]^2
    ]
end

"""
Jacobian ∂f/∂θ (2×2 matrix).
    θ = [atanh(ϕ), log(κ)]

    ∂f_i/∂θ_1 = ∂f_i/∂ϕ · ∂ϕ/∂θ_1 = h_i · (1 - ϕ²)
    ∂f_i/∂θ_2 = ∂f_i/∂κ · ∂κ/∂θ_2 = -h_i³ · κ
"""
function Df_θ(::BivariateSVModel{T}, h::SVector{2,T}, θ::SVector{2,T}) where {T}
    ϕ, κ = _sv_params(θ)
    dϕ_dθ1 = 1 - ϕ^2  # sech²(θ[1])
    dκ_dθ2 = κ

    return @SMatrix [
        h[1]*dϕ_dθ1 -h[1]^3*dκ_dθ2
        h[2]*dϕ_dθ1 -h[2]^3*dκ_dθ2
    ]
end

"""
Second derivatives ∂(Df_x[:, d])/∂h for d = 1, 2.

Df_x = diag(ϕ - 3κh_1², ϕ - 3κh_2²)

For d=1 (first column [ϕ - 3κh_1², 0]'):
    ∂/∂h_1 = -6κh_1, ∂/∂h_2 = 0

For d=2 (second column [0, ϕ - 3κh_2²]'):
    ∂/∂h_1 = 0, ∂/∂h_2 = -6κh_2
"""
function D2f_xx(::BivariateSVModel{T}, h::SVector{2,T}, θ::SVector{2,T}) where {T}
    _, κ = _sv_params(θ)

    Hf_1 = @SMatrix [
        -6κ*h[1] zero(T)
        zero(T) zero(T)
    ]

    Hf_2 = @SMatrix [
        zero(T) zero(T)
        zero(T) -6κ*h[2]
    ]

    return (Hf_1, Hf_2)
end

"""
Mixed derivatives ∂(Df_x[:, d])/∂θ for d = 1, 2.

For d=1: ∂/∂θ of [ϕ - 3κh_1², 0]'
    ∂/∂θ_1 = (1-ϕ²), ∂/∂θ_2 = -3κh_1²

For d=2: ∂/∂θ of [0, ϕ - 3κh_2²]'
    ∂/∂θ_1 = (1-ϕ²), ∂/∂θ_2 = -3κh_2²
"""
function D2f_xθ(::BivariateSVModel{T}, h::SVector{2,T}, θ::SVector{2,T}) where {T}
    ϕ, κ = _sv_params(θ)
    dϕ_dθ1 = 1 - ϕ^2

    Hf_θ_1 = @SMatrix [
        dϕ_dθ1 -3κ*h[1]^2
        zero(T) zero(T)
    ]

    Hf_θ_2 = @SMatrix [
        zero(T) zero(T)
        dϕ_dθ1 -3κ*h[2]^2
    ]

    return (Hf_θ_1, Hf_θ_2)
end

"""
Second parameter derivatives ∂²f/∂θ_p∂θ for p = 1, 2.

Df_θ = [h_1·(1-ϕ²), -h_1³·κ; h_2·(1-ϕ²), -h_2³·κ]

For p=1: ∂(Df_θ)/∂θ_1
    ∂/∂θ_1 of [h_1·(1-ϕ²), h_2·(1-ϕ²)]' = [-2ϕ(1-ϕ²)h_1, -2ϕ(1-ϕ²)h_2]'
    ∂/∂θ_2 = [0, 0]'

For p=2: ∂(Df_θ)/∂θ_2
    ∂/∂θ_1 = [0, 0]'
    ∂/∂θ_2 of [-h_1³·κ, -h_2³·κ]' = [-h_1³·κ, -h_2³·κ]'
"""
function D2f_θθ(::BivariateSVModel{T}, h::SVector{2,T}, θ::SVector{2,T}) where {T}
    ϕ, κ = _sv_params(θ)
    d2ϕ_dθ1 = -2ϕ * (1 - ϕ^2)

    Hf_θ_1 = @SMatrix [
        h[1]*d2ϕ_dθ1 zero(T)
        h[2]*d2ϕ_dθ1 zero(T)
    ]

    Hf_θ_2 = @SMatrix [
        zero(T) -h[1]^3*κ
        zero(T) -h[2]^3*κ
    ]

    return (Hf_θ_1, Hf_θ_2)
end

"""
Derivatives of state Jacobian w.r.t. parameters: ∂(Df_x)/∂θ_p for p = 1, 2.

Df_x = diag(ϕ - 3κh_1², ϕ - 3κh_2²)

For p=1: ∂(Df_x)/∂θ_1 = diag((1-ϕ²), (1-ϕ²))
For p=2: ∂(Df_x)/∂θ_2 = diag(-3κh_1², -3κh_2²)
"""
function DDf_x_θ(::BivariateSVModel{T}, h::SVector{2,T}, θ::SVector{2,T}) where {T}
    ϕ, κ = _sv_params(θ)
    dϕ_dθ1 = 1 - ϕ^2

    dDf_x_dθ1 = @SMatrix [
        dϕ_dθ1 zero(T)
        zero(T) dϕ_dθ1
    ]

    dDf_x_dθ2 = @SMatrix [
        -3κ*h[1]^2 zero(T)
        zero(T) -3κ*h[2]^2
    ]

    return (dDf_x_dθ1, dDf_x_dθ2)
end

# =============================================================================
# Observations: η(h, θ) = exp(-h) (precision parameterization)
# =============================================================================

"""
Natural parameter function for observations.
For GaussianPrecision, η = τ = 1/σ² = exp(-h).
"""
function η(::BivariateSVModel{T}, h::SVector{2,T}, θ::SVector{2,T}) where {T}
    return SVector{2,T}(exp(-h[1]), exp(-h[2]))
end

"""
Jacobian ∂η/∂h (2×2 diagonal matrix).
    η_i = exp(-h_i)
    ∂η_i/∂h_i = -exp(-h_i)
"""
function Dη_x(::BivariateSVModel{T}, h::SVector{2,T}, θ::SVector{2,T}) where {T}
    return @SMatrix [
        -exp(-h[1]) zero(T)
        zero(T) -exp(-h[2])
    ]
end

"""
Jacobian ∂η/∂θ (2×2 matrix). η does not depend on θ, so this is zero.
"""
function Dη_θ(::BivariateSVModel{T}, h::SVector{2,T}, θ::SVector{2,T}) where {T}
    return @SMatrix zeros(T, 2, 2)
end

"""
Second derivatives ∂(Dη_x[:, d])/∂h for d = 1, 2.

Dη_x = diag(-exp(-h_1), -exp(-h_2))

For d=1: ∂/∂h of [-exp(-h_1), 0]' = [exp(-h_1), 0; 0, 0]
For d=2: ∂/∂h of [0, -exp(-h_2)]' = [0, 0; 0, exp(-h_2)]
"""
function D2η_xx(::BivariateSVModel{T}, h::SVector{2,T}, θ::SVector{2,T}) where {T}
    Hη_1 = @SMatrix [
        exp(-h[1]) zero(T)
        zero(T) zero(T)
    ]

    Hη_2 = @SMatrix [
        zero(T) zero(T)
        zero(T) exp(-h[2])
    ]

    return (Hη_1, Hη_2)
end

"""
Mixed derivatives ∂(Dη_x[:, d])/∂θ. All zeros since η doesn't depend on θ.
"""
function D2η_xθ(::BivariateSVModel{T}, h::SVector{2,T}, θ::SVector{2,T}) where {T}
    Z = @SMatrix zeros(T, 2, 2)
    return (Z, Z)
end

"""
Second parameter derivatives ∂²η/∂θ_p∂θ. All zeros since η doesn't depend on θ.
"""
function D2η_θθ(::BivariateSVModel{T}, h::SVector{2,T}, θ::SVector{2,T}) where {T}
    Z = @SMatrix zeros(T, 2, 2)
    return (Z, Z)
end

"""
Derivatives of state Jacobian w.r.t. parameters. All zeros since Dη_x doesn't depend on θ.
"""
function DDη_x_θ(::BivariateSVModel{T}, h::SVector{2,T}, θ::SVector{2,T}) where {T}
    Z = @SMatrix zeros(T, 2, 2)
    return (Z, Z)
end
