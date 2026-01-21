"""
Van der Pol oscillator state space model.

State: x = [u, v] where u is position, v is velocity
Parameter: θ = [log(μ)] where μ controls the nonlinearity strength

Continuous-time dynamics:
    u̇ = v
    v̇ = μ(1 - u²)v - u

Euler discretization with step δt:
    u_{t+1} = u_t + δt * v_t
    v_{t+1} = v_t + δt * [μ(1 - u_t²)v_t - u_t] + ε_t

Observations: partial linear observation of u with Gaussian noise
    y_t = u_t + η_t where η_t ~ N(0, σ_obs²)
"""

export VanDerPolModel

struct VanDerPolModel{T} <: StateSpaceModel{2,1,1}
    δt::T           # Time step
    σ_u::T          # Process noise std on u component
    σ_v::T          # Process noise std on v component
    σ_obs::T        # Observation noise std
    μ0::SVector{2,T}  # Initial state mean
    Σ0::Diagonal{T,SVector{2,T}}  # Initial state covariance
end

function VanDerPolModel(;
    δt::T=0.1,
    σ_u::T=0.01,
    σ_v::T=0.1,
    σ_obs::T=0.5,
    μ0::SVector{2,T}=SVector{2,T}(1.0, 0.0),
    Σ0_diag::SVector{2,T}=SVector{2,T}(0.1, 0.1),
) where {T}
    Σ0 = Diagonal(Σ0_diag)
    return VanDerPolModel{T}(δt, σ_u, σ_v, σ_obs, μ0, Σ0)
end

# =============================================================================
# Configuration Methods
# =============================================================================

function dynamics_covariance(model::VanDerPolModel{T}) where {T}
    return Diagonal(SVector{2,T}(model.σ_u^2, model.σ_v^2))
end

function dynamics_covariance_inv(model::VanDerPolModel{T}) where {T}
    return Diagonal(SVector{2,T}(1 / model.σ_u^2, 1 / model.σ_v^2))
end

function initial_prior(model::VanDerPolModel)
    return MvNormal(model.μ0, model.Σ0)
end

function observation_family(model::VanDerPolModel)
    return ProductExponentialFamily{1}(GaussianMean(model.σ_obs^2))
end

# =============================================================================
# Dynamics: f(x, θ)
# =============================================================================

function f(model::VanDerPolModel{T}, x::SVector{2,T}, θ::SVector{1,T}) where {T}
    u, v = x
    log_μ = θ[1]
    μ = exp(log_μ)
    δt = model.δt

    new_u = u + δt * v
    new_v = v + δt * (μ * (1 - u^2) * v - u)

    return SVector{2,T}(new_u, new_v)
end

"""
Jacobian ∂f/∂x (2×2 matrix).

∂f_u/∂u = 1,                    ∂f_u/∂v = δt
∂f_v/∂u = δt(μ(-2u)v - 1),      ∂f_v/∂v = 1 + δt·μ(1-u²)
"""
function Df_x(model::VanDerPolModel{T}, x::SVector{2,T}, θ::SVector{1,T}) where {T}
    u, v = x
    μ = exp(θ[1])
    δt = model.δt

    return @SMatrix [
        one(T) δt
        δt*(μ*(-2u)*v - 1) 1+δt*μ*(1-u^2)
    ]
end

"""
Jacobian ∂f/∂θ (2×1 matrix).

θ = [log(μ)], using chain rule: ∂f/∂θ = ∂f/∂μ * μ

∂f_u/∂θ = 0
∂f_v/∂θ = δt·μ(1-u²)v
"""
function Df_θ(model::VanDerPolModel{T}, x::SVector{2,T}, θ::SVector{1,T}) where {T}
    u, v = x
    μ = exp(θ[1])
    δt = model.δt

    return @SMatrix [
        zero(T)
        δt * μ * (1 - u^2) * v
    ]
end

"""
Derivatives of Jacobian columns: ∂(Df_x[:, d])/∂x for d = 1, 2.

Df_x = [1, δt; δt*(μ*(-2u)*v - 1), 1 + δt*μ*(1-u²)]

For d=1 (derivatives of first column w.r.t. x):
    Df_x[:, 1] = [1, δt*(μ*(-2u)*v - 1)]
    ∂/∂u: [0, δt*μ*(-2)*v]
    ∂/∂v: [0, δt*μ*(-2u)]

For d=2 (derivatives of second column w.r.t. x):
    Df_x[:, 2] = [δt, 1 + δt*μ*(1-u²)]
    ∂/∂u: [0, δt*μ*(-2u)]
    ∂/∂v: [0, 0]
"""
function D2f_xx(model::VanDerPolModel{T}, x::SVector{2,T}, θ::SVector{1,T}) where {T}
    u, v = x
    μ = exp(θ[1])
    δt = model.δt

    # ∂(Df_x[:, 1])/∂x - derivative of first Jacobian column
    Hf_1 = @SMatrix [
        zero(T) zero(T)
        -2*δt*μ*v -2*δt*μ*u
    ]

    # ∂(Df_x[:, 2])/∂x - derivative of second Jacobian column
    Hf_2 = @SMatrix [
        zero(T) zero(T)
        -2*δt*μ*u zero(T)
    ]

    return (Hf_1, Hf_2)
end

"""
Derivatives of Jacobian columns w.r.t. parameters: ∂(Df_x[:, d])/∂θ for d = 1, 2.

Df_θ = [0; δt·μ(1-u²)v]

For d=1: ∂(Df_θ)/∂u = [0; δt·μ·(-2u)·v]
For d=2: ∂(Df_θ)/∂v = [0; δt·μ(1-u²)]
"""
function D2f_xθ(model::VanDerPolModel{T}, x::SVector{2,T}, θ::SVector{1,T}) where {T}
    u, v = x
    μ = exp(θ[1])
    δt = model.δt

    # ∂(Df_x[:, 1])/∂θ = ∂(Df_θ)/∂x_1 (by symmetry of mixed partials)
    Hf_θ_1 = @SMatrix [
        zero(T)
        -2*δt*μ*u*v
    ]

    # ∂(Df_x[:, 2])/∂θ = ∂(Df_θ)/∂x_2
    Hf_θ_2 = @SMatrix [
        zero(T)
        δt*μ*(1-u^2)
    ]

    return (Hf_θ_1, Hf_θ_2)
end

"""
Second parameter derivatives ∂²f/∂θ_p∂θ.

For log-parameterization with θ = log(μ):
∂f/∂θ = δt·μ(1-u²)v · [0; 1]
∂²f/∂θ² = δt·μ(1-u²)v · [0; 1] = ∂f/∂θ (since ∂μ/∂θ = μ)
"""
function D2f_θθ(model::VanDerPolModel{T}, x::SVector{2,T}, θ::SVector{1,T}) where {T}
    df_dθ = Df_θ(model, x, θ)
    return (df_dθ,)  # 1-tuple since Dp = 1
end

"""
Derivatives of state Jacobian w.r.t. parameters: ∂(Df_x)/∂θ.

Df_x = [1, δt; δt*(μ*(-2u)*v - 1), 1 + δt*μ*(1-u²)]

∂(Df_x)/∂θ = [0, 0; δt*μ*(-2u)*v, δt*μ*(1-u²)]
"""
function DDf_x_θ(model::VanDerPolModel{T}, x::SVector{2,T}, θ::SVector{1,T}) where {T}
    u, v = x
    μ = exp(θ[1])
    δt = model.δt

    dDf_x_dθ = @SMatrix [
        zero(T) zero(T)
        δt*μ*(-2u)*v δt*μ*(1-u^2)
    ]

    return (dDf_x_dθ,)  # 1-tuple since Dp = 1
end

# =============================================================================
# Observations: η(x, θ) - partial linear observation of u
# =============================================================================

"""
Natural parameter function. For Gaussian with mean parameterization, η = μ = u.
"""
function η(model::VanDerPolModel{T}, x::SVector{2,T}, θ::SVector{1,T}) where {T}
    return SVector{1,T}(x[1])  # Observe u only
end

"""
Jacobian ∂η/∂x (1×2 matrix). η = u, so ∂η/∂x = [1, 0].
"""
function Dη_x(model::VanDerPolModel{T}, x::SVector{2,T}, θ::SVector{1,T}) where {T}
    return @SMatrix [one(T) zero(T)]
end

"""
Jacobian ∂η/∂θ (1×1 matrix). η does not depend on θ, so this is zero.
"""
function Dη_θ(model::VanDerPolModel{T}, x::SVector{2,T}, θ::SVector{1,T}) where {T}
    return @SMatrix [zero(T)]
end

"""
Derivatives of Jacobian columns: ∂(Dη_x[:, d])/∂x for d = 1, 2.
η is linear in x, so all zeros.
Returns Dx (=2) matrices, each Dy × Dx (1 × 2).
"""
function D2η_xx(model::VanDerPolModel{T}, x::SVector{2,T}, θ::SVector{1,T}) where {T}
    Z = @SMatrix zeros(T, 1, 2)
    return (Z, Z)  # Dx-tuple
end

"""
Derivatives of Jacobian columns w.r.t. parameters: ∂(Dη_x[:, d])/∂θ for d = 1, 2.
η doesn't depend on θ, so all zeros.
Returns Dx (=2) matrices, each Dy × Dp (1 × 1).
"""
function D2η_xθ(model::VanDerPolModel{T}, x::SVector{2,T}, θ::SVector{1,T}) where {T}
    Z = @SMatrix zeros(T, 1, 1)
    return (Z, Z)  # Dx-tuple
end

"""
Second parameter derivatives ∂²η/∂θ_p∂θ. All zeros since η doesn't depend on θ.
"""
function D2η_θθ(model::VanDerPolModel{T}, x::SVector{2,T}, θ::SVector{1,T}) where {T}
    return (@SMatrix(zeros(T, 1, 1)),)  # 1-tuple since Dp = 1
end

"""
Derivatives of state Jacobian w.r.t. parameters. All zeros since Dη_x doesn't depend on θ.
"""
function DDη_x_θ(model::VanDerPolModel{T}, x::SVector{2,T}, θ::SVector{1,T}) where {T}
    return (@SMatrix(zeros(T, 1, 2)),)  # 1-tuple since Dp = 1
end
