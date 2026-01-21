"""
Random walk dynamics with Poisson observations for benchmarking RHMC.

State: x = [x_1, x_2] - 2D random walk
Parameters: θ = [a_1, a_2, b] - linear predictor coefficients

Dynamics (identity/random walk):
    x_{t+1} = x_t + ε_t where ε_t ~ N(0, Q)

Observations (Poisson with log-linear predictor):
    y_t ~ Poisson(λ_t) where λ_t = exp(η_t)
    η_t = a_1·x_1 + a_2·x_2 + b

This creates challenging geometry because:
- The parameters enter only through the observation model
- The state-dependent Fisher information S = exp(η) varies with state
- Random walk dynamics allow the state to drift, creating heterogeneous curvature
"""

export RandomWalkPoissonModel

struct RandomWalkPoissonModel{T} <: StateSpaceModel{2,1,3}
    σ_1::T    # Process noise std for first component
    σ_2::T    # Process noise std for second component
    μ0::SVector{2,T}  # Initial state mean
    Σ0::Diagonal{T,SVector{2,T}}  # Initial state covariance
end

function RandomWalkPoissonModel(;
    σ_1::T=0.01,
    σ_2::T=0.1,
    μ0::SVector{2,T}=SVector{2,T}(1.0, 2.0),
    Σ0_diag::SVector{2,T}=SVector{2,T}(1.0, 1.0),
) where {T}
    Σ0 = Diagonal(Σ0_diag)
    return RandomWalkPoissonModel{T}(σ_1, σ_2, μ0, Σ0)
end

# =============================================================================
# Configuration Methods
# =============================================================================

function dynamics_covariance(model::RandomWalkPoissonModel{T}) where {T}
    return Diagonal(SVector{2,T}(model.σ_1^2, model.σ_2^2))
end

function dynamics_covariance_inv(model::RandomWalkPoissonModel{T}) where {T}
    return Diagonal(SVector{2,T}(1 / model.σ_1^2, 1 / model.σ_2^2))
end

function initial_prior(model::RandomWalkPoissonModel)
    return MvNormal(model.μ0, model.Σ0)
end

function observation_family(::RandomWalkPoissonModel)
    return ProductExponentialFamily{1}(PoissonNatural())
end

# =============================================================================
# Dynamics: f(x, θ) = x (identity / random walk)
# =============================================================================

function f(::RandomWalkPoissonModel{T}, x::SVector{2,T}, θ::SVector{3,T}) where {T}
    return x
end

"""
Jacobian ∂f/∂x = I (identity matrix).
"""
function Df_x(::RandomWalkPoissonModel{T}, x::SVector{2,T}, θ::SVector{3,T}) where {T}
    return @SMatrix [
        one(T) zero(T)
        zero(T) one(T)
    ]
end

"""
Jacobian ∂f/∂θ = 0 (dynamics don't depend on parameters).
"""
function Df_θ(::RandomWalkPoissonModel{T}, x::SVector{2,T}, θ::SVector{3,T}) where {T}
    return @SMatrix zeros(T, 2, 3)
end

"""
∂(Df_x[:, d])/∂x = 0 (f is linear in x).
"""
function D2f_xx(::RandomWalkPoissonModel{T}, x::SVector{2,T}, θ::SVector{3,T}) where {T}
    Z = @SMatrix zeros(T, 2, 2)
    return (Z, Z)
end

"""
∂(Df_x[:, d])/∂θ = 0 (Jacobian doesn't depend on θ).
"""
function D2f_xθ(::RandomWalkPoissonModel{T}, x::SVector{2,T}, θ::SVector{3,T}) where {T}
    Z = @SMatrix zeros(T, 2, 3)
    return (Z, Z)
end

"""
∂²f/∂θ_p∂θ = 0 (f doesn't depend on θ).
"""
function D2f_θθ(::RandomWalkPoissonModel{T}, x::SVector{2,T}, θ::SVector{3,T}) where {T}
    Z = @SMatrix zeros(T, 2, 3)
    return (Z, Z, Z)
end

"""
∂(Df_x)/∂θ_p = 0 (state Jacobian doesn't depend on θ).
"""
function DDf_x_θ(::RandomWalkPoissonModel{T}, x::SVector{2,T}, θ::SVector{3,T}) where {T}
    Z = @SMatrix zeros(T, 2, 2)
    return (Z, Z, Z)
end

# =============================================================================
# Observations: η(x, θ) = a_1·x_1 + a_2·x_2 + b
# =============================================================================

"""
Natural parameter function. η = a_1·x_1 + a_2·x_2 + b
Returns as 1-element vector since observation is scalar Poisson.
"""
function η(::RandomWalkPoissonModel{T}, x::SVector{2,T}, θ::SVector{3,T}) where {T}
    a_1, a_2, b = θ
    x_1, x_2 = x
    return SVector{1,T}(a_1 * x_1 + a_2 * x_2 + b)
end

"""
Jacobian ∂η/∂x (1×2 matrix).
∂η/∂x = [a_1, a_2]
"""
function Dη_x(::RandomWalkPoissonModel{T}, x::SVector{2,T}, θ::SVector{3,T}) where {T}
    a_1, a_2, _ = θ
    return @SMatrix [a_1 a_2]
end

"""
Jacobian ∂η/∂θ (1×3 matrix).
∂η/∂θ = [x_1, x_2, 1]
"""
function Dη_θ(::RandomWalkPoissonModel{T}, x::SVector{2,T}, θ::SVector{3,T}) where {T}
    x_1, x_2 = x
    return @SMatrix [x_1 x_2 one(T)]
end

"""
∂(Dη_x[:, d])/∂x for d = 1, 2.
η is linear in x, so all zeros.
Returns 2 matrices (one per state dim), each 1×2.
"""
function D2η_xx(::RandomWalkPoissonModel{T}, x::SVector{2,T}, θ::SVector{3,T}) where {T}
    Z = @SMatrix zeros(T, 1, 2)
    return (Z, Z)
end

"""
∂(Dη_x[:, d])/∂θ for d = 1, 2.
Since Dη_x = [a_1, a_2] and this depends on θ:
- ∂(Dη_x[:, 1])/∂θ = ∂a_1/∂θ = [1, 0, 0]
- ∂(Dη_x[:, 2])/∂θ = ∂a_2/∂θ = [0, 1, 0]
Returns 2 matrices (one per state dim), each 1×3.
"""
function D2η_xθ(::RandomWalkPoissonModel{T}, x::SVector{2,T}, θ::SVector{3,T}) where {T}
    Hη_θ_1 = @SMatrix [one(T) zero(T) zero(T)]
    Hη_θ_2 = @SMatrix [zero(T) one(T) zero(T)]
    return (Hη_θ_1, Hη_θ_2)
end

"""
∂²η/∂θ_p∂θ for p = 1, 2, 3.
η is linear in θ, so all zeros.
Returns 3 matrices (one per param), each 1×3.
"""
function D2η_θθ(::RandomWalkPoissonModel{T}, x::SVector{2,T}, θ::SVector{3,T}) where {T}
    Z = @SMatrix zeros(T, 1, 3)
    return (Z, Z, Z)
end

"""
∂(Dη_x)/∂θ_p for p = 1, 2, 3.
Dη_x = [a_1, a_2], so:
- ∂Dη_x/∂a_1 = [1, 0]
- ∂Dη_x/∂a_2 = [0, 1]
- ∂Dη_x/∂b = [0, 0]
Returns 3 matrices (one per param), each 1×2.
"""
function DDη_x_θ(::RandomWalkPoissonModel{T}, x::SVector{2,T}, θ::SVector{3,T}) where {T}
    dDη_x_da1 = @SMatrix [one(T) zero(T)]
    dDη_x_da2 = @SMatrix [zero(T) one(T)]
    dDη_x_db = @SMatrix [zero(T) zero(T)]
    return (dDη_x_da1, dDη_x_da2, dDη_x_db)
end
