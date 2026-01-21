"""
Log-Gaussian Cox Process with linear AR(p) latent dynamics.

This is a simplified version without nonlinear self-regulation, where the
GN approximation equals the exact observed Hessian for the transition term.

State: z ∈ R^p representing [x_t, x_{t-1}, ..., x_{t-p+1}] (current and lagged log-intensities)
Parameters: θ = [φ₁, ..., φₚ, μ, b] where
    - φᵢ: AR coefficients (unconstrained)
    - μ: drift/long-run mean
    - b: observation intercept

Dynamics (linear AR(p)):
    x_{t+1} = μ + Σᵢ φᵢ (x_{t-i+1} - μ) + σ ε_t
            = μ(1 - Σφ) + Σᵢ φᵢ z[i] + σ ε_t

In state-space form with z_t = [x_t, ..., x_{t-p+1}]:
    z_{t+1}[1] = μ(1 - Σφᵢ) + Σᵢ φᵢ z_t[i] + σ ε_t
    z_{t+1}[j] = z_t[j-1] for j = 2, ..., p

Observations (Poisson with log-linear predictor):
    y_t | z_t ~ Poisson(exp(η_t)) where η_t = b + z_t[1]

Process noise: Q = diag(σ², ε², ..., ε²) with small ε for regularization.

Why this is challenging for HMC:
- Poisson Fisher information exp(η) creates state-dependent observation curvature
- Near-persistent dynamics (Σφᵢ ≈ 1) create long-range correlations
- GN = exact observed Hessian for dynamics (no approximation error)
"""

export LGCPARpLinearModel

struct LGCPARpLinearModel{P,Dp,T} <: StateSpaceModel{P,1,Dp}
    σ::T                            # Process noise std (main component)
    ε::T                            # Regularization noise std (lag components)
    μ0::SVector{P,T}                # Initial state mean
    Σ0::Diagonal{T,SVector{P,T}}    # Initial state covariance

    function LGCPARpLinearModel{P,Dp,T}(σ, ε, μ0, Σ0) where {P,Dp,T}
        @assert Dp == P + 2 "Parameter dimension Dp must equal P + 2"
        return new{P,Dp,T}(σ, ε, μ0, Σ0)
    end
end

"""
    LGCPARpLinearModel{P}(; σ=0.1, ε=1e-6, s0=1.0)

Construct a linear LGCP AR(p) model.

# Type Parameter
- `P`: AR order (number of lags)

# Keyword Arguments
- `σ`: Process noise std for the main (first) state component (default: 0.1)
- `ε`: Regularization noise std for lag components (default: 1e-6)
- `s0`: Initial state prior std (default: 1.0), so Σ0 = s0² I
"""
function LGCPARpLinearModel{P}(; σ=0.1, ε=1e-6, s0=1.0) where {P}
    T = typeof(σ)
    μ0 = zeros(SVector{P,T})
    Σ0_diag = SVector{P,T}(ntuple(_ -> T(s0)^2, Val(P)))
    Σ0 = Diagonal(Σ0_diag)
    Dp = P + 2
    return LGCPARpLinearModel{P,Dp,T}(T(σ), T(ε), μ0, Σ0)
end

function LGCPARpLinearModel(; P::Int, σ=0.1, ε=1e-6, s0=1.0)
    return LGCPARpLinearModel{P}(; σ=σ, ε=ε, s0=s0)
end

ar_order(::LGCPARpLinearModel{P}) where {P} = P

# =============================================================================
# Configuration Methods
# =============================================================================

function dynamics_covariance(model::LGCPARpLinearModel{P,Dp,T}) where {P,Dp,T}
    diag_vals = SVector{P,T}(ntuple(i -> i == 1 ? model.σ^2 : model.ε^2, Val(P)))
    return Diagonal(diag_vals)
end

function dynamics_covariance_inv(model::LGCPARpLinearModel{P,Dp,T}) where {P,Dp,T}
    diag_vals = SVector{P,T}(ntuple(i -> i == 1 ? 1 / model.σ^2 : 1 / model.ε^2, Val(P)))
    return Diagonal(diag_vals)
end

function initial_prior(model::LGCPARpLinearModel)
    return MvNormal(model.μ0, model.Σ0)
end

function observation_family(::LGCPARpLinearModel)
    return ProductExponentialFamily{1}(PoissonNatural())
end

# =============================================================================
# Parameter indexing helpers
# θ = [φ₁, ..., φₚ, μ, b]
# =============================================================================

@inline function unpack_params(
    ::LGCPARpLinearModel{P,Dp,T}, θ::SVector{N,T}
) where {P,Dp,T,N}
    φ = SVector{P,T}(ntuple(i -> θ[i], Val(P)))
    μ = θ[P + 1]
    b = θ[P + 2]
    return φ, μ, b
end

# =============================================================================
# Dynamics: f(z, θ) - LINEAR
# z_{t+1}[1] = μ(1 - Σφᵢ) + Σᵢ φᵢ z[i]
# z_{t+1}[j] = z[j-1] for j = 2, ..., P
# =============================================================================

function f(
    model::LGCPARpLinearModel{P,Dp,T}, z::SVector{P,T}, θ::SVector{N,T}
) where {P,Dp,T,N}
    φ, μ, _ = unpack_params(model, θ)

    sum_φ = sum(φ)
    ar_term = sum(φ[i] * z[i] for i in 1:P)
    f1 = μ * (1 - sum_φ) + ar_term

    return SVector{P,T}(ntuple(j -> j == 1 ? f1 : z[j - 1], Val(P)))
end

"""
Jacobian ∂f/∂z (P × P matrix).

Df_x[1, i] = φᵢ
Df_x[j, j-1] = 1 for j = 2, ..., P
All other entries = 0

This is the AR companion matrix (constant, doesn't depend on z).
"""
function Df_x(
    model::LGCPARpLinearModel{P,Dp,T}, z::SVector{P,T}, θ::SVector{N,T}
) where {P,Dp,T,N}
    φ, _, _ = unpack_params(model, θ)

    return SMatrix{P,P,T}(
        ntuple(Val(P * P)) do idx
            i = mod1(idx, P)
            j = div(idx - 1, P) + 1
            if i == 1
                φ[j]
            elseif i == j + 1
                one(T)
            else
                zero(T)
            end
        end,
    )
end

"""
Jacobian ∂f/∂θ (P × (P+2) matrix).
θ = [φ₁, ..., φₚ, μ, b]

∂f[1]/∂φᵢ = z[i] - μ
∂f[1]/∂μ = 1 - Σφᵢ
∂f[1]/∂b = 0
∂f[j]/∂θ = 0 for j ≥ 2
"""
function Df_θ(
    model::LGCPARpLinearModel{P,Dp,T}, z::SVector{P,T}, θ::SVector{N,T}
) where {P,Dp,T,N}
    φ, μ, _ = unpack_params(model, θ)
    sum_φ = sum(φ)

    return SMatrix{P,P + 2,T}(
        ntuple(Val(P * (P + 2))) do idx
            i = mod1(idx, P)
            j = div(idx - 1, P) + 1
            if i == 1
                if j <= P
                    z[j] - μ
                elseif j == P + 1
                    1 - sum_φ
                else
                    zero(T)
                end
            else
                zero(T)
            end
        end,
    )
end

"""
∂(Df_x[:, d])/∂z for d = 1, ..., P.

f is linear in z, so all second derivatives are zero.
"""
function D2f_xx(
    model::LGCPARpLinearModel{P,Dp,T}, z::SVector{P,T}, θ::SVector{N,T}
) where {P,Dp,T,N}
    Z = @SMatrix zeros(T, P, P)
    return ntuple(_ -> Z, Val(P))
end

"""
∂(Df_x[:, d])/∂θ for d = 1, ..., P.

Df_x[:, d] = [φ_d, 0, ..., 1 at row d+1 (if d < P), ..., 0]ᵀ
∂(Df_x[:, d])/∂φ_d = 1 at row 1

Returns P matrices, each P × (P+2).
"""
function D2f_xθ(
    model::LGCPARpLinearModel{P,Dp,T}, z::SVector{P,T}, θ::SVector{N,T}
) where {P,Dp,T,N}
    return ntuple(Val(P)) do d
        SMatrix{P,P + 2,T}(
            ntuple(Val(P * (P + 2))) do idx
                i = mod1(idx, P)
                j = div(idx - 1, P) + 1
                (i == 1 && j == d) ? one(T) : zero(T)
            end,
        )
    end
end

"""
∂²f/∂θₖ∂θ for k = 1, ..., P+2.

∂f[1]/∂φᵢ = z[i] - μ → ∂²f[1]/∂φᵢ∂μ = -1
∂f[1]/∂μ = 1 - Σφ → ∂²f[1]/∂μ∂φᵢ = -1

Returns (P+2) matrices, each P × (P+2).
"""
function D2f_θθ(
    model::LGCPARpLinearModel{P,Dp,T}, z::SVector{P,T}, θ::SVector{N,T}
) where {P,Dp,T,N}
    return ntuple(Val(P + 2)) do k
        SMatrix{P,P + 2,T}(
            ntuple(Val(P * (P + 2))) do idx
                i = mod1(idx, P)
                j = div(idx - 1, P) + 1
                if i == 1
                    if k <= P && j == P + 1
                        -one(T)
                    elseif k == P + 1 && j <= P
                        -one(T)
                    else
                        zero(T)
                    end
                else
                    zero(T)
                end
            end,
        )
    end
end

"""
∂(Df_x)/∂θₖ for k = 1, ..., P+2.

Df_x[1, i] = φᵢ (doesn't depend on z)
∂(Df_x)/∂φₖ has 1 at [1, k]
∂(Df_x)/∂μ = 0, ∂(Df_x)/∂b = 0

Returns (P+2) matrices, each P × P.
"""
function DDf_x_θ(
    model::LGCPARpLinearModel{P,Dp,T}, z::SVector{P,T}, θ::SVector{N,T}
) where {P,Dp,T,N}
    return ntuple(Val(P + 2)) do k
        SMatrix{P,P,T}(
            ntuple(Val(P * P)) do idx
                i = mod1(idx, P)
                j = div(idx - 1, P) + 1
                (i == 1 && k <= P && j == k) ? one(T) : zero(T)
            end,
        )
    end
end

# =============================================================================
# Observations: η(z, θ) = b + z[1]
# =============================================================================

"""
Natural parameter function. η = b + z[1] where b = θ[P+2].
"""
function η(
    model::LGCPARpLinearModel{P,Dp,T}, z::SVector{P,T}, θ::SVector{N,T}
) where {P,Dp,T,N}
    b = θ[P + 2]
    return SVector{1,T}(b + z[1])
end

"""
Jacobian ∂η/∂z (1 × P matrix).
∂η/∂z = [1, 0, ..., 0]
"""
function Dη_x(
    model::LGCPARpLinearModel{P,Dp,T}, z::SVector{P,T}, θ::SVector{N,T}
) where {P,Dp,T,N}
    return SMatrix{1,P,T}(ntuple(j -> j == 1 ? one(T) : zero(T), Val(P)))
end

"""
Jacobian ∂η/∂θ (1 × (P+2) matrix).
∂η/∂θ = [0, ..., 0, 0, 1]
"""
function Dη_θ(
    model::LGCPARpLinearModel{P,Dp,T}, z::SVector{P,T}, θ::SVector{N,T}
) where {P,Dp,T,N}
    return SMatrix{1,P + 2,T}(ntuple(j -> j == P + 2 ? one(T) : zero(T), Val(P + 2)))
end

"""
All second derivatives of η w.r.t. z are zero (η is linear in z).
"""
function D2η_xx(
    model::LGCPARpLinearModel{P,Dp,T}, z::SVector{P,T}, θ::SVector{N,T}
) where {P,Dp,T,N}
    Z = @SMatrix zeros(T, 1, P)
    return ntuple(_ -> Z, Val(P))
end

"""
Dη_x doesn't depend on θ.
"""
function D2η_xθ(
    model::LGCPARpLinearModel{P,Dp,T}, z::SVector{P,T}, θ::SVector{N,T}
) where {P,Dp,T,N}
    Z = @SMatrix zeros(T, 1, P + 2)
    return ntuple(_ -> Z, Val(P))
end

"""
η is linear in θ, so all second derivatives are zero.
"""
function D2η_θθ(
    model::LGCPARpLinearModel{P,Dp,T}, z::SVector{P,T}, θ::SVector{N,T}
) where {P,Dp,T,N}
    Z = @SMatrix zeros(T, 1, P + 2)
    return ntuple(_ -> Z, Val(P + 2))
end

"""
Dη_x doesn't depend on θ.
"""
function DDη_x_θ(
    model::LGCPARpLinearModel{P,Dp,T}, z::SVector{P,T}, θ::SVector{N,T}
) where {P,Dp,T,N}
    Z = @SMatrix zeros(T, 1, P)
    return ntuple(_ -> Z, Val(P + 2))
end
