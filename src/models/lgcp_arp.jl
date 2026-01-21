"""
Log-Gaussian Cox Process with AR(p) latent dynamics and self-regulation.

This model is designed to create challenging geometry for RHMC benchmarking.

State: z ∈ R^p representing [x_t, x_{t-1}, ..., x_{t-p+1}] (current and lagged log-intensities)
Parameters: θ = [φ₁, ..., φₚ, μ, log(β), b] where
    - φᵢ: AR coefficients (unconstrained)
    - μ: drift/long-run mean
    - β > 0: self-regulation strength (log-parameterized)
    - b: observation intercept

Dynamics (AR(p) with self-regulation):
    x_{t+1} = μ + Σᵢ φᵢ (x_{t-i+1} - μ) - β exp(x_t) + σ ε_t

In state-space form with z_t = [x_t, ..., x_{t-p+1}]:
    z_{t+1}[1] = μ(1 - Σφᵢ) + Σᵢ φᵢ z_t[i] - β exp(z_t[1]) + σ ε_t
    z_{t+1}[j] = z_t[j-1] for j = 2, ..., p

Observations (Poisson with log-linear predictor):
    y_t | z_t ~ Poisson(exp(η_t)) where η_t = b + z_t[1]

Process noise: Q = diag(σ², ε², ..., ε²) with small ε for regularization.

Why this is hard for HMC:
- Self-regulation term -β exp(z) creates state-dependent dynamics Jacobian
- Poisson Fisher information exp(η) creates state-dependent observation curvature
- Both scale with exp(z), creating correlated curvature changes
- Near-persistent dynamics (Σφᵢ ≈ 1) create long-range correlations
"""

export LGCPARpModel, ar_order

struct LGCPARpModel{P,Dp,T} <: StateSpaceModel{P,1,Dp}
    σ::T                            # Process noise std (main component)
    ε::T                            # Regularization noise std (lag components)
    μ0::SVector{P,T}                # Initial state mean
    Σ0::Diagonal{T,SVector{P,T}}    # Initial state covariance

    function LGCPARpModel{P,Dp,T}(σ, ε, μ0, Σ0) where {P,Dp,T}
        @assert Dp == P + 3 "Parameter dimension Dp must equal P + 3"
        return new{P,Dp,T}(σ, ε, μ0, Σ0)
    end
end

"""
    LGCPARpModel{P}(; σ=0.1, ε=1e-6, s0=1.0)

Construct an LGCP AR(p) model.

# Type Parameter
- `P`: AR order (number of lags)

# Keyword Arguments
- `σ`: Process noise std for the main (first) state component (default: 0.1)
- `ε`: Regularization noise std for lag components (default: 1e-6)
- `s0`: Initial state prior std (default: 1.0), so Σ0 = s0² I
"""
function LGCPARpModel{P}(; σ=0.1, ε=1e-6, s0=1.0) where {P}
    T = typeof(σ)
    μ0 = zeros(SVector{P,T})
    Σ0_diag = SVector{P,T}(ntuple(_ -> T(s0)^2, Val(P)))
    Σ0 = Diagonal(Σ0_diag)
    Dp = P + 3
    return LGCPARpModel{P,Dp,T}(T(σ), T(ε), μ0, Σ0)
end

# Convenience constructor with explicit P
function LGCPARpModel(; P::Int, σ=0.1, ε=1e-6, s0=1.0)
    return LGCPARpModel{P}(; σ=σ, ε=ε, s0=s0)
end

ar_order(::LGCPARpModel{P}) where {P} = P

# =============================================================================
# Configuration Methods
# =============================================================================

function dynamics_covariance(model::LGCPARpModel{P,Dp,T}) where {P,Dp,T}
    diag_vals = SVector{P,T}(ntuple(i -> i == 1 ? model.σ^2 : model.ε^2, Val(P)))
    return Diagonal(diag_vals)
end

function dynamics_covariance_inv(model::LGCPARpModel{P,Dp,T}) where {P,Dp,T}
    diag_vals = SVector{P,T}(ntuple(i -> i == 1 ? 1 / model.σ^2 : 1 / model.ε^2, Val(P)))
    return Diagonal(diag_vals)
end

function initial_prior(model::LGCPARpModel)
    return MvNormal(model.μ0, model.Σ0)
end

function observation_family(::LGCPARpModel)
    return ProductExponentialFamily{1}(PoissonNatural())
end

# =============================================================================
# Parameter indexing helpers
# θ = [φ₁, ..., φₚ, μ, log(β), b]
# =============================================================================

@inline function unpack_params(::LGCPARpModel{P,Dp,T}, θ::SVector{N,T}) where {P,Dp,T,N}
    φ = SVector{P,T}(ntuple(i -> θ[i], Val(P)))
    μ = θ[P + 1]
    β = exp(θ[P + 2])
    b = θ[P + 3]
    return φ, μ, β, b
end

# =============================================================================
# Dynamics: f(z, θ)
# z_{t+1}[1] = μ(1 - Σφᵢ) + Σᵢ φᵢ z[i] - β exp(z[1])
# z_{t+1}[j] = z[j-1] for j = 2, ..., P
# =============================================================================

function f(model::LGCPARpModel{P,Dp,T}, z::SVector{P,T}, θ::SVector{N,T}) where {P,Dp,T,N}
    φ, μ, β, _ = unpack_params(model, θ)

    # First component: AR(p) + self-regulation
    sum_φ = sum(φ)
    ar_term = sum(φ[i] * z[i] for i in 1:P)
    f1 = μ * (1 - sum_φ) + ar_term - β * exp(z[1])

    # Remaining components: shift down
    return SVector{P,T}(ntuple(j -> j == 1 ? f1 : z[j - 1], Val(P)))
end

"""
Jacobian ∂f/∂z (P × P matrix).

Df_x[1, i] = φᵢ - β exp(z[1]) δ_{i,1}
Df_x[j, j-1] = 1 for j = 2, ..., P
All other entries = 0
"""
function Df_x(
    model::LGCPARpModel{P,Dp,T}, z::SVector{P,T}, θ::SVector{N,T}
) where {P,Dp,T,N}
    φ, _, β, _ = unpack_params(model, θ)
    exp_z1 = exp(z[1])

    return SMatrix{P,P,T}(
        ntuple(Val(P * P)) do idx
            i = mod1(idx, P)  # row
            j = div(idx - 1, P) + 1  # col
            if i == 1
                j == 1 ? φ[1] - β * exp_z1 : φ[j]
            elseif i == j + 1
                one(T)
            else
                zero(T)
            end
        end,
    )
end

"""
Jacobian ∂f/∂θ (P × (P+3) matrix).
θ = [φ₁, ..., φₚ, μ, log(β), b]

∂f[1]/∂φᵢ = z[i] - μ
∂f[1]/∂μ = 1 - Σφᵢ
∂f[1]/∂log(β) = -β exp(z[1])
∂f[1]/∂b = 0
∂f[j]/∂θ = 0 for j ≥ 2
"""
function Df_θ(
    model::LGCPARpModel{P,Dp,T}, z::SVector{P,T}, θ::SVector{N,T}
) where {P,Dp,T,N}
    φ, μ, β, _ = unpack_params(model, θ)
    exp_z1 = exp(z[1])
    sum_φ = sum(φ)

    return SMatrix{P,P + 3,T}(
        ntuple(Val(P * (P + 3))) do idx
            i = mod1(idx, P)  # row
            j = div(idx - 1, P) + 1  # col (parameter index)
            if i == 1
                if j <= P
                    # ∂f[1]/∂φⱼ = z[j] - μ
                    z[j] - μ
                elseif j == P + 1
                    # ∂f[1]/∂μ = 1 - Σφᵢ
                    1 - sum_φ
                elseif j == P + 2
                    # ∂f[1]/∂log(β) = -β exp(z[1])
                    -β * exp_z1
                else
                    # ∂f[1]/∂b = 0
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

Only f[1] has nonlinear dependence on z (via exp(z[1])).
∂²f[1]/∂z[1]² = -β exp(z[1])
All other second derivatives = 0.

Returns P matrices, each P × P.
"""
function D2f_xx(
    model::LGCPARpModel{P,Dp,T}, z::SVector{P,T}, θ::SVector{N,T}
) where {P,Dp,T,N}
    _, _, β, _ = unpack_params(model, θ)
    exp_z1 = exp(z[1])
    val = -β * exp_z1

    return ntuple(Val(P)) do d
        SMatrix{P,P,T}(
            ntuple(Val(P * P)) do idx
                i = mod1(idx, P)
                j = div(idx - 1, P) + 1
                (d == 1 && i == 1 && j == 1) ? val : zero(T)
            end,
        )
    end
end

"""
∂(Df_x[:, d])/∂θ for d = 1, ..., P.

Df_x[:, 1] = [φ₁ - β exp(z[1]), 1, 0, ..., 0]ᵀ
∂(Df_x[:, 1])/∂φ₁ = 1 at row 1
∂(Df_x[:, 1])/∂log(β) = -β exp(z[1]) at row 1

Df_x[:, d] = [φ_d, 0, ..., 1 at row d+1, ..., 0]ᵀ for d ≥ 2
∂(Df_x[:, d])/∂φ_d = 1 at row 1

Returns P matrices, each P × (P+3).
"""
function D2f_xθ(
    model::LGCPARpModel{P,Dp,T}, z::SVector{P,T}, θ::SVector{N,T}
) where {P,Dp,T,N}
    _, _, β, _ = unpack_params(model, θ)
    exp_z1 = exp(z[1])

    return ntuple(Val(P)) do d
        SMatrix{P,P + 3,T}(
            ntuple(Val(P * (P + 3))) do idx
                i = mod1(idx, P)  # row
                j = div(idx - 1, P) + 1  # col (parameter)
                if i == 1
                    if j == d
                        # ∂(Df_x[1, d])/∂φ_d = 1
                        one(T)
                    elseif d == 1 && j == P + 2
                        # ∂(Df_x[1, 1])/∂log(β) = -β exp(z[1])
                        -β * exp_z1
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
∂²f/∂θₖ∂θ for k = 1, ..., P+3.

∂f[1]/∂φᵢ = z[i] - μ → ∂²f[1]/∂φᵢ∂μ = -1
∂f[1]/∂μ = 1 - Σφ → ∂²f[1]/∂μ∂φᵢ = -1
∂f[1]/∂log(β) = -β exp(z[1]) → ∂²f[1]/∂log(β)² = -β exp(z[1])

Returns (P+3) matrices, each P × (P+3).
"""
function D2f_θθ(
    model::LGCPARpModel{P,Dp,T}, z::SVector{P,T}, θ::SVector{N,T}
) where {P,Dp,T,N}
    _, _, β, _ = unpack_params(model, θ)
    exp_z1 = exp(z[1])

    return ntuple(Val(P + 3)) do k
        SMatrix{P,P + 3,T}(
            ntuple(Val(P * (P + 3))) do idx
                i = mod1(idx, P)  # row
                j = div(idx - 1, P) + 1  # col
                if i == 1
                    if k <= P && j == P + 1
                        # ∂²f[1]/∂φₖ∂μ = -1
                        -one(T)
                    elseif k == P + 1 && j <= P
                        # ∂²f[1]/∂μ∂φⱼ = -1
                        -one(T)
                    elseif k == P + 2 && j == P + 2
                        # ∂²f[1]/∂log(β)² = -β exp(z[1])
                        -β * exp_z1
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
∂(Df_x)/∂θₖ for k = 1, ..., P+3.

Df_x[1, i] = φᵢ - β exp(z[1]) δ_{i,1}
∂(Df_x)/∂φₖ has 1 at [1, k]
∂(Df_x)/∂log(β) has -β exp(z[1]) at [1, 1]
∂(Df_x)/∂μ = 0, ∂(Df_x)/∂b = 0

Returns (P+3) matrices, each P × P.
"""
function DDf_x_θ(
    model::LGCPARpModel{P,Dp,T}, z::SVector{P,T}, θ::SVector{N,T}
) where {P,Dp,T,N}
    _, _, β, _ = unpack_params(model, θ)
    exp_z1 = exp(z[1])

    return ntuple(Val(P + 3)) do k
        SMatrix{P,P,T}(
            ntuple(Val(P * P)) do idx
                i = mod1(idx, P)  # row
                j = div(idx - 1, P) + 1  # col
                if i == 1
                    if k <= P && j == k
                        # ∂(Df_x[1, k])/∂φₖ = 1
                        one(T)
                    elseif k == P + 2 && j == 1
                        # ∂(Df_x[1, 1])/∂log(β) = -β exp(z[1])
                        -β * exp_z1
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

# =============================================================================
# Observations: η(z, θ) = b + z[1]
# =============================================================================

"""
Natural parameter function. η = b + z[1] where b = θ[P+3].
Returns as 1-element vector for Poisson observation.
"""
function η(model::LGCPARpModel{P,Dp,T}, z::SVector{P,T}, θ::SVector{N,T}) where {P,Dp,T,N}
    b = θ[P + 3]
    return SVector{1,T}(b + z[1])
end

"""
Jacobian ∂η/∂z (1 × P matrix).
∂η/∂z = [1, 0, ..., 0]
"""
function Dη_x(
    model::LGCPARpModel{P,Dp,T}, z::SVector{P,T}, θ::SVector{N,T}
) where {P,Dp,T,N}
    return SMatrix{1,P,T}(ntuple(j -> j == 1 ? one(T) : zero(T), Val(P)))
end

"""
Jacobian ∂η/∂θ (1 × (P+3) matrix).
∂η/∂θ = [0, ..., 0, 0, 0, 1]
Only b contributes.
"""
function Dη_θ(
    model::LGCPARpModel{P,Dp,T}, z::SVector{P,T}, θ::SVector{N,T}
) where {P,Dp,T,N}
    return SMatrix{1,P + 3,T}(ntuple(j -> j == P + 3 ? one(T) : zero(T), Val(P + 3)))
end

"""
∂(Dη_x[:, d])/∂z for d = 1, ..., P.
η is linear in z, so all zeros.
Returns P matrices, each 1 × P.
"""
function D2η_xx(
    model::LGCPARpModel{P,Dp,T}, z::SVector{P,T}, θ::SVector{N,T}
) where {P,Dp,T,N}
    Z = @SMatrix zeros(T, 1, P)
    return ntuple(_ -> Z, Val(P))
end

"""
∂(Dη_x[:, d])/∂θ for d = 1, ..., P.
Dη_x = [1, 0, ..., 0] doesn't depend on θ.
Returns P matrices, each 1 × (P+3).
"""
function D2η_xθ(
    model::LGCPARpModel{P,Dp,T}, z::SVector{P,T}, θ::SVector{N,T}
) where {P,Dp,T,N}
    Z = @SMatrix zeros(T, 1, P + 3)
    return ntuple(_ -> Z, Val(P))
end

"""
∂²η/∂θₖ∂θ for k = 1, ..., P+3.
η is linear in θ, so all zeros.
Returns (P+3) matrices, each 1 × (P+3).
"""
function D2η_θθ(
    model::LGCPARpModel{P,Dp,T}, z::SVector{P,T}, θ::SVector{N,T}
) where {P,Dp,T,N}
    Z = @SMatrix zeros(T, 1, P + 3)
    return ntuple(_ -> Z, Val(P + 3))
end

"""
∂(Dη_x)/∂θₖ for k = 1, ..., P+3.
Dη_x doesn't depend on θ.
Returns (P+3) matrices, each 1 × P.
"""
function DDη_x_θ(
    model::LGCPARpModel{P,Dp,T}, z::SVector{P,T}, θ::SVector{N,T}
) where {P,Dp,T,N}
    Z = @SMatrix zeros(T, 1, P)
    return ntuple(_ -> Z, Val(P + 3))
end
