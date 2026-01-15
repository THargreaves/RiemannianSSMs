export PoissonMultiChannelParam
export h_param, calc_Jh_param, calc_Hhs_param, calc_R_param, calc_Rinv_param
export calc_h_θ, calc_Hh_θs, calc_Jh_θ, calc_h_θθ

"""
Multi-channel Poisson observation model with scaled log-linear predictor.

Observations are M-dimensional counts: y_t^{(m)} ~ Poisson(λ_t^{(m)})
where λ_t^{(m)} = exp(η_t^{(m)}) and the linear predictor is:

    η_t^{(m)} = c_m · s(θ) · (A_m · x_t) + b_m

with s(θ) = exp(θ) (single scalar parameter).

The parameter θ controls the overall "gain" of the observation, creating
coupling between state and parameter that produces challenging geometry.

Fields:
- A: M×D projection matrix (each row selects/mixes state components)
- c: M-vector of per-channel scale coefficients
- b: M-vector of per-channel offset coefficients

For the FHN + Poisson test case (M=2, D=2):
    η₁ = 1.0 · exp(θ) · x₁ + b₁
    η₂ = 0.6 · exp(θ) · x₂ + b₂

corresponds to A = I₂, c = [1.0, 0.6], b = [b₁, b₂].
"""
struct PoissonMultiChannelParam{T,M,D,MD}
    A::SMatrix{M,D,T,MD}  # Projection matrix
    c::SVector{M,T}       # Per-channel scales
    b::SVector{M,T}       # Per-channel offsets
end

function PoissonMultiChannelParam(
    A::SMatrix{M,D,T}, c::SVector{M,T}, b::SVector{M,T}
) where {T,M,D}
    return PoissonMultiChannelParam{T,M,D,M * D}(A, c, b)
end

"""
Observation function h(z, θ) returns the linear predictor η (M-vector).
η_m = c_m · exp(θ) · (A_m · x) + b_m
"""
function h_param(
    model::PoissonMultiChannelParam{T,M,D}, z::SVector{D,T}, θ::SVector{1,T}
) where {T,M,D}
    s = exp(θ[1])
    # η = diag(c) * A * z * s + b
    η = model.c .* (model.A * z) .* s .+ model.b
    return η
end

"""
Jacobian ∂η/∂x (M×D matrix).
∂η_m/∂x_d = c_m · exp(θ) · A[m, d]
"""
function calc_Jh_param(
    model::PoissonMultiChannelParam{T,M,D}, z::SVector{D,T}, θ::SVector{1,T}
) where {T,M,D}
    s = exp(θ[1])
    # Jh = s * diag(c) * A
    Jh = s .* (model.c .* model.A)
    return Jh
end

"""
Parameter Jacobian ∂η/∂θ (M×1 matrix).
∂η_m/∂θ = c_m · exp(θ) · (A_m · x) = η_m - b_m
"""
function calc_h_θ(
    model::PoissonMultiChannelParam{T,M,D}, z::SVector{D,T}, θ::SVector{1,T}
) where {T,M,D}
    s = exp(θ[1])
    # ∂η/∂θ = c .* (A * z) .* s = η - b
    dη_dθ = model.c .* (model.A * z) .* s
    return hcat(dη_dθ)  # Convert to M×1 matrix
end

"""
Hessians ∂²η/∂x_d∂x (D matrices, each M×D).
Since η is linear in x, all second derivatives are zero.
"""
function calc_Hhs_param(
    model::PoissonMultiChannelParam{T,M,D}, z::SVector{D,T}, θ::SVector{1,T}
) where {T,M,D}
    Z = @SMatrix zeros(T, M, D)
    return ntuple(_ -> Z, D)
end

"""
Mixed Hessians ∂²η/∂x_d∂θ (D matrices, each M×1).
∂Jh[m, d]/∂θ = c_m · exp(θ) · A[m, d] = Jh[m, d]
"""
function calc_Hh_θs(
    model::PoissonMultiChannelParam{T,M,D}, z::SVector{D,T}, θ::SVector{1,T}
) where {T,M,D}
    s = exp(θ[1])
    # For each state dimension d, return ∂Jh[:, d]/∂θ as M×1 matrix
    return ntuple(D) do d
        # Column d of Jh, reshaped as M×1
        col_d = s .* model.c .* model.A[:, d]
        hcat(col_d)
    end
end

"""
Jacobian derivatives ∂Jh/∂θ^{(p)} (P=1 matrices, each M×D).
Since Jh = exp(θ) · diag(c) · A, we have ∂Jh/∂θ = Jh.
"""
function calc_Jh_θ(
    model::PoissonMultiChannelParam{T,M,D}, z::SVector{D,T}, θ::SVector{1,T}
) where {T,M,D}
    Jh = calc_Jh_param(model, z, θ)
    return (Jh,)  # Single-element tuple for P=1
end

"""
Second parameter derivatives ∂²η/∂θ²  (P=1 matrices, each M×1).
Since h_θ = (η - b) and ∂exp(θ)/∂θ = exp(θ), we have ∂h_θ/∂θ = h_θ.
"""
function calc_h_θθ(
    model::PoissonMultiChannelParam{T,M,D}, z::SVector{D,T}, θ::SVector{1,T}
) where {T,M,D}
    h_θ = calc_h_θ(model, z, θ)
    return (h_θ,)  # Single-element tuple for P=1
end

"""
Weight matrix W_t = diag(λ) for the generalized GN metric.
λ_m = exp(η_m) is state and parameter dependent.
Returns M×M diagonal matrix.
"""
function calc_Rinv_param(
    model::PoissonMultiChannelParam{T,M,D}, z::SVector{D,T}, θ::SVector{1,T}
) where {T,M,D}
    η = h_param(model, z, θ)
    λ = exp.(η)
    return Diagonal(λ)
end

"""
Reciprocal of weight matrix (for compatibility).
"""
function calc_R_param(
    model::PoissonMultiChannelParam{T,M,D}, z::SVector{D,T}, θ::SVector{1,T}
) where {T,M,D}
    η = h_param(model, z, θ)
    λ = exp.(η)
    return Diagonal(one(T) ./ λ)
end
