export PoissonObservationParam
export h_param, calc_Jh_param, calc_Hhs_param, calc_R_param, calc_Rinv_param
export calc_h_θ, calc_Hh_θs, calc_Jh_θ, calc_h_θθ

"""
Poisson observation model with log-linear predictor for RHMC benchmarking.

Observations are counts: y_t ~ Poisson(λ_t) where λ_t = exp(η_t)
and η_t = a_1·x_1 + a_2·x_2 + b is the linear predictor.

Parameters θ = [a_1, a_2, b] define the link between state and observation rate.

For the generalized Gauss-Newton approximation:
- Work with the linear predictor η rather than the rate λ
- Weight matrix W_t = λ_t (expected Hessian)
- Metric contribution: J_η^T · W_t · J_η where J_η = ∂η/∂x

Many derivatives are simple because η is linear in both x and θ.
"""
struct PoissonObservationParam{T}
    # No fields needed - parameters are passed as θ
end

function PoissonObservationParam()
    return PoissonObservationParam{Float64}()
end

"""
Observation function h(z, θ) returns the linear predictor η_t.
η_t = a_1·x_1 + a_2·x_2 + b

Returns a 1-element SVector for scalar observation.
"""
function h_param(
    model::PoissonObservationParam{T}, z::SVector{2,T}, θ::SVector{3,T}
) where {T}
    a_1, a_2, b = θ
    x_1, x_2 = z
    η = a_1 * x_1 + a_2 * x_2 + b
    return @SVector [η]
end

"""
Jacobian ∂η/∂x (1×2 matrix).
∂η/∂x = [a_1, a_2]
"""
function calc_Jh_param(
    model::PoissonObservationParam{T}, z::SVector{2,T}, θ::SVector{3,T}
) where {T}
    a_1, a_2, b = θ
    return @SMatrix [a_1 a_2]
end

"""
Parameter Jacobian ∂η/∂θ (1×3 matrix).
∂η/∂θ = [x_1, x_2, 1]
"""
function calc_h_θ(
    model::PoissonObservationParam{T}, z::SVector{2,T}, θ::SVector{3,T}
) where {T}
    x_1, x_2 = z
    return @SMatrix [x_1 x_2 one(T)]
end

"""
Hessians ∂²η/∂x_d∂x (D matrices, each 1×2).
Since η is linear in x, all second derivatives are zero.
"""
function calc_Hhs_param(
    model::PoissonObservationParam{T}, z::SVector{2,T}, θ::SVector{3,T}
) where {T}
    Z = @SMatrix zeros(T, 1, 2)
    return (Z, Z)
end

"""
Mixed Hessians ∂²η/∂x_d∂θ (D matrices, each 1×3).
Since η = a_1·x_1 + a_2·x_2 + b is bilinear:
∂²η/∂x_1∂θ = [1, 0, 0]
∂²η/∂x_2∂θ = [0, 1, 0]
"""
function calc_Hh_θs(
    model::PoissonObservationParam{T}, z::SVector{2,T}, θ::SVector{3,T}
) where {T}
    # ∂²η/∂x_1∂θ
    Hh_θ_1 = @SMatrix [one(T) zero(T) zero(T)]

    # ∂²η/∂x_2∂θ
    Hh_θ_2 = @SMatrix [zero(T) one(T) zero(T)]

    return (Hh_θ_1, Hh_θ_2)
end

"""
Jacobian derivatives ∂Jh/∂θ^{(p)} (P matrices, each 1×2).
Jh = [a_1, a_2], so:
∂Jh/∂a_1 = [1, 0]
∂Jh/∂a_2 = [0, 1]
∂Jh/∂b   = [0, 0]
"""
function calc_Jh_θ(
    model::PoissonObservationParam{T}, z::SVector{2,T}, θ::SVector{3,T}
) where {T}
    # ∂Jh/∂a_1
    dJh_da1 = @SMatrix [one(T) zero(T)]

    # ∂Jh/∂a_2
    dJh_da2 = @SMatrix [zero(T) one(T)]

    # ∂Jh/∂b
    dJh_db = @SMatrix [zero(T) zero(T)]

    return (dJh_da1, dJh_da2, dJh_db)
end

"""
Second parameter derivatives ∂²η/∂θ^{(p)}∂θ (P matrices, each 1×3).
Since η is linear in θ, all second derivatives w.r.t. θ are zero.
"""
function calc_h_θθ(
    model::PoissonObservationParam{T}, z::SVector{2,T}, θ::SVector{3,T}
) where {T}
    Z = @SMatrix zeros(T, 1, 3)
    return (Z, Z, Z)
end

"""
Weight matrix W_t = λ_t = exp(η_t) for the generalized GN metric.

IMPORTANT: For Poisson, this is state and parameter dependent!
Returns Diagonal([λ_t]) where λ_t = exp(a_1·x_1 + a_2·x_2 + b).
"""
function calc_Rinv_param(
    model::PoissonObservationParam{T}, z::SVector{2,T}, θ::SVector{3,T}
) where {T}
    η = h_param(model, z, θ)
    λ = exp(η[1])
    return Diagonal(@SVector([λ]))
end

"""
Reciprocal of weight matrix (for compatibility).
Not really used in Poisson case - likelihood uses λ directly.
"""
function calc_R_param(
    model::PoissonObservationParam{T}, z::SVector{2,T}, θ::SVector{3,T}
) where {T}
    η = h_param(model, z, θ)
    λ = exp(η[1])
    return Diagonal(@SVector([one(T) / λ]))
end
