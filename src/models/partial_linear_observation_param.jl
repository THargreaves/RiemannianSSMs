export PartialLinearObservationParam
export h_param, calc_Jh_param, calc_Hhs_param, calc_R_param, calc_Rinv_param
export calc_h_θ, calc_Hh_θs, calc_Jh_θ, calc_h_θθ

"""
A simple linear observation model that observes a single dimension of the state.

For use with Van der Pol oscillator: y = u + η where η ~ N(0, σ²).

The observation function is h(z, θ) = z[obs_dim] (defaults to 1 for u).
Since h is linear in z and independent of θ, all second derivatives and θ-derivatives are zero.
"""
struct PartialLinearObservationParam{T}
    σ::T       # Observation noise std
    obs_dim::Int   # Which dimension is observed (1 for u in Van der Pol)
end

function PartialLinearObservationParam(σ::T) where {T}
    return PartialLinearObservationParam{T}(σ, 1)
end

"""
Observation function h(z, θ) = z[obs_dim].
Returns a 1-element SVector for scalar observation.
"""
function h_param(
    model::PartialLinearObservationParam{T}, z::SVector{D,T}, θ::SVector{P,T}
) where {T,D,P}
    return @SVector [z[model.obs_dim]]
end

"""
Jacobian ∂h/∂z (1×D matrix).
For obs_dim=1: [1, 0, ...]
"""
function calc_Jh_param(
    model::PartialLinearObservationParam{T}, z::SVector{D,T}, θ::SVector{P,T}
) where {T,D,P}
    Jh = zeros(MMatrix{1,D,T})
    Jh[1, model.obs_dim] = one(T)
    return SMatrix{1,D,T}(Jh)
end

"""
Parameter Jacobian ∂h/∂θ (1×P matrix).
Since h does not depend on θ, this returns zeros.
"""
function calc_h_θ(
    model::PartialLinearObservationParam{T}, z::SVector{D,T}, θ::SVector{P,T}
) where {T,D,P}
    return @SMatrix zeros(T, 1, P)
end

"""
Hessians ∂²h/∂z_d∂z (D matrices, each 1×D).
Since h is linear in z, all second derivatives are zero.
"""
function calc_Hhs_param(
    model::PartialLinearObservationParam{T}, z::SVector{D,T}, θ::SVector{P,T}
) where {T,D,P}
    Z = @SMatrix zeros(T, 1, D)
    return ntuple(_ -> Z, D)
end

"""
Mixed Hessians ∂²h/∂z_d∂θ (D matrices, each 1×P).
Since h does not depend on θ, all return zeros.
"""
function calc_Hh_θs(
    model::PartialLinearObservationParam{T}, z::SVector{D,T}, θ::SVector{P,T}
) where {T,D,P}
    Z = @SMatrix zeros(T, 1, P)
    return ntuple(_ -> Z, D)
end

"""
Jacobian derivatives ∂Jh/∂θ^{(p)} (P matrices, each 1×D).
Since h does not depend on θ, all return zeros.
"""
function calc_Jh_θ(
    model::PartialLinearObservationParam{T}, z::SVector{D,T}, θ::SVector{P,T}
) where {T,D,P}
    Z = @SMatrix zeros(T, 1, D)
    return ntuple(_ -> Z, P)
end

"""
Second parameter derivatives ∂²h/∂θ^{(p)}∂θ (P matrices, each 1×P).
Since h does not depend on θ, all return zeros.
"""
function calc_h_θθ(
    model::PartialLinearObservationParam{T}, z::SVector{D,T}, θ::SVector{P,T}
) where {T,D,P}
    Z = @SMatrix zeros(T, 1, P)
    return ntuple(_ -> Z, P)
end

function calc_R_param(model::PartialLinearObservationParam)
    σ = model.σ
    return Diagonal(@SVector([σ^2]))
end

function calc_Rinv_param(model::PartialLinearObservationParam)
    σ = model.σ
    return Diagonal(@SVector([1.0 / σ^2]))
end
