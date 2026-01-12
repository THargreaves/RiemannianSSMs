export RandomWalkDynamicsParam
export f_param, calc_Jf_param, calc_Hfs_param, calc_Q_param, calc_Qinv_param
export calc_f_θ, calc_Hf_θs, calc_Jf_θ, calc_f_θθ

"""
Random walk dynamics for Poisson SSM benchmarking.

The state is `[x_1, x_2]` and evolves as a random walk:
    x_{k+1} = x_k + ε_k

where ε_k ~ N(0, Q) with Q = diag(σ_1², σ_2²).

The dynamics are parameter-free (identity map), so all parameter
derivatives are zero. This allows the challenging geometry to come
entirely from the Poisson observation model.
"""
struct RandomWalkDynamicsParam{T}
    σ_1::T    # Process noise std for first component
    σ_2::T    # Process noise std for second component
end

const P_RW = 3  # Number of parameters (a_1, a_2, b) - but dynamics don't depend on them

"""
Dynamics function f(z, θ) = z (identity).
"""
function f_param(
    dyn::RandomWalkDynamicsParam{T}, z::SVector{2,T}, θ::SVector{P,T}
) where {T,P}
    return z  # Identity dynamics
end

"""
Jacobian ∂f/∂z = I (identity matrix).
"""
function calc_Jf_param(
    dyn::RandomWalkDynamicsParam{T}, z::SVector{2,T}, θ::SVector{P,T}
) where {T,P}
    return @SMatrix [
        one(T) zero(T)
        zero(T) one(T)
    ]
end

"""
Hessians ∂²f/∂z_d∂z (D matrices, each 2×2).
All zero since f is linear.
"""
function calc_Hfs_param(
    dyn::RandomWalkDynamicsParam{T}, z::SVector{2,T}, θ::SVector{P,T}
) where {T,P}
    Z = @SMatrix zeros(T, 2, 2)
    return (Z, Z)
end

"""
Parameter Jacobian ∂f/∂θ (2×P matrix).
All zero since f doesn't depend on θ.
"""
function calc_f_θ(
    dyn::RandomWalkDynamicsParam{T}, z::SVector{2,T}, θ::SVector{P,T}
) where {T,P}
    return @SMatrix zeros(T, 2, P)
end

"""
Mixed Hessians ∂²f/∂z_d∂θ (D matrices, each 2×P).
All zero since f doesn't depend on θ.
"""
function calc_Hf_θs(
    dyn::RandomWalkDynamicsParam{T}, z::SVector{2,T}, θ::SVector{P,T}
) where {T,P}
    Z = @SMatrix zeros(T, 2, P)
    return (Z, Z)
end

"""
Jacobian derivatives ∂Jf/∂θ^{(p)} (P matrices, each 2×2).
All zero since Jf = I is constant.
"""
function calc_Jf_θ(
    dyn::RandomWalkDynamicsParam{T}, z::SVector{2,T}, θ::SVector{P,T}
) where {T,P}
    Z = @SMatrix zeros(T, 2, 2)
    return ntuple(_ -> Z, P)
end

"""
Second parameter derivatives ∂²f/∂θ^{(p)}∂θ (P matrices, each 2×P).
All zero since f doesn't depend on θ.
"""
function calc_f_θθ(
    dyn::RandomWalkDynamicsParam{T}, z::SVector{2,T}, θ::SVector{P,T}
) where {T,P}
    Z = @SMatrix zeros(T, 2, P)
    return ntuple(_ -> Z, P)
end

function calc_Q_param(dyn::RandomWalkDynamicsParam)
    σ_1 = dyn.σ_1
    σ_2 = dyn.σ_2
    return Diagonal(@SVector([σ_1^2, σ_2^2]))
end

function calc_Qinv_param(dyn::RandomWalkDynamicsParam)
    σ_1 = dyn.σ_1
    σ_2 = dyn.σ_2
    return Diagonal(@SVector([1.0 / σ_1^2, 1.0 / σ_2^2]))
end
