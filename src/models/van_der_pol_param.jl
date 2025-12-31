export VanDerPolDynamicsParam
export f_param, calc_Jf_param, calc_Hfs_param, calc_Q_param, calc_Qinv_param
export calc_f_θ, calc_Hf_θs, calc_Jf_θ, calc_f_θθ

"""
Van der Pol oscillator dynamics for RHMC benchmarking.

The state is `[u, v]` where u is position and v is velocity.
The parameter is θ = log(μ) where μ controls the stiffness.

Continuous-time dynamics:
    u̇ = v
    v̇ = μ(1 - u²)v - u

Euler discretization:
    u_{k+1} = u_k + δt * v_k
    v_{k+1} = v_k + δt * [μ(1 - u_k²)v_k - u_k] + ε_k

where ε_k ~ N(0, Q) with Q = diag(σ_u², σ_v²).

For stiff μ (e.g., μ ≥ 3), the system exhibits relaxation oscillations
with position-dependent curvature that makes RHMC advantageous over HMC.
"""
struct VanDerPolDynamicsParam{T}
    σ_u::T    # Process noise std on u component (small, e.g., 0.01)
    σ_v::T    # Process noise std on v component (main noise source)
    δt::T     # Time step
end

const P_VDP = 1  # Number of parameters (log_μ)

"""
Dynamics function f(z, θ) where θ = [log(μ)].
"""
function f_param(dyn::VanDerPolDynamicsParam{T}, z::SVector{2,T}, θ::SVector{1,T}) where {T}
    u, v = z
    log_μ = θ[1]
    μ = exp(log_μ)
    δt = dyn.δt

    new_u = u + δt * v
    new_v = v + δt * (μ * (1 - u^2) * v - u)

    return @SVector [new_u, new_v]
end

"""
Jacobian ∂f/∂z (2×2 matrix).

∂f_u/∂u = 1,         ∂f_u/∂v = δt
∂f_v/∂u = δt(μ(-2u)v - 1),  ∂f_v/∂v = 1 + δt·μ(1-u²)
"""
function calc_Jf_param(
    dyn::VanDerPolDynamicsParam{T}, z::SVector{2,T}, θ::SVector{1,T}
) where {T}
    u, v = z
    log_μ = θ[1]
    μ = exp(log_μ)
    δt = dyn.δt

    df_u_du = one(T)
    df_u_dv = δt
    df_v_du = δt * (μ * (-2u) * v - 1)
    df_v_dv = 1 + δt * μ * (1 - u^2)

    Jf = @SMatrix [
        df_u_du df_u_dv
        df_v_du df_v_dv
    ]
    return Jf
end

"""
Parameter Jacobian ∂f/∂θ (2×1 matrix).
θ = [log(μ)], so we use chain rule: ∂f/∂θ = ∂f/∂μ * μ

∂f_u/∂θ = 0
∂f_v/∂θ = δt·μ(1-u²)v
"""
function calc_f_θ(
    dyn::VanDerPolDynamicsParam{T}, z::SVector{2,T}, θ::SVector{1,T}
) where {T}
    u, v = z
    log_μ = θ[1]
    μ = exp(log_μ)
    δt = dyn.δt

    df_dlog_μ = @SVector [zero(T), δt * μ * (1 - u^2) * v]

    return hcat(df_dlog_μ)
end

"""
Hessians ∂²f/∂z_d∂z (D matrices, each 2×2).

For f_u = u + δt*v:
    All second derivatives are zero.

For f_v = v + δt*(μ(1-u²)v - u):
    ∂²f_v/∂u² = δt*μ*(-2)*v
    ∂²f_v/∂v∂u = δt*μ*(-2u)
    ∂²f_v/∂u∂v = δt*μ*(-2u)
    ∂²f_v/∂v² = 0
"""
function calc_Hfs_param(
    dyn::VanDerPolDynamicsParam{T}, z::SVector{2,T}, θ::SVector{1,T}
) where {T}
    u, v = z
    log_μ = θ[1]
    μ = exp(log_μ)
    δt = dyn.δt

    # Hf_du (∂²f/∂u∂z) - derivatives of Jf[:, 1] w.r.t. z
    # Jf[:, 1] = [1, δt*(μ*(-2u)*v - 1)]
    # ∂/∂u: [0, δt*μ*(-2)*v]
    # ∂/∂v: [0, δt*μ*(-2u)]
    Hf1 = @SMatrix [
        zero(T) zero(T)
        δt*μ*(-2)*v δt*μ*(-2u)
    ]

    # Hf_dv (∂²f/∂v∂z) - derivatives of Jf[:, 2] w.r.t. z
    # Jf[:, 2] = [δt, 1 + δt*μ*(1-u²)]
    # ∂/∂u: [0, δt*μ*(-2u)]
    # ∂/∂v: [0, 0]
    Hf2 = @SMatrix [
        zero(T) zero(T)
        δt*μ*(-2u) zero(T)
    ]

    return Hf1, Hf2
end

"""
Mixed Hessians ∂²f/∂z_d∂θ (D matrices, each 2×P).
These are needed for computing ∂G_{t,θ}/∂x_t^{(d)}.

∂²f/∂u∂θ = [0, δt*μ*(-2u)*v]  (derivative of f_θ w.r.t. u)
∂²f/∂v∂θ = [0, δt*μ*(1-u²)]   (derivative of f_θ w.r.t. v)
"""
function calc_Hf_θs(
    dyn::VanDerPolDynamicsParam{T}, z::SVector{2,T}, θ::SVector{1,T}
) where {T}
    u, v = z
    log_μ = θ[1]
    μ = exp(log_μ)
    δt = dyn.δt

    # ∂²f/∂u∂θ (2×1 matrix)
    Hf_θ_1 = @SMatrix [
        zero(T)
        δt * μ * (-2u) * v
    ]

    # ∂²f/∂v∂θ (2×1 matrix)
    Hf_θ_2 = @SMatrix [
        zero(T)
        δt * μ * (1 - u^2)
    ]

    return Hf_θ_1, Hf_θ_2
end

"""
Jacobian derivatives ∂Jf/∂θ^{(p)} (P matrices, each 2×2).
These are needed for computing ∂G_{tt}/∂θ^{(p)}.

Since θ = log(μ), ∂/∂θ = μ*∂/∂μ.

Jf = [1, δt; δt*(μ*(-2u)*v - 1), 1 + δt*μ*(1-u²)]

∂Jf/∂θ = μ * ∂Jf/∂μ = [0, 0; δt*μ*(-2u)*v, δt*μ*(1-u²)]
"""
function calc_Jf_θ(
    dyn::VanDerPolDynamicsParam{T}, z::SVector{2,T}, θ::SVector{1,T}
) where {T}
    u, v = z
    log_μ = θ[1]
    μ = exp(log_μ)
    δt = dyn.δt

    # ∂Jf/∂(log μ)
    dJf_dlog_μ = @SMatrix [
        zero(T) zero(T)
        δt * μ * (-2u) * v δt * μ * (1 - u^2)
    ]

    return (dJf_dlog_μ,)
end

"""
Second parameter derivatives ∂²f/∂θ^{(p)}∂θ (P matrices, each 2×P).
For log-parameterization: ∂²f/∂θ² = f_θ since ∂²μ/∂θ² = μ = ∂μ/∂θ.

f_θ = [0, δt*μ*(1-u²)*v]
∂²f/∂θ² = [0, δt*μ*(1-u²)*v] = f_θ
"""
function calc_f_θθ(
    dyn::VanDerPolDynamicsParam{T}, z::SVector{2,T}, θ::SVector{1,T}
) where {T}
    f_θ = calc_f_θ(dyn, z, θ)

    # ∂²f/∂θ₁∂θ = [∂²f/∂θ₁²] = [f_θ[:, 1]]
    df_θ_1 = @SMatrix [
        f_θ[1, 1]
        f_θ[2, 1]
    ]

    return (df_θ_1,)
end

function calc_Q_param(dyn::VanDerPolDynamicsParam)
    σ_u = dyn.σ_u
    σ_v = dyn.σ_v
    return Diagonal(@SVector([σ_u^2, σ_v^2]))
end

function calc_Qinv_param(dyn::VanDerPolDynamicsParam)
    σ_u = dyn.σ_u
    σ_v = dyn.σ_v
    return Diagonal(@SVector([1.0 / σ_u^2, 1.0 / σ_v^2]))
end
