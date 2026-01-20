export VanDerPolDynamicsParam
export f_param,
    calc_Jf_param,
    calc_Hfs_param,
    calc_dyn_hessians,
    calc_dyn_hessian_derivs,
    calc_Q_param,
    calc_Qinv_param
export calc_f_θ,
    calc_Hf_θs, calc_mixed_hessians, calc_mixed_hessian_derivs, calc_Jf_θ, calc_f_θθ

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
Component Hessians ∇²f_d (D symmetric matrices, each D×D).

Returns the Hessian of each output component with respect to state:
    ∇²f_d[i,j] = ∂²f_d/∂z_i∂z_j

These are the matrices needed for the observed Hessian correction:
    H_{x_{t-1},x_{t-1}}^{obs} = -F'SF + Σ_d (Sr)_d ∇²f_d

Relationship to calc_Hfs_param output:
    ∇²f_d[i,j] = Hf_i[d,j]
where Hf_i = ∂(Jf[:,i])/∂z.
"""
function calc_dyn_hessians(
    dyn::VanDerPolDynamicsParam{T}, z::SVector{2,T}, θ::SVector{1,T}
) where {T}
    Hf1, Hf2 = calc_Hfs_param(dyn, z, θ)

    # ∇²f_1 (Hessian of first component f_u) - all zeros since f_u is linear
    H_f1 = @SMatrix [
        Hf1[1, 1] Hf1[1, 2]
        Hf2[1, 1] Hf2[1, 2]
    ]

    # ∇²f_2 (Hessian of second component f_v) - symmetric
    H_f2 = @SMatrix [
        Hf1[2, 1] Hf1[2, 2]
        Hf2[2, 1] Hf2[2, 2]
    ]

    return H_f1, H_f2
end

"""
Component mixed Hessians ∇²_{z,θ} f_i (D matrices, each D×P).

Returns the mixed Hessian of each output component with respect to state and parameters:
    ∇²_{z,θ} f_i[j,p] = ∂²f_i/(∂z_j ∂θ_p)

These are the matrices needed for the observed Hessian correction in the border block:
    H_{x_{t-1},θ}^{obs} = Σ_i (Sr)_i ∇²_{z,θ} f_i

Relationship to calc_Hf_θs output:
    ∇²_{z,θ} f_i[j,p] = Hf_θ_j[i,p]
where Hf_θ_j = ∂(f_θ)/∂z_j.
"""
function calc_mixed_hessians(
    dyn::VanDerPolDynamicsParam{T}, z::SVector{2,T}, θ::SVector{1,T}
) where {T}
    Hf_θ_1, Hf_θ_2 = calc_Hf_θs(dyn, z, θ)

    # ∇²_{z,θ} f_1 (mixed Hessian of first component f_u) - all zeros since f_u is linear in θ
    H_mixed_f1 = @SMatrix [
        Hf_θ_1[1, 1]
        Hf_θ_2[1, 1]
    ]

    # ∇²_{z,θ} f_2 (mixed Hessian of second component f_v)
    H_mixed_f2 = @SMatrix [
        Hf_θ_1[2, 1]
        Hf_θ_2[2, 1]
    ]

    return H_mixed_f1, H_mixed_f2
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

"""
Third derivatives of component Hessians: ∂(∇²f_d)/∂z_c (D×D matrices for each d,c).

Returns a tuple of D tuples, where result[d][c] = ∂(∇²f_d)/∂z_c.

For VDP:
- f₁ is linear, so all third derivatives are zero.
- For f₂:
    ∇²f₂ = [[δt·μ·(-2)·v, δt·μ·(-2u)], [δt·μ·(-2u), 0]]
    ∂(∇²f₂)/∂u = [[0, -2·δt·μ], [-2·δt·μ, 0]]
    ∂(∇²f₂)/∂v = [[-2·δt·μ, 0], [0, 0]]
"""
function calc_dyn_hessian_derivs(
    dyn::VanDerPolDynamicsParam{T}, z::SVector{2,T}, θ::SVector{1,T}
) where {T}
    log_μ = θ[1]
    μ = exp(log_μ)
    δt = dyn.δt

    Z = @SMatrix zeros(T, 2, 2)

    # Component 1 (f_u is linear): all third derivatives are zero
    dH_f1_du = Z
    dH_f1_dv = Z

    # Component 2 (f_v):
    # ∂(∇²f₂)/∂u = [[0, -2·δt·μ], [-2·δt·μ, 0]]
    dH_f2_du = @SMatrix [
        zero(T) -2*δt*μ
        -2*δt*μ zero(T)
    ]

    # ∂(∇²f₂)/∂v = [[-2·δt·μ, 0], [0, 0]]
    dH_f2_dv = @SMatrix [
        -2*δt*μ zero(T)
        zero(T) zero(T)
    ]

    return ((dH_f1_du, dH_f1_dv), (dH_f2_du, dH_f2_dv))
end

"""
Third derivatives of mixed Hessians: ∂(∇²_{z,θ}f_i)/∂z_d (D×P matrices for each i,d).

Returns a tuple of D tuples, where result[i][d] = ∂(∇²_{z,θ}f_i)/∂z_d.

For VDP:
- f₁ is linear, so all third derivatives are zero.
- For f₂:
    ∇²_{z,θ}f₂ = [δt·μ·(-2u)·v; δt·μ·(1-u²)]
    ∂(∇²_{z,θ}f₂)/∂u = [-2·δt·μ·v; -2·δt·μ·u]
    ∂(∇²_{z,θ}f₂)/∂v = [-2·δt·μ·u; 0]
"""
function calc_mixed_hessian_derivs(
    dyn::VanDerPolDynamicsParam{T}, z::SVector{2,T}, θ::SVector{1,T}
) where {T}
    u, v = z
    log_μ = θ[1]
    μ = exp(log_μ)
    δt = dyn.δt

    Z = @SMatrix zeros(T, 2, 1)

    # Component 1 (f_u is linear): all third derivatives are zero
    dMH_f1_du = Z
    dMH_f1_dv = Z

    # Component 2 (f_v):
    # ∂(∇²_{z,θ}f₂)/∂u = [-2·δt·μ·v; -2·δt·μ·u]
    dMH_f2_du = @SMatrix [
        -2*δt*μ*v
        -2*δt*μ*u
    ]

    # ∂(∇²_{z,θ}f₂)/∂v = [-2·δt·μ·u; 0]
    dMH_f2_dv = @SMatrix [
        -2*δt*μ*u
        zero(T)
    ]

    return ((dMH_f1_du, dMH_f1_dv), (dMH_f2_du, dMH_f2_dv))
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
