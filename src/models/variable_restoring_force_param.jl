export VariableRestoringForceDynamicsParam
export f_param, calc_Jf_param, calc_Hfs_param, calc_Q_param, calc_Qinv_param
export calc_f_θ, calc_Hf_θs, calc_Jf_θ, calc_f_θθ

"""
A 2D motion model with a variable restoring force toward the origin and non-linear drag.
This version treats α and β as parameters to be inferred (in log space).

The state is `[x, y, vx, vy]`, where `(x, y)` is position and `(vx, vy)` is velocity.
The parameters are θ = [log(α), log(β)] where:
- α: restoring force coefficient
- β: damping coefficient

The dynamics are given by:
x_{t+1} = x_t + δt * vx_t
y_{t+1} = y_t + δt * vy_t
vx_{t+1} = vx_t * (1 - β * ||v_t|| * δt) - α * (1 + γ * r_t) * x_t * δt
vy_{t+1} = vy_t * (1 - β * ||v_t|| * δt) - α * (1 + γ * r_t) * y_t * δt

where r_t = sqrt(x_t^2 + y_t^2) is the distance from the origin and
||v_t|| = sqrt(vx_t^2 + vy_t^2) is the speed.
"""
struct VariableRestoringForceDynamicsParam{T}
    γ::T    # force scaling (fixed)
    σ_p::T  # position noise (fixed)
    σ_v::T  # velocity noise (fixed)
    δt::T   # time step (fixed)
end

const P_PARAM = 2  # Number of parameters (log_α, log_β)

"""
Dynamics function f(z, θ) where θ = [log(α), log(β)].
"""
function f_param(
    dyn::VariableRestoringForceDynamicsParam{T}, z::SVector{4,T}, θ::SVector{2,T}
) where {T}
    x, y, vx, vy = z
    log_α, log_β = θ
    α = exp(log_α)
    β = exp(log_β)
    γ, δt = dyn.γ, dyn.δt

    new_x = x + δt * vx
    new_y = y + δt * vy

    v_norm = sqrt(vx^2 + vy^2)
    r = sqrt(x^2 + y^2)
    rest_force = α * (1 + γ * r)

    new_vx = vx * (1 - β * v_norm * δt) - rest_force * x * δt
    new_vy = vy * (1 - β * v_norm * δt) - rest_force * y * δt

    return @SVector [new_x, new_y, new_vx, new_vy]
end

"""
Jacobian ∂f/∂z (4×4 matrix).
"""
function calc_Jf_param(
    dyn::VariableRestoringForceDynamicsParam{T}, z::SVector{4,T}, θ::SVector{2,T}
) where {T}
    x, y, vx, vy = z
    log_α, log_β = θ
    α = exp(log_α)
    β = exp(log_β)
    γ, δt = dyn.γ, dyn.δt

    r = sqrt(x^2 + y^2)
    v_norm = sqrt(vx^2 + vy^2)
    F = T(1) + γ * r

    J_vx_x = -α * δt * (F + γ * x^2 / r)
    J_vx_y = -α * δt * (γ * x * y / r)
    J_vy_x = -α * δt * (γ * x * y / r)
    J_vy_y = -α * δt * (F + γ * y^2 / r)
    J_vx_vx = 1 - β * δt * (v_norm + vx^2 / v_norm)
    J_vx_vy = -β * δt * (vx * vy / v_norm)
    J_vy_vx = -β * δt * (vx * vy / v_norm)
    J_vy_vy = 1 - β * δt * (v_norm + vy^2 / v_norm)

    Jf = @SMatrix [
        1.0 0.0 δt 0.0
        0.0 1.0 0.0 δt
        J_vx_x J_vx_y J_vx_vx J_vx_vy
        J_vy_x J_vy_y J_vy_vx J_vy_vy
    ]
    return Jf
end

"""
Parameter Jacobian ∂f/∂θ (4×2 matrix).
θ = [log(α), log(β)], so we use chain rule: ∂f/∂θ₁ = ∂f/∂α * α, ∂f/∂θ₂ = ∂f/∂β * β
"""
function calc_f_θ(
    dyn::VariableRestoringForceDynamicsParam{T}, z::SVector{4,T}, θ::SVector{2,T}
) where {T}
    x, y, vx, vy = z
    log_α, log_β = θ
    α = exp(log_α)
    β = exp(log_β)
    γ, δt = dyn.γ, dyn.δt

    r = sqrt(x^2 + y^2)
    v_norm = sqrt(vx^2 + vy^2)
    F = T(1) + γ * r

    # ∂f/∂(log α) = ∂f/∂α * α
    df_dlog_α = @SVector [zero(T), zero(T), -α * F * x * δt, -α * F * y * δt]

    # ∂f/∂(log β) = ∂f/∂β * β
    df_dlog_β = @SVector [zero(T), zero(T), -β * v_norm * vx * δt, -β * v_norm * vy * δt]

    return hcat(df_dlog_α, df_dlog_β)
end

"""
Hessians ∂²f/∂z_d∂z (D matrices, each 4×4).
Same as calc_Hfs but with parameterized α, β.
"""
function calc_Hfs_param(
    dyn::VariableRestoringForceDynamicsParam{T}, z::SVector{4,T}, θ::SVector{2,T}
) where {T}
    x, y, vx, vy = z
    log_α, log_β = θ
    α = exp(log_α)
    β = exp(log_β)
    γ, δt = dyn.γ, dyn.δt

    r2 = x^2 + y^2
    s2 = vx^2 + vy^2
    r3 = r2 * sqrt(r2)
    s3 = s2 * sqrt(s2)

    # Hf_dx (∂²f/∂x∂z)
    t11 = -α * γ * δt * x * (2x^2 + 3y^2) / r3
    t12 = -α * γ * δt * y^3 / r3
    t21 = -α * γ * δt * y^3 / r3
    t22 = -α * γ * δt * x^3 / r3
    Hf1 = @SMatrix [
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
        t11 t12 0.0 0.0
        t21 t22 0.0 0.0
    ]

    # Hf_dy (∂²f/∂y∂z)
    t11 = -α * γ * δt * y^3 / r3
    t12 = -α * γ * δt * x^3 / r3
    t21 = -α * γ * δt * x^3 / r3
    t22 = -α * γ * δt * y * (2y^2 + 3x^2) / r3
    Hf2 = @SMatrix [
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
        t11 t12 0.0 0.0
        t21 t22 0.0 0.0
    ]

    # Hf_dvx (∂²f/∂vx∂z)
    t11 = -β * δt * vx * (2vx^2 + 3vy^2) / s3
    t12 = -β * δt * vy^3 / s3
    t21 = -β * δt * vy^3 / s3
    t22 = -β * δt * vx^3 / s3
    Hf3 = @SMatrix [
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
        0.0 0.0 t11 t12
        0.0 0.0 t21 t22
    ]

    # Hf_dvy (∂²f/∂vy∂z)
    t11 = -β * δt * vy^3 / s3
    t12 = -β * δt * vx^3 / s3
    t21 = -β * δt * vx^3 / s3
    t22 = -β * δt * vy * (2vy^2 + 3vx^2) / s3
    Hf4 = @SMatrix [
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
        0.0 0.0 t11 t12
        0.0 0.0 t21 t22
    ]

    return Hf1, Hf2, Hf3, Hf4
end

"""
Mixed Hessians ∂²f/∂z_d∂θ (D matrices, each 4×2).
These are needed for computing ∂G_{t,θ}/∂x_t^{(d)}.
"""
function calc_Hf_θs(
    dyn::VariableRestoringForceDynamicsParam{T}, z::SVector{4,T}, θ::SVector{2,T}
) where {T}
    x, y, vx, vy = z
    log_α, log_β = θ
    α = exp(log_α)
    β = exp(log_β)
    γ, δt = dyn.γ, dyn.δt

    r = sqrt(x^2 + y^2)
    v_norm = sqrt(vx^2 + vy^2)
    F = T(1) + γ * r

    # ∂²f/∂x∂θ (4×2 matrix)
    Hf_θ_1 = @SMatrix [
        zero(T) zero(T)
        zero(T) zero(T)
        -α * δt * (F + γ * x^2 / r) zero(T)
        -α * δt * γ * x * y / r zero(T)
    ]

    # ∂²f/∂y∂θ (4×2 matrix)
    Hf_θ_2 = @SMatrix [
        zero(T) zero(T)
        zero(T) zero(T)
        -α * δt * γ * x * y / r zero(T)
        -α * δt * (F + γ * y^2 / r) zero(T)
    ]

    # ∂²f/∂vx∂θ (4×2 matrix)
    Hf_θ_3 = @SMatrix [
        zero(T) zero(T)
        zero(T) zero(T)
        zero(T) -β * δt * (v_norm + vx^2 / v_norm)
        zero(T) -β * δt * vx * vy / v_norm
    ]

    # ∂²f/∂vy∂θ (4×2 matrix)
    Hf_θ_4 = @SMatrix [
        zero(T) zero(T)
        zero(T) zero(T)
        zero(T) -β * δt * vx * vy / v_norm
        zero(T) -β * δt * (v_norm + vy^2 / v_norm)
    ]

    return Hf_θ_1, Hf_θ_2, Hf_θ_3, Hf_θ_4
end

"""
Jacobian derivatives ∂Jf/∂θ^{(p)} (P matrices, each 4×4).
These are needed for computing ∂G_{tt}/∂θ^{(p)}.
"""
function calc_Jf_θ(
    dyn::VariableRestoringForceDynamicsParam{T}, z::SVector{4,T}, θ::SVector{2,T}
) where {T}
    x, y, vx, vy = z
    log_α, log_β = θ
    α = exp(log_α)
    β = exp(log_β)
    γ, δt = dyn.γ, dyn.δt

    r = sqrt(x^2 + y^2)
    v_norm = sqrt(vx^2 + vy^2)
    F = T(1) + γ * r

    # ∂Jf/∂(log α) - only the α-dependent parts of Jf, multiplied by α
    # The α terms are in rows 3,4 and columns 1,2
    J_vx_x = -α * δt * (F + γ * x^2 / r)
    J_vx_y = -α * δt * (γ * x * y / r)
    J_vy_x = -α * δt * (γ * x * y / r)
    J_vy_y = -α * δt * (F + γ * y^2 / r)

    dJf_dlog_α = @SMatrix [
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
        J_vx_x J_vx_y 0.0 0.0
        J_vy_x J_vy_y 0.0 0.0
    ]

    # ∂Jf/∂(log β) - only the β-dependent parts of Jf, multiplied by β
    # The β terms are in rows 3,4 and columns 3,4
    J_vx_vx_β = -β * δt * (v_norm + vx^2 / v_norm)
    J_vx_vy_β = -β * δt * (vx * vy / v_norm)
    J_vy_vx_β = -β * δt * (vx * vy / v_norm)
    J_vy_vy_β = -β * δt * (v_norm + vy^2 / v_norm)

    dJf_dlog_β = @SMatrix [
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
        0.0 0.0 J_vx_vx_β J_vx_vy_β
        0.0 0.0 J_vy_vx_β J_vy_vy_β
    ]

    return dJf_dlog_α, dJf_dlog_β
end

"""
Second parameter derivatives ∂²f/∂θ^{(p)}∂θ (P matrices, each 4×2).
For log-parameterization: ∂²f/∂θ_p² = f_θ[:, p] since ∂²α/∂θ₁² = α = ∂α/∂θ₁.
Cross derivatives are zero since α and β don't interact.
"""
function calc_f_θθ(
    dyn::VariableRestoringForceDynamicsParam{T}, z::SVector{4,T}, θ::SVector{2,T}
) where {T}
    f_θ = calc_f_θ(dyn, z, θ)

    # ∂²f/∂θ₁∂θ = [∂²f/∂θ₁², ∂²f/∂θ₁∂θ₂] = [f_θ[:, 1], 0]
    df_θ_1 = @SMatrix [
        f_θ[1, 1] zero(T)
        f_θ[2, 1] zero(T)
        f_θ[3, 1] zero(T)
        f_θ[4, 1] zero(T)
    ]

    # ∂²f/∂θ₂∂θ = [∂²f/∂θ₂∂θ₁, ∂²f/∂θ₂²] = [0, f_θ[:, 2]]
    df_θ_2 = @SMatrix [
        zero(T) f_θ[1, 2]
        zero(T) f_θ[2, 2]
        zero(T) f_θ[3, 2]
        zero(T) f_θ[4, 2]
    ]

    return df_θ_1, df_θ_2
end

function calc_Q_param(dyn::VariableRestoringForceDynamicsParam)
    σ_p = dyn.σ_p
    σ_v = dyn.σ_v
    return Diagonal(@SVector([σ_p^2, σ_p^2, σ_v^2, σ_v^2]))
end

function calc_Qinv_param(dyn::VariableRestoringForceDynamicsParam)
    σ_p = dyn.σ_p
    σ_v = dyn.σ_v
    return Diagonal(@SVector([1.0 / σ_p^2, 1.0 / σ_p^2, 1.0 / σ_v^2, 1.0 / σ_v^2]))
end
