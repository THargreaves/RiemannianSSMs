export VariableRestoringForceDynamics, f, calc_Jf, calc_Hfs, calc_Q, calc_Qinv

"""
A 2D motion model with a variable restoring force toward the origin and non-linear drag.

The state is `[x, y, vx, vy]`, where `(x, y)` is position and `(vx, vy)` is velocity. The
dynamics are given by:

x_{t+1} = x_t + δt * vx_t
y_{t+1} = y_t + δt * vy_t
vx_{t+1} = vx_t * (1 - β * ||v_t|| * δt) - α * (1 + γ * r_t) * x_t * δt
vy_{t+1} = vy_t * (1 - β * ||v_t|| * δt) - α * (1 + γ * r_t) * y_t * δt

where `r_t = sqrt(x_t^2 + y_t^2)` is the distance from the origin and
`||v_t|| = sqrt(vx_t^2 + vy_t^2)` is the speed.
``` 
"""
struct VariableRestoringForceDynamics{T}
    α::T    # restoring force
    β::T    # damping coefficient
    γ::T    # force scaling
    σ_p::T  # position noise
    σ_v::T  # velocity noise
    δt::T   # time step
end

function f(dyn::VariableRestoringForceDynamics{T}, z::SVector{4,T}) where {T}
    x, y, vx, vy = z
    α, β, γ, δt = dyn.α, dyn.β, dyn.γ, dyn.δt

    new_x = x + δt * vx
    new_y = y + δt * vy

    v_norm = sqrt(vx^2 + vy^2)
    r = sqrt(x^2 + y^2)
    rest_force = α * (1 + γ * r)

    new_vx = vx * (1 - β * v_norm * δt) - rest_force * x * δt
    new_vy = vy * (1 - β * v_norm * δt) - rest_force * y * δt

    return @SVector [new_x, new_y, new_vx, new_vy]
end

# AbstractVector version
function f(dyn::VariableRestoringForceDynamics, z::AbstractVector)
    x, y, vx, vy = z
    α, β, γ, δt = dyn.α, dyn.β, dyn.γ, dyn.δt
    new_x = x + δt * vx
    new_y = y + δt * vy
    v_norm = sqrt(vx^2 + vy^2)
    r = sqrt(x^2 + y^2)
    rest_force = α * (1 + γ * r)
    new_vx = vx * (1 - β * v_norm * δt) - rest_force * x * δt
    new_vy = vy * (1 - β * v_norm * δt) - rest_force * y * δt
    return [new_x, new_y, new_vx, new_vy]
end

function calc_Jf(dyn::VariableRestoringForceDynamics{T}, z::SVector{4,T}) where {T}
    x, y, vx, vy = z
    α, β, γ, δt = dyn.α, dyn.β, dyn.γ, dyn.δt

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

# TODO: take advantage of block sparsity
function calc_Hfs(dyn::VariableRestoringForceDynamics, z::SVector{4,T}) where {T}
    x, y, vx, vy = z
    α, β, γ, δt = dyn.α, dyn.β, dyn.γ, dyn.δt
    r3 = (x^2 + y^2)^(3 / 2)
    s3 = (vx^2 + vy^2)^(3 / 2)  # speed cubed

    # Jf_dx
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

    # Jf_dy
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

    # Jf_vx
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

    # Jf_vy
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

function calc_Q(dyn::VariableRestoringForceDynamics)
    σ_p = dyn.σ_p
    σ_v = dyn.σ_v
    return Diagonal(@SVector([σ_p^2, σ_p^2, σ_v^2, σ_v^2]))
end

function calc_Qinv(dyn::VariableRestoringForceDynamics)
    σ_p = dyn.σ_p
    σ_v = dyn.σ_v
    return Diagonal(@SVector([1.0 / σ_p^2, 1.0 / σ_p^2, 1.0 / σ_v^2, 1.0 / σ_v^2]))
end
