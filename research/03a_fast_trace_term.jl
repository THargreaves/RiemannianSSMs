# Need to efficiently compute tr( G^{-1} * dG )
# G is block tridiagonal and dG only has non-zero blocks in (k, k), (k, k+1) and (k+1, k)
# If we can extract the same blocks from G^{-1} without computing a dense inverse, we can
# use the "sum of Hadamard product" trick

using BenchmarkTools
using BandedMatrices
using Distributions
using FillArrays
using LinearAlgebra
using Plots
using ProgressMeter
using Random
using StaticArrays
using OffsetArrays
using FiniteDiff

SEED = 4
rng = MersenneTwister(SEED)

K = 100  # wow, remarkably stable
δt = 1.0

α = 0.01
β = 0.1
γ = 0.005
σ_p = 0.1
σ_v = 0.5
σ_r = 0.2
σ_θ = 5.0 * π / 180.0  # radians

μ0 = @SVector [0.0, 0.0, 0.0, 0.0]
Σ0 = Diagonal(@SVector([0.5, 0.5, 0.1, 0.1]))

##########################
#### MODEL DEFINITION ####
##########################

struct VariableRestoringForceMotion
    α::Float64    # restoring force
    β::Float64    # damping coefficient
    γ::Float64    # force scaling
    σ_p::Float64  # position noise — HACK: had to add this temporarily to avoid singularity issues
    σ_v::Float64  # velocity noise
    δt::Float64   # time step
end

function f(dyn::VariableRestoringForceMotion, z::AbstractVector)
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

function calc_Jf(dyn::VariableRestoringForceMotion, z::AbstractVector)
    x, y, vx, vy = z
    α, β, γ, δt = dyn.α, dyn.β, dyn.γ, dyn.δt

    r = sqrt(x^2 + y^2)
    v_norm = sqrt(vx^2 + vy^2)
    F = 1 + γ * r

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

function calc_Hfs(dyn::VariableRestoringForceMotion, z::AbstractVector)
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

function calc_Q(dyn::VariableRestoringForceMotion)
    σ_p = dyn.σ_p
    σ_v = dyn.σ_v
    return Diagonal(@SVector([σ_p^2, σ_p^2, σ_v^2, σ_v^2]))
end

function calc_Qinv(dyn::VariableRestoringForceMotion)
    σ_p = dyn.σ_p
    σ_v = dyn.σ_v
    return Diagonal(@SVector([1.0 / σ_p^2, 1.0 / σ_p^2, 1.0 / σ_v^2, 1.0 / σ_v^2]))
end

struct BearingRangeSensor
    σ_r::Float64
    σ_θ::Float64
end

function h(::BearingRangeSensor, z::AbstractVector)
    r = sqrt(z[1]^2 + z[2]^2)
    θ = atan(z[2], z[1])
    return @SVector([r, θ])
end

function calc_Jh(sensor::BearingRangeSensor, z::AbstractVector)
    r2 = z[1]^2 + z[2]^2
    r = sqrt(r2)
    return @SMatrix [z[1]/r z[2]/r 0.0 0.0; -z[2]/r2 z[1]/r2 0.0 0.0]
end

function calc_R(sensor::BearingRangeSensor)
    σ_r = sensor.σ_r
    σ_θ = sensor.σ_θ
    return Diagonal(@SVector([σ_r^2, σ_θ^2]))
end

function calc_Rinv(sensor::BearingRangeSensor)
    σ_r = sensor.σ_r
    σ_θ = sensor.σ_θ
    return Diagonal(@SVector([1.0 / σ_r^2, 1.0 / σ_θ^2]))
end

function calc_Hhs(::BearingRangeSensor, z::AbstractVector)
    x, y = z[1], z[2]
    r = sqrt(z[1]^2 + z[2]^2)
    r3 = r^3
    r4 = r^4
    # Column-major storage 
    Hh1 = SMatrix{2,4,Float64,8}(
        y^2 / r3, 2 * x * y / r4, -x * y / r3, (y^2 - x^2) / r4, 0.0, 0.0, 0.0, 0.0
    )
    Hh2 = SMatrix{2,4,Float64,8}(
        -x * y / r3, (y^2 - x^2) / r4, x^2 / r3, -2 * x * y / r4, 0.0, 0.0, 0.0, 0.0
    )
    return Hh1, Hh2, Zeros(2, 4), Zeros(2, 4)
end

dyn = VariableRestoringForceMotion(α, β, γ, σ_p, σ_v, δt)
sensor = BearingRangeSensor(σ_r, σ_θ)
prior = MvNormal(μ0, Σ0)

####################
#### SIMULATION ####
####################

struct SSM{PT<:MvNormal}
    prior::PT
    dyn::VariableRestoringForceMotion
    sensor::BearingRangeSensor
end

ssm = SSM(prior, dyn, sensor)

function simulate(
    rng::AbstractRNG,
    prior::MvNormal,
    dyn::VariableRestoringForceMotion,
    sensor::BearingRangeSensor,
    K::Int,
)
    zs = Vector{SVector{4,Float64}}(undef, K)
    ys = Vector{SVector{2,Float64}}(undef, K)

    for k in 1:K
        if k == 1
            z = rand(rng, prior)
        else
            z = f(dyn, zs[k - 1]) + rand(rng, MvNormal(zeros(4), calc_Q(dyn)))
        end
        zs[k] = z

        y = h(sensor, z) + rand(rng, MvNormal(zeros(2), calc_R(sensor)))
        ys[k] = y
    end

    return zs, ys
end
zs_true, ys = simulate(rng, prior, dyn, sensor, K);

# Flip x coordinates to avoid singularity issues at the origin
zs_true = [SVector(-z[1], z[2], -z[3], z[4]) for z in zs_true]
# Regenerate measurements
ys = Vector{SVector{2,Float64}}(undef, K)
for k in 1:K
    y = h(sensor, zs_true[k]) + rand(rng, MvNormal(zeros(2), calc_R(sensor)))
    ys[k] = y
end

function calc_G(
    zs, Σ0_inv, dyn::VariableRestoringForceMotion, sensor::BearingRangeSensor, K::Int
)
    G = BandedMatrix(Zeros(4 * K, 4 * K), (7, 7))

    # Compute diagonal
    for k in 1:K
        # Base term
        Λ = k == 1 ? Σ0_inv : calc_Qinv(dyn)
        # Observation term
        Jh = calc_Jh(sensor, zs[k])
        Λ += Jh' * calc_Rinv(sensor) * Jh
        # Dynamics term
        if k < K
            Jf = calc_Jf(dyn, zs[k])
            Λ += Jf' * calc_Qinv(dyn) * Jf
        end

        # Insert into G
        i1 = (k - 1) * 4 + 1
        i2 = k * 4
        G[i1:i2, i1:i2] = Λ
    end

    # Compute off-diagonal term
    for k in 1:(K - 1)
        Jf = calc_Jf(dyn, zs[k])
        Λ = -Jf' * calc_Qinv(dyn)
        i1 = (k - 1) * 4 + 1
        i2 = k * 4
        j1 = k * 4 + 1
        j2 = (k + 1) * 4
        G[i1:i2, j1:j2] = Λ
        G[j1:j2, i1:i2] = Λ'
    end

    return Symmetric(G)
end

G = calc_G(zs_true, inv(Σ0), dyn, sensor, K)
G_chol = cholesky(G)

# Return two matrix of matrices — the first corresponds to diagonal terms and the second the
# off-diagonal terms. First dimension is variables and second is the time steps.
function calc_dGs(zs, dyn::VariableRestoringForceMotion, sensor::BearingRangeSensor, K::Int)
    diagonal_hessians = Matrix{SMatrix{4,4,Float64,16}}(undef, 4, K)
    off_diagonal_hessians = Matrix{SMatrix{4,4,Float64,16}}(undef, 4, K - 1)

    for k in 1:K
        Jf = calc_Jf(dyn, zs[k])
        Jh = calc_Jh(sensor, zs[k])
        Hfs = calc_Hfs(dyn, zs[k])
        Hhs = calc_Hhs(sensor, zs[k])  # latter pair are zeros but include for generality
        Q_inv = calc_Qinv(dyn)
        R_inv = calc_Rinv(sensor)

        # Cache matrices
        Mh = R_inv * Jh
        Mf = Q_inv * Jf

        for d in 1:4
            diagonal_hessians[d, k] = Hhs[d]' * Mh + Mh' * Hhs[d]
            if k < K
                diagonal_hessians[d, k] += Hfs[d]' * Mf + Mf' * Hfs[d]
            end
        end

        if k < K
            for d in 1:4
                off_diagonal_hessians[d, k] = -Hfs[d]' * Q_inv
            end
        end
    end

    return diagonal_hessians, off_diagonal_hessians
end

function ∇θ_H_naive(G_chol, dGs_diag, dGs_off, K)
    ∇θ = Vector{Float64}(undef, 4K)
    for k in 1:K
        for d in 1:4
            i = (k - 1) * 4 + d
            bi = (k - 1) * 4 + 1
            v = 0.0
            # Trace term — compute by solving for two non-zero columns of dG
            # Block k first
            c1 = zeros(Float64, 4K, 4)
            c1[bi:(bi + 3), :] = dGs_diag[d, k]
            if k < K
                c1[(bi + 4):(bi + 7), :] = dGs_off[d, k]
            end
            c1_solve = G_chol \ c1
            v += -0.5 * tr(@view c1_solve[bi:(bi + 3), :])
            if k < K
                # Block k+1
                c2 = zeros(Float64, 4K, 4)
                c2[bi:(bi + 3), :] = dGs_off[d, k]'
                c2_solve = G_chol \ c2
                v += -0.5 * tr(@view c2_solve[(bi + 4):(bi + 7), :])
            end

            ∇θ[i] = v
        end
    end

    return ∇θ
end

dGs_diag, dGs_off = calc_dGs(zs_true, dyn, sensor, K)

∇θ_naive = ∇θ_H_naive(G_chol, dGs_diag, dGs_off, K)
# display(@benchmark ∇θ_H_naive($G_chol, $dGs_diag, $dGs_off, $K))
# @profview begin
#     for _ in 1:10000
#         ∇θ_H_naive(G_chol, dGs_diag, dGs_off, K)
#     end
# end

# Doesn't work in more general setting:
# https://chatgpt.com/c/688fc9fe-44bc-800a-a744-561195a3cc99
# Test inversion
G_inv = inv(G)

# # Forward recurrences
# Θs = OffsetVector(Vector{Matrix{Float64}}(undef, K + 1), -1)
# Θs[0] = Matrix{Float64}(I, 4, 4)
# Θs[1] = G[1:4, 1:4]
# for k in 2:K
#     Ak = G[((k - 1) * 4 + 1):(k * 4), ((k - 1) * 4 + 1):(k * 4)]
#     Bk = G[((k - 1) * 4 + 1):(k * 4), ((k - 2) * 4 + 1):((k - 1) * 4)]  # B_{k-1}
#     Θs[k] = Ak * Θs[k - 1] - Bk' * Θs[k - 2] * Bk
# end

# # Backward recurrences
# Φs = Vector{Matrix{Float64}}(undef, K + 1)
# Φs[K + 1] = Matrix{Float64}(I, 4, 4)
# Φs[K] = G[((K - 1) * 4 + 1):(K * 4), ((K - 1) * 4 + 1):(K * 4)]
# for k in (K - 1):-1:1
#     Ak = G[((k - 1) * 4 + 1):(k * 4), ((k - 1) * 4 + 1):(k * 4)]
#     Bk = G[(k * 4 + 1):((k + 1) * 4), ((k - 1) * 4 + 1):(k * 4)]  # Actually B_k, which is below
#     Φs[k] = Ak * Φs[k + 1] - Bk * Φs[k + 2] * Bk'
# end

# # Compute diagonal and off-diagonal blocks of G^{-1}
# Δ_inv = inv(Θs[K])
# G_inv_diags = Vector{Matrix{Float64}}(undef, K)
# G_inv_off = Vector{Matrix{Float64}}(undef, K - 1)
# for k in 1:K
#     G_inv_diags[k] = Θs[k - 1] * Φs[k + 1] * Δ_inv
#     if k < K
#         Bk = G[(k * 4 + 1):((k + 1) * 4), ((k - 1) * 4 + 1):(k * 4)]
#         G_inv_off[k] = -Θs[k - 1] * Bk' * Φs[k + 2] * Δ_inv
#     end
# end

# display(G_inv[1:4, 1:4])
# display(G_inv_diags[1])
# println("Error: ", maximum(abs.(G_inv[1:4, 1:4] .- G_inv_diags[1])))

# - Diagonal blocks:

# $$
# \left(A^{-1}\right)_{i i}=\Theta_{i-1} \cdot \Phi_{i+1} \cdot \Delta^{-1}
# $$

# - Off-diagonal blocks:

# $$
# \left(A^{-1}\right)_{i, i+1}=-\Theta_{i-1} \cdot B_i^{\top} \cdot \Phi_{i+2} \cdot \Delta^{-1}
# $$

# See: https://iopscience.iop.org/article/10.1088/1749-4699/5/1/014009/meta
function compute_block_inverse_diagonal_and_offdiagonal(G, block_size, num_blocks)
    # Forward elimination: compute reduced matrices
    S_forward = Vector{Matrix{Float64}}(undef, num_blocks)
    S_forward[1] = zeros(block_size, block_size)

    for k in 2:num_blocks
        A_k = G[
            ((k - 2) * block_size + 1):((k - 1) * block_size),
            ((k - 2) * block_size + 1):((k - 1) * block_size),
        ]
        B_prev = G[
            ((k - 1) * block_size + 1):(k * block_size),
            ((k - 2) * block_size + 1):((k - 1) * block_size),
        ]

        S_forward[k] = B_prev * inv(A_k - S_forward[k - 1]) * B_prev'
    end

    # Backward elimination: compute reduced matrices  
    S_backward = Vector{Matrix{Float64}}(undef, num_blocks)
    S_backward[end] = zeros(block_size, block_size)

    for k in (num_blocks - 1):-1:1
        A_k = G[
            (k * block_size + 1):((k + 1) * block_size),
            (k * block_size + 1):((k + 1) * block_size),
        ]
        B_km1 = G[
            (k * block_size + 1):((k + 1) * block_size),
            ((k - 1) * block_size + 1):(k * block_size),
        ]

        S_backward[k] = B_km1' * inv(A_k - S_backward[k + 1]) * B_km1
    end

    # Compute diagonal blocks
    diag_blocks = Vector{Matrix{Float64}}(undef, num_blocks)
    for k in 1:num_blocks
        Ak = G[
            ((k - 1) * block_size + 1):(k * block_size),
            ((k - 1) * block_size + 1):(k * block_size),
        ]
        diag_blocks[k] = inv(Ak - S_forward[k] - S_backward[k])
    end

    # Compute off-diagonal blocks
    off_diag_blocks = Vector{Matrix{Float64}}(undef, num_blocks - 1)
    for k in 1:(num_blocks - 1)
        # m = k + 1
        # n = k
        Am = G[
            (k * block_size + 1):((k + 1) * block_size),
            (k * block_size + 1):((k + 1) * block_size),
        ]
        Bn = G[
            (k * block_size + 1):((k + 1) * block_size),
            ((k - 1) * block_size + 1):(k * block_size),
        ]
        off_diag_blocks[k] = -inv(Am - S_backward[k + 1]) * Bn * diag_blocks[k]
    end

    return diag_blocks, off_diag_blocks
end

diag_blocks, off_diag_blocks = compute_block_inverse_diagonal_and_offdiagonal(G, 4, K)

# Check correctness
# println("Error: ", maximum(abs.(G_inv[1:4, 1:4].- diag_blocks[1])))
# println("Error: ", maximum(abs.(G_inv[5:8, 5:8].- diag_blocks[2])))
# println("Error: ", maximum(abs.(G_inv[1:4, 5:8].- off_diag_blocks[1])))

diag_errors = Vector{Float64}(undef, K)
for k in 1:K
    diag_errors[k] = maximum(
        abs.(
            G_inv[((k - 1) * 4 + 1):(k * 4), ((k - 1) * 4 + 1):(k * 4)] .-
            diag_blocks[k]
        ),
    )
end
println("Maximum diagonal errors: ", maximum(diag_errors))

off_diag_errors = Vector{Float64}(undef, K - 1)
for k in 1:(K - 1)
    off_diag_errors[k] = maximum(
        abs.(
            G_inv[(4k + 1):((k + 1) * 4), ((k - 1) * 4 + 1):(k * 4)] .- off_diag_blocks[k]
        ),
    )
end
println("Maximum off-diagonal errors: ", maximum(off_diag_errors))
