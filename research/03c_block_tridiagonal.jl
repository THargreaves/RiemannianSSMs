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

display(@benchmark cholesky($G))

function block_tridiagonal_cholesky(
    diag_blocks::Vector{<:SMatrix{D,D,T}}, supdiag_blocks::Vector{<:SMatrix{D,D,T}}
) where {D,T}
    K = length(diag_blocks)
    @assert length(supdiag_blocks) == K - 1 "supdiag_blocks must have length K-1"

    # Outputs
    # TODO: can be done in-place
    U_diag = Vector{UpperTriangular{T,SMatrix{D,D,T,D^2}}}(undef, K)
    U_supdiag = Vector{SMatrix{D,D,T,D^2}}(undef, K - 1)

    # First block
    U_diag[1] = cholesky(diag_blocks[1]).U
    if K > 1
        U_supdiag[1] = U_diag[1]' \ supdiag_blocks[1]
    end

    # Remaining blocks
    @inbounds for k in 2:K
        # Schur complement update
        Ã = diag_blocks[k] - U_supdiag[k - 1]' * U_supdiag[k - 1]

        # Cholesky
        U_diag[k] = cholesky(Ã).U

        # Superdiagonal, if not last block
        if k < K
            U_supdiag[k] = U_diag[k]' \ supdiag_blocks[k]
        end
    end

    return U_diag, U_supdiag
end

G_diag = Vector{SMatrix{4,4,Float64,16}}(undef, K);
G_supdiag = Vector{SMatrix{4,4,Float64,16}}(undef, K - 1);
for k in 1:K
    i1 = (k - 1) * 4 + 1
    i2 = k * 4
    G_diag[k] = SMatrix{4,4,Float64}(G[i1:i2, i1:i2])

    if k < K
        j1 = k * 4 + 1
        j2 = (k + 1) * 4
        G_supdiag[k] = SMatrix{4,4,Float64}(G[i1:i2, j1:j2])
    end
end;

U_diag, U_supdiag = block_tridiagonal_cholesky(G_diag, G_supdiag)

U_diag[14] ≈ G_chol.U[53:56, 53:56]
U_supdiag[14] ≈ G_chol.U[53:56, 57:60]

display(@benchmark block_tridiagonal_cholesky($G_diag, $G_supdiag))

@profview begin
    for _ in 1:100000
        block_tridiagonal_cholesky(G_diag, G_supdiag)
    end
end

# Other ops we need:
# - Forward and backward solves
# - Forward multiplication 

println("Triangular solve:")

y = rand(rng, 4 * K);
display(@benchmark $(G_chol.U) \ $y)

function block_upper_solve(
    U_diag::Vector{<:UpperTriangular{T,<:SMatrix{D,D,T}}},
    U_supdiag::Vector{<:SMatrix{D,D,T}},
    y::AbstractVector{SVector{D,T}},
) where {D,T}
    K = length(U_diag)
    @assert length(y) == K
    @assert length(U_supdiag) == max(0, K - 1)

    x = Vector{SVector{D,T}}(undef, K)

    # Solve last block: U_K * x_K = y_K
    # TOOD: can probably remove these type conversions
    # x[K] = SVector{D,T}(UpperTriangular(U_diag[K]) \ y[K])
    x[K] = U_diag[K] \ y[K]

    # Back-substitution for upper triangular (K-1 downto 1):
    # TODO: likewise, could be done in-place
    @inbounds for i in (K - 1):-1:1
        rhs = y[i] - U_supdiag[i] * x[i + 1]     # SMatrix * SVector -> SVector
        # x[i] = SVector{D,T}(UpperTriangular(U_diag[i]) \ rhs)
        x[i] = U_diag[i] \ rhs
    end

    return x
end

y_sv = reinterpret(SVector{4,Float64}, y)
# y_sv = SVector{K, SVector{4,Float64}}([SVector{4, Float64}(y[(i - 1) * 4 + 1:i * 4]) for i in 1:K])
x = block_upper_solve(U_diag, U_supdiag, y_sv)
x_truth = G_chol.U \ y
println("Upper solve pass: ", x[3] ≈ x_truth[9:12])

# Benchmark 
display(@benchmark block_upper_solve($U_diag, $U_supdiag, $y_sv))

@profview begin
    for _ in 1:1000000
        block_upper_solve(U_diag, U_supdiag, y_sv)
    end
end

function block_lower_solve(
    U_diag::Vector{<:UpperTriangular{T,<:SMatrix{D,D,T}}},
    U_supdiag::Vector{<:SMatrix{D,D,T}},
    y::AbstractVector{SVector{D,T}},
) where {D,T}
    K = length(U_diag)
    @assert length(y) == K
    @assert length(U_supdiag) == max(0, K - 1)

    x = Vector{SVector{D,T}}(undef, K)

    # first block
    # x[1] = UpperTriangular(U_diag[1])' \ y[1]
    x[1] = U_diag[1]' \ y[1]

    # forward substitution
    @inbounds for k in 2:K
        rhs = y[k] - U_supdiag[k - 1]' * x[k - 1]
        # x[k] = UpperTriangular(U_diag[k])' \ rhs
        x[k] = U_diag[k]' \ rhs
    end

    return x
end

x = block_lower_solve(U_diag, U_supdiag, y_sv);
x_truth = G_chol.U' \ y;
println("Lower solve pass: ", x[3] ≈ x_truth[9:12])

# Benchmark
display(@benchmark block_lower_solve($U_diag, $U_supdiag, $y_sv))

function block_lower_mul(
    U_diag::Vector{<:UpperTriangular{T,<:SMatrix{D,D,T}}},
    U_supdiag::Vector{<:SMatrix{D,D,T}},
    y::AbstractVector{SVector{D,T}},
) where {D,T}
    K = length(U_diag)
    @assert length(y) == K
    @assert length(U_supdiag) == max(0, K - 1)

    z = Vector{SVector{D,T}}(undef, K)

    # first block
    z[1] = U_diag[1]' * y[1]

    # remaining blocks
    @inbounds for k in 2:K
        z[k] = U_diag[k]' * y[k] + U_supdiag[k - 1]' * y[k - 1]
    end

    return z
end

z = block_lower_mul(U_diag, U_supdiag, y_sv);
z_truth = G_chol.U' * y;
println("Lower mul pass: ", z[3] ≈ z_truth[9:12])

# Benchmark
display(@benchmark ($(G_chol.L) * $y))
display(@benchmark block_lower_mul($U_diag, $U_supdiag, $y_sv))
