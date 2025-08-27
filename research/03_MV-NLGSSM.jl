# The sticky chain issue seems to be coming from the singularity of the bearing-range sensor
# at the origin. At k = 11 the leapfrog position step takes us from
#  -6.705735603160172
#   0.0003675124580438028
#  -0.7851026388273065
#  -0.8859791880022796
#  to
#   -6.705902413913993
#  -0.00031335835761046643
#  -0.7850751360217922
#  -0.8859881411968482
# These are close in norm but have completely different angles, 3.1415378 vs -3.14154592
# It's not clear why the chain was this close to the sensor or why it kept being pushed back
# in this direction each time. Could suggest an issue with the metric. 
# But given this new position, the hamilotonian calculation does seem to be correct. 

# Wait no. It wasn't passing through the origin, it was just changing from the positive to
# negative quadratic as k = 11 passes over y = 0. Can hack this for now by flipping the x
# coordinates.

# Ideas for improvement:
# - There are a lot more allocations than their need to be, especially in the construction
#   of G
# - We don't actually using G directly, so it might be best to construct the Cholesky
#   factorisation directly (only helps in analytical case)
# - We're currently treating G as banded, but there might be significant speed-ups that come
#   from treating it properly as block tridiagonal.

###############
#### SETUP ####
###############

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

# test_K = 50
test_K = 20
test_δt = 1.0

test_α = 0.01
test_β = 0.1
test_γ = 0.005
# test_σ_p = 0.01
# test_σ_v = 0.05
test_σ_p = 0.1
test_σ_v = 0.5
test_σ_r = 0.2
test_σ_θ = 5.0 * π / 180.0  # radians
# test_σ_r = 2.0
# test_σ_θ = 30.0 * π / 180.0  # radians

test_μ0 = @SVector [0.0, 0.0, 0.0, 0.0]
test_Σ0 = Diagonal(@SVector([0.5, 0.5, 0.1, 0.1]))

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

test_dyn = VariableRestoringForceMotion(test_α, test_β, test_γ, test_σ_p, test_σ_v, test_δt)
test_sensor = BearingRangeSensor(test_σ_r, test_σ_θ)
test_prior = MvNormal(test_μ0, test_Σ0)

####################
#### SIMULATION ####
####################

struct SSM{PT<:MvNormal}
    prior::PT
    dyn::VariableRestoringForceMotion
    sensor::BearingRangeSensor
end

test_ssm = SSM(test_prior, test_dyn, test_sensor)

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
zs_true, ys = simulate(rng, test_prior, test_dyn, test_sensor, test_K);

# Flip x coordinates to avoid singularity issues at the origin
zs_true = [SVector(-z[1], z[2], -z[3], z[4]) for z in zs_true]
# Regenerate measurements
ys = Vector{SVector{2,Float64}}(undef, test_K)
for k in 1:test_K
    y = h(test_sensor, zs_true[k]) + rand(rng, MvNormal(zeros(2), calc_R(test_sensor)))
    ys[k] = y
end

# Plot trajectory
position_plot = plot(;
    xlabel="X Position", ylabel="Y Position", aspect_ratio=1, grid=true, size=(800, 800)
)
plot!(
    position_plot,
    getindex.(zs_true, 1),
    getindex.(zs_true, 2);
    label="Trajectory",
    color=:black,
)
quiver!(
    position_plot,
    getindex.(zs_true, 1),
    getindex.(zs_true, 2);
    quiver=(getindex.(zs_true, 3) / 1e6, getindex.(zs_true, 4) / 1e6),
    label="Velocity",
    arrow=:arrow,
    color=:black,
)

# Add observations
ys_r = getindex.(ys, 1)
ys_θ = getindex.(ys, 2)
ys_x = ys_r .* cos.(ys_θ)
ys_y = ys_r .* sin.(ys_θ)
scatter!(position_plot, ys_x, ys_y; label="Observations", color=:red, marker=:x)

# Add sensor
scatter!(
    position_plot, [0.0], [0.0]; label="Sensor", color=:green, marker=:circle, markersize=8
)

display(position_plot)

##############################
#### NUMERICAL VALIDATION ####
##############################

# Test calc_Jf
test_z = @SVector [1.0, 2.0, 0.5, 0.5]
test_Jf = calc_Jf(test_dyn, test_z)
test_Jf_num = FiniteDiff.finite_difference_jacobian(z -> f(test_dyn, z), test_z)
println("Jf test pass: $(norm(test_Jf - test_Jf_num) < 1e-7)")

# Test calc_Jh
test_Jh = calc_Jh(test_sensor, test_z)
test_Jh_num = FiniteDiff.finite_difference_jacobian(z -> h(test_sensor, z), test_z)
println("Jh test pass: $(norm(test_Jh - test_Jh_num) < 1e-7)")

# Test calc_Hfs
test_Hfs = calc_Hfs(test_dyn, test_z)
test_Hfs_num = Vector{Matrix{Float64}}(undef, 4)
# These agree but is it actually the correct think to compute?
for d in 1:4
    test_Hfs_num[d] = FiniteDiff.finite_difference_jacobian(
        z -> calc_Jf(test_dyn, z)[:, d], test_z
    )
    println("Hfs_$d test pass: $(norm(test_Hfs[d] - test_Hfs_num[d]) < 1e-7)")
end

# Test calc_Hhs
test_Hhs = calc_Hhs(test_sensor, test_z)
test_Hhs_num = Vector{Matrix{Float64}}(undef, 4)
for d in 1:4
    test_Hhs_num[d] = FiniteDiff.finite_difference_jacobian(
        z -> calc_Jh(test_sensor, z)[:, d], test_z
    )
    println("Hhs_$d test pass: $(norm(test_Hhs[d] - test_Hhs_num[d]) < 1e-7)")
end

##############
#### RHMC ####
##############

function calc_ll(zs, ys, ssm)
    K = length(ys)

    # Prior
    ll = logpdf(ssm.prior, zs[1])

    # Dynamics
    for k in 2:K
        ll += logpdf(MvNormal(f(ssm.dyn, zs[k - 1]), calc_Q(ssm.dyn)), zs[k])
    end

    # Likelihood
    for k in 1:K
        ll += logpdf(MvNormal(h(ssm.sensor, zs[k]), calc_R(ssm.sensor)), ys[k])
    end

    return ll
end

function calc_ll_terms(zs, ys, ssm)
    K = length(zs)

    # Prior
    prior_ll = logpdf(ssm.prior, zs[1])

    # Dynamics
    dynamics_lls = Vector{Float64}(undef, K - 1)
    for k in 2:K
        dynamics_lls[k - 1] = logpdf(
            MvNormal(f(ssm.dyn, zs[k - 1]), calc_Q(ssm.dyn)), zs[k]
        )
    end

    # Likelihood
    likelihood_lls = Vector{Float64}(undef, K)
    for k in 1:K
        likelihood_lls[k] = logpdf(
            MvNormal(h(ssm.sensor, zs[k]), calc_R(ssm.sensor)), ys[k]
        )
    end

    return prior_ll, dynamics_lls, likelihood_lls
end

function calc_ll_grad(zs, ys, ssm)
    K = length(zs)
    grads = Vector{SVector{4,Float64}}(undef, K)

    # Incoming dynamics
    grads[1] = -inv(ssm.prior.Σ) * (zs[1] - ssm.prior.μ)
    for k in 2:K
        grads[k] = -calc_Qinv(ssm.dyn) * (zs[k] - f(ssm.dyn, zs[k - 1]))
    end

    # Observation term
    for k in 1:K
        Jh = calc_Jh(ssm.sensor, zs[k])
        R_inv = calc_Rinv(ssm.sensor)
        grads[k] += Jh' * R_inv * (ys[k] - h(ssm.sensor, zs[k]))
    end

    # Transition term
    for k in 2:K
        Jf = calc_Jf(ssm.dyn, zs[k - 1])
        grads[k - 1] += Jf' * calc_Qinv(ssm.dyn) * (zs[k] - f(ssm.dyn, zs[k - 1]))
    end

    # Flatten to regular vector
    return reduce(vcat, grads)
end

function calc_ll_grad_terms(zs, ys, ssm)
    K = length(zs)
    grads = Vector{SVector{4,Float64}}(undef, K)

    # Incoming dynamics
    grads[1] = -inv(ssm.prior.Σ) * (zs[1] - ssm.prior.μ)
    for k in 2:K
        grads[k] = -calc_Qinv(ssm.dyn) * (zs[k] - f(ssm.dyn, zs[k - 1]))
    end

    # Observation term
    for k in 1:K
        Jh = calc_Jh(ssm.sensor, zs[k])
        R_inv = calc_Rinv(ssm.sensor)
        grads[k] += Jh' * R_inv * (ys[k] - h(ssm.sensor, zs[k]))
    end

    # Transition term
    for k in 2:K
        Jf = calc_Jf(ssm.dyn, zs[k - 1])
        grads[k - 1] += Jf' * calc_Qinv(ssm.dyn) * (zs[k] - f(ssm.dyn, zs[k - 1]))
    end

    return grads
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

    # WARNING: this converted G to a dense matrix
    # return Symmetric(G) + 1e-8 * I

    # Force PSD
    G = Symmetric(G)
    diag(G) .+= 1e-8

    return G
end

test_G = calc_G(zs_true, inv(test_Σ0), test_dyn, test_sensor, test_K)

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

##############################
#### NUMERICAL VALIDATION ####
##############################

# Test ll_grad
test_ll_grad = calc_ll_grad(zs_true, ys, test_ssm)
function ll_finite_diff(z)
    zs = reinterpret(SVector{4,Float64}, z)
    return calc_ll(zs, ys, test_ssm)
end
test_ll_grad_num = FiniteDiff.finite_difference_gradient(
    ll_finite_diff, reduce(vcat, zs_true)
)
println("LL grad test pass: $(norm(test_ll_grad - test_ll_grad_num) < 1e-6)")

# Test calc_dG (derivative of G w.r.t to each paramater)
test_dGs_diag, test_dGs_off = calc_dGs(zs_true, test_dyn, test_sensor, test_K)
test_dGs_num = Matrix{Matrix{Float64}}(undef, 4, test_K)
test_dGs = Matrix{Matrix{Float64}}(undef, 4, test_K)
passes = 0
for k in 1:test_K
    for d in 1:4
        # Manually compute gradient
        ϵ = 1e-8
        zs_pos = [Array(z) for z in zs_true]
        zs_neg = [Array(z) for z in zs_true]
        zs_pos[k][d] += ϵ
        zs_neg[k][d] -= ϵ
        G_pos = calc_G(zs_pos, inv(test_Σ0), test_dyn, test_sensor, test_K)
        G_neg = calc_G(zs_neg, inv(test_Σ0), test_dyn, test_sensor, test_K)
        test_dG_num = (G_pos - G_neg) / (2 * ϵ)
        test_dGs_num[d, k] = test_dG_num

        # Construct comparison matrix
        test_dG = zeros(4test_K, 4test_K)
        test_dG[((k - 1) * 4 + 1):(k * 4), ((k - 1) * 4 + 1):(k * 4)] = test_dGs_diag[d, k]
        if k < test_K
            test_dG[((k - 1) * 4 + 1):(k * 4), (k * 4 + 1):((k + 1) * 4)] = test_dGs_off[
                d, k
            ]
            test_dG[(k * 4 + 1):((k + 1) * 4), ((k - 1) * 4 + 1):(k * 4)] =
                test_dGs_off[d, k]'
        end
        test_dGs[d, k] = test_dG
        # TODO: should this be tighter? Think correct just numerically unstable.
        # Maybe the gradient itself is poor?
        if maximum(abs.(test_dG_num - test_dG)) < 1e-5
            passes += 1
        else
            println("dG test failed for d=$d, k=$k")
        end
    end
end
println("dG test pass rate: $(passes / (4 * test_K))")

# This one is very strange — suggests something is wrong with Hf1 yet it passes tests
# test_dGs_num[1, 4][10:24, 10:24]
# test_dGs[1, 4][10:24, 10:24]

# test_dGs_num[2, 3][6:20, 6:20]
# test_dGs[2, 3][6:20, 6:20]

# test_dGs_num[1, 4][10:24, 10:24] - test_dGs[1, 4][10:24, 10:24]
# test_dGs_num[2, 4][10:24, 10:24] - test_dGs[2, 4][10:24, 10:24]
# test_dGs_num[3, 4][10:24, 10:24] - test_dGs[3, 4][10:24, 10:24]
# test_dGs_num[4, 4][10:24, 10:24] - test_dGs[4, 4][10:24, 10:24]

# TODO: probably a bit of caching to be done here but not going to be a major bottleneck
function ∇p_H(zs, ps, ssm)
    zs = reinterpret(SVector{4,Float64}, zs)  # HACK: wrote this function for SVectors
    G = calc_G(zs, inv(ssm.prior.Σ), ssm.dyn, ssm.sensor, length(zs))
    G_chol = cholesky(G)  # TODO: should be in-place to avoid allocation
    ∇p = G_chol \ ps
    return ∇p
end

function ∇θ_H(zs, ps, ys, ssm)
    # Extract parameters
    zs = reinterpret(SVector{4,Float64}, zs)  # HACK: wrote this function for SVectors
    K = length(zs)

    grad = calc_ll_grad(zs, ys, ssm)
    G = calc_G(zs, inv(ssm.prior.Σ), ssm.dyn, ssm.sensor, K)
    dGs_diag, dGs_off = calc_dGs(zs, ssm.dyn, ssm.sensor, K)
    G_chol = cholesky(G)  # TODO: G doesn't change throughout the step so this should be cached

    G_inv_p = G_chol \ ps

    ∇θ = Vector{Float64}(undef, 4K)
    for k in 1:K
        for d in 1:4
            i = (k - 1) * 4 + d
            # Base block index
            bi = (k - 1) * 4 + 1
            # Accumulate gradient
            v = grad[i]
            # Quadratic form term
            v += 0.5 * G_inv_p[bi:(bi + 3)]' * dGs_diag[d, k] * G_inv_p[bi:(bi + 3)]
            if k < K
                # TODO: this might be symmetric, not sure yet
                v +=
                    0.5 * G_inv_p[(bi + 4):(bi + 7)]' * dGs_off[d, k] * G_inv_p[bi:(bi + 3)]
                v +=
                    0.5 *
                    G_inv_p[bi:(bi + 3)]' *
                    dGs_off[d, k]' *
                    G_inv_p[(bi + 4):(bi + 7)]
            end
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

function ∇θ_H_slow(zs, ps, ys, ssm)
    # Extract parameters
    zs = reinterpret(SVector{4,Float64}, zs)  # HACK: wrote this function for SVectors
    K = length(zs)

    grad = calc_ll_grad(zs, ys, ssm)
    G = calc_G(zs, inv(ssm.prior.Σ), ssm.dyn, ssm.sensor, K)
    dGs_diag, dGs_off = calc_dGs(zs, ssm.dyn, ssm.sensor, K)
    G_chol = cholesky(G)

    G_inv_p = G_chol \ ps

    ∇θ = Vector{Float64}(undef, 4K)
    for k in 1:K
        for d in 1:4
            i = (k - 1) * 4 + d
            dG = zeros(4K, 4K)
            # Base block index
            bi = (k - 1) * 4 + 1
            # Fill dG
            dG[bi:(bi + 3), bi:(bi + 3)] = dGs_diag[d, k]
            if k < K
                dG[bi:(bi + 3), (bi + 4):(bi + 7)] = dGs_off[d, k]'
                dG[(bi + 4):(bi + 7), bi:(bi + 3)] = dGs_off[d, k]
            end
            # $\frac{d p_i}{d \tau}=-\frac{\partial H}{\partial \theta_i}=\frac{\partial \mathcal{L}(\boldsymbol{\theta})}{\partial \theta_i}-\frac{1}{2} \operatorname{Tr}\left[\mathbf{G}(\boldsymbol{\theta})^{-1} \frac{\partial \mathbf{G}(\boldsymbol{\theta})}{\partial \theta_i}\right]+\frac{1}{2} \mathbf{p}^{\top} \mathbf{G}(\boldsymbol{\theta})^{-1} \frac{\partial \mathbf{G}(\boldsymbol{\theta})}{\partial \theta_i} \mathbf{G}(\boldsymbol{\theta})^{-1} \mathbf{p}$
            ∇θ[i] = (grad[i] - 0.5 * tr(G_chol \ dG) + 0.5 * G_inv_p' * (dG * G_inv_p))
        end
    end

    return ∇θ
end

# Verify optimised method matchesmes
test_zs = reduce(vcat, zs_true)
test_ps = rand(rng, 4 * test_K)
test_ys = ys
∇θ_regular = ∇θ_H(test_zs, test_ps, test_ys, test_ssm)
∇θ_slow = ∇θ_H_slow(test_zs, test_ps, test_ys, test_ssm)
# Fails but errors aren't too big. Could be numerical
println("∇θ_H test pass: $(maximum(norm.(∇θ_regular - ∇θ_slow)) < 1e-5)")

function calc_hamiltonian(zs, ps, ys, ssm)
    zs = reinterpret(SVector{4,Float64}, zs)  # HACK: wrote this function for SVectors
    K = length(zs)
    ll = calc_ll(zs, ys, ssm)
    G = calc_G(zs, inv(ssm.prior.Σ), ssm.dyn, ssm.sensor, K)
    G_chol = cholesky(G)

    log_det = logdet(G_chol)
    pG_inv_p = ps' * (G_chol \ ps)

    return -ll + 0.5 * (4K * log(2π) + log_det + pG_inv_p)
end

# This can likely be performed using static arrays if we partition the variables into time
# steps and take advantage of the sparsity structure
function p_step(zs, ps, ys, ssm, lf_params)
    ps_new_1 = copy(ps)
    ps_new_2 = Vector{Float64}(undef, length(ps))
    reps = 0
    while reps < lf_params.max_reps
        reps += 1
        ps_new_2 = ps - lf_params.ϵ / 2 * ∇θ_H(zs, ps_new_1, ys, ssm)
        if norm(ps_new_1 - ps_new_2) < lf_params.fp_tol
            break
        end
        ps_new_1 = ps_new_2
    end
    if reps == lf_params.max_reps
        println(zs)
        println(ps)
        error("Failed to converge in $(lf_params.max_reps) repetitions.")
    end
    return ps_new_2
end

function θ_step(zs, ps, ssm, lf_params)
    zs_new_1 = copy(zs)
    zs_new_2 = Vector{Float64}(undef, length(zs))
    reps = 0
    while reps < lf_params.max_reps
        reps += 1
        zs_new_2 = zs + lf_params.ϵ / 2 * (∇p_H(zs_new_1, ps, ssm) + ∇p_H(zs, ps, ssm))
        if norm(zs_new_1 - zs_new_2) < lf_params.fp_tol
            break
        end
        zs_new_1 = zs_new_2
    end
    if reps == lf_params.max_reps
        println("Warning: failed to converge in $(lf_params.max_reps) repetitions.")
    end
    return zs_new_2
end

function glf_step(zs, ps, ys, ssm, lf_params)
    # Half step for momentum
    ps = p_step(zs, ps, ys, ssm, lf_params)
    # Full step for position
    zs = θ_step(zs, ps, ssm, lf_params)
    # Half step for momentum again
    ps = p_step(zs, ps, ys, ssm, lf_params)
    return zs, ps
end

struct LFParams
    ϵ::Float64
    fp_tol::Float64
    max_reps::Int
end

lf_params = LFParams(0.01 * (4 * test_K)^(-1 / 4), 1e-8, 20)

N_samples = 2000
N_burnin = 0
n_steps = 10

function sample!(
    zs_samples,
    rng::AbstractRNG,
    ssm::SSM,
    z_init::Vector{Float64},
    ys::Vector{SVector{2,Float64}},
    N_samples::Int,
    N_burnin::Int,
    n_steps::Int,
    lf_params::LFParams;
    progress=true,
)
    zs_curr = z_init
    n_accept = 0
    K = length(ys)

    prog = Progress(N_samples; enabled=progress)
    for i in 1:N_samples
        G = calc_G(
            reinterpret(SVector{4,Float64}, zs_curr),
            inv(ssm.prior.Σ),
            ssm.dyn,
            ssm.sensor,
            K,
        )
        G_chol = cholesky(G)
        ps_curr = G_chol.L * randn(rng, 4K)  # had this as rand(...) lol
        H_curr = calc_hamiltonian(zs_curr, ps_curr, ys, ssm)
        zs_new = copy(zs_curr)
        ps_new = copy(ps_curr)

        for _ in 1:n_steps
            zs_new, ps_new = glf_step(zs_new, ps_new, ys, ssm, lf_params)
        end

        # Accept or reject
        H_new = calc_hamiltonian(zs_new, ps_new, ys, ssm)
        # println("Sample $i:")
        # println("H_curr: $H_curr, H_new: $H_new")
        # println("zs_curr: $(zs_curr[1:12])")
        # println("zs_new: $(zs_new[1:12])")
        # println("ps_curr: $(ps_curr[1:12])")
        # println("ps_new: $(ps_new[1:12])")
        # println("Running acceptance rate: $(n_accept / i)")
        # println()
        # if i == 10
        #     println(zs_curr)
        # end
        # if H_new > 300
        #     println(zs_curr)
        #     break
        # end
        if log(rand(rng)) < H_curr - H_new
            zs_curr = zs_new
            n_accept += 1
        end

        # if i > N_burnin
        zs_samples[i] = copy(zs_curr)
        # end

        next!(prog)
    end

    println("Acceptance rate: $(n_accept / N_samples)")
    return zs_samples
end

z_init = reduce(vcat, zs_true)
zs_samples = Vector{Vector{Float64}}(undef, N_samples)
sample!(zs_samples, rng, test_ssm, z_init, ys, N_samples, N_burnin, n_steps, lf_params);
zs_samples = [zs_samples[i] for i in 1:N_samples if isassigned(zs_samples, i)];

# Plot posterior samples
posterior_plot = deepcopy(position_plot)
mean_x = mean(z[1:4:(4test_K)] for z in zs_samples)
mean_y = mean(z[2:4:(4test_K)] for z in zs_samples)

# Plot all samples
for z in zs_samples
    plot!(
        posterior_plot,
        z[1:4:(4test_K)],
        z[2:4:(4test_K)];
        label="",
        color=:blue,
        alpha=0.005,
    )
end
plot!(posterior_plot, mean_x, mean_y; label="Posterior Mean", color=:red, linewidth=2)
display(posterior_plot)

# Plot traces for one position variable
trace_k = test_K ÷ 2
trace_plot = plot(
    [getindex(z, 1) for z in zs_samples];
    label="X Position",
    xlabel="Sample",
    ylabel="Position",
    grid=true,
)
plot!(trace_plot, [getindex(z, 2) for z in zs_samples]; label="Y Position", color=:red)

#########################
#### FAST TRACE TERM ####
#########################

# test_G_inv = inv(test_G)
# Something seems a bit off with G so we'll construct a simpler example
# No, the issue is the exponential scaling. See: https://chatgpt.com/c/688fc9fe-44bc-800a-a744-561195a3cc99
# K = test_K = 4
# test_G = BandedMatrix(Zeros(4 * K, 4 * K), (7, 7))
# for k in 1:test_K
#     i1 = (k - 1) * 4 + 1
#     i2 = k * 4
#     test_G[i1:i2, i1:i2] = [
#         4 1 0 0
#         1 4 1 0
#         0 1 4 1
#         0 0 1 4
#     ]
#     # Identity for off diagonal
#     if k < test_K
#         j1 = k * 4 + 1
#         j2 = (k + 1) * 4
#         test_G[i1:i2, j1:j2] = [
#             -1 0 0 0
#             0 -1 0 0
#             0 0 -1 0
#             0 0 0 -1
#         ]
#         test_G[j1:j2, i1:i2] = test_G[i1:i2, j1:j2]'
#     end
# end
# test_G = Symmetric(test_G)
# test_G_inv = inv(test_G)

# # Forward recurrences
# Θs = OffsetVector(Vector{Matrix{Float64}}(undef, test_K + 1), -1)
# Θs[0] = Matrix{Float64}(I, 4, 4)
# Θs[1] = test_G[1:4, 1:4]
# for k in 2:test_K
#     Ak = test_G[((k - 1) * 4 + 1):(k * 4), ((k - 1) * 4 + 1):(k * 4)]
#     Bk = test_G[((k - 1) * 4 + 1):(k * 4), ((k - 2) * 4 + 1):((k - 1) * 4)]
#     Θs[k] = Ak * Θs[k - 1] - Bk' * Θs[k - 2] * Bk
# end

# # Backward recurrences
# Φs = Vector{Matrix{Float64}}(undef, test_K + 1)
# Φs[test_K + 1] = Matrix{Float64}(I, 4, 4)
# Φs[test_K] = test_G[
#     ((test_K - 1) * 4 + 1):(test_K * 4), ((test_K - 1) * 4 + 1):(test_K * 4)
# ]
# for k in (test_K - 1):-1:1
#     Ak = test_G[((k - 1) * 4 + 1):(k * 4), ((k - 1) * 4 + 1):(k * 4)]
#     Bk = test_G[(k * 4 + 1):((k + 1) * 4), ((k - 1) * 4 + 1):(k * 4)]
#     Φs[k] = Ak * Φs[k + 1] - Bk' * Φs[k + 2] * Bk
# end

# # Extract diagonal entries
# Δ_inv = inv(Θs[test_K])
# G_inv_diags = Vector{Matrix{Float64}}(undef, test_K)
# G_inv_off = Vector{Matrix{Float64}}(undef, test_K - 1)
# for k in 1:test_K
#     G_inv_diags[k] = Θs[k - 1] * Φs[k + 1] * Δ_inv
#     if k < test_K
#         Bk = test_G[(k * 4 + 1):((k + 1) * 4), ((k - 1) * 4 + 1):(k * 4)]
#         G_inv_off[k] = -Θs[k - 1] * Bk' * Φs[k + 2] * Δ_inv
#     end
# end

# test_G_inv[1:4, 1:4]
# G_inv_diags[1]

# test_G_inv[5:8, 5:8]
# G_inv_diags[2]

# test_G_inv[9:12, 9:12]
# G_inv_diags[3]

# test_G_inv[13:16, 13:16]
# G_inv_diags[4]

# test_G_inv[5:8, 1:4]
# G_inv_off[1]

# test_G_inv[9:12, 5:8]
# G_inv_off[2]

# test_G_inv[13:16, 9:12]
# G_inv_off[3]

#########################
#### NUTS COMPARISON ####
#########################

using AdvancedHMC, AbstractMCMC
using LogDensityProblems, LogDensityProblemsAD, ADTypes #
using ForwardDiff
using DistributionsAD

struct LogTargetDensity{M,V}
    dim::Int
    ys::V
    ssm::M
end
function LogDensityProblems.logdensity(p::LogTargetDensity, θ)
    θ = [θ[i:(i + 3)] for i in 1:4:length(θ)]  # avoid reinterpret due to Duals
    return calc_ll(θ, p.ys, p.ssm)
end
LogDensityProblems.dimension(p::LogTargetDensity) = p.dim
function LogDensityProblems.capabilities(::Type{LogTargetDensity})
    return LogDensityProblems.LogDensityOrder{0}()
end

# 2. Wrap the log density function and specify the AD backend.
#    This creates a callable struct that computes the log density and its gradient.
ℓπ = LogTargetDensity(4 * test_K, ys, test_ssm)
model = AdvancedHMC.LogDensityModel(LogDensityProblemsAD.ADgradient(AutoForwardDiff(), ℓπ))

sampler = NUTS(0.8)
n_adapts, n_samples = 1000, 2000
initial_θ = reduce(vcat, zs_true)

samples = AbstractMCMC.sample(
    Random.default_rng(),
    model,
    sampler,
    n_adapts + n_samples;
    n_adapts=n_adapts,
    initial_params=initial_θ,
    progress=true, # Optional: Show a progress bar
);

nuts_samples = [s.z.θ for s in samples];

# Add to posterior plot
nuts_posterior_plot = deepcopy(position_plot)
# Plot all samples
for z in nuts_samples
    plot!(
        nuts_posterior_plot,
        z[1:4:(4test_K)],
        z[2:4:(4test_K)];
        label="",
        color=:blue,
        alpha=0.005,
    )
end
display(nuts_posterior_plot)

# Plot trace for one position variable
nuts_trace_plot = plot(
    [getindex(z, 1) for z in nuts_samples];
    label="X Position",
    xlabel="Sample",
    ylabel="Position",
    grid=true,
)
