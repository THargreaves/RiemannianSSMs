"""
Debug script to visualize leapfrog trajectories step-by-step.

This plots the state trajectory after each leapfrog step to see where
numerical instabilities occur.
"""

using Distributions
using LinearAlgebra
using Printf
using Random
using Plots
using SparseArrays
using StaticArrays

using AdvancedHMC
using LogDensityProblems

using RiemannianSSMs
using RiemannianSSMs.MCRHMC

# ============================================================================
# Setup (same as test script)
# ============================================================================

SEED = 42
rng = MersenneTwister(SEED)

D = 2
P = 1
δt = 0.1
K_y = 10
N_obs = 5
K = N_obs * K_y
obs_indices = collect(K_y:K_y:K)
n = K * D + P

μ0 = @SVector [1.0, 0.0]
Σ0 = Diagonal(@SVector([0.1, 0.1]))
prior = MvNormal(μ0, Σ0)

μ_true = 1.0
θ_true = @SVector [log(μ_true)]

σ_u = 0.1
σ_v = 0.1
dyn = VanDerPolDynamicsParam(σ_u, σ_v, δt)

σ_obs = 0.05
obs = PartialLinearObservationParam(σ_obs)

ssm = SSMParam(prior, dyn, obs)

prior_mean = @SVector [log(μ_true)]
prior_var = @SVector [4.0]
prior_prec = Vector(1.0 ./ prior_var)

function simulate(rng::AbstractRNG, ssm, θ, K::Int, obs_indices)
    zs = Vector{SVector{2,Float64}}(undef, K)
    ys = Vector{SVector{1,Float64}}(undef, length(obs_indices))
    obs_idx = 1
    for k in 1:K
        if k == 1
            z = SVector{2,Float64}(rand(rng, ssm.prior))
        else
            z =
                f_param(ssm.dyn, zs[k - 1], θ) +
                rand(rng, MvNormal(zeros(2), calc_Q_param(ssm.dyn)))
        end
        zs[k] = z
        if k in obs_indices
            y =
                h_param(ssm.sensor, z, θ) +
                rand(rng, MvNormal(zeros(1), calc_R_param(ssm.sensor)))
            ys[obs_idx] = y
            obs_idx += 1
        end
    end
    return zs, ys
end

zs_true, ys = simulate(rng, ssm, θ_true, K, obs_indices)
zs_true_block = BlockVector{Float64,2}(zs_true)
initial_θ = vcat(from_block_vector(zs_true_block), collect(θ_true))

# ============================================================================
# Manual Leapfrog Integration with Tracking
# ============================================================================

"""
Extract (u, v) trajectory from flat state vector z.
"""
function extract_trajectory(z, K, D)
    us = [z[1 + (k - 1) * D] for k in 1:K]
    vs = [z[2 + (k - 1) * D] for k in 1:K]
    return us, vs
end

"""
Compute kinetic energy gradient manually for debugging.
"""
function compute_kinetic_grad(θ, r, ssm, K, D, P, prior_prec, obs_indices, u)
    G = calc_observed_hessian(θ, ssm, K, D, P, prior_prec, obs_indices)
    F = modified_cholesky(G, u, 0)
    v = F \ r

    H_grad = MCRHMC.hamiltonian_grad(F, G, v)
    dGs = calc_observed_hessian_derivs(θ, ssm, K, D, P, obs_indices)

    kinetic_grad = zeros(length(θ))
    for i in 1:length(θ)
        kinetic_grad[i] = MCRHMC.sparse_trace_symmetric(H_grad, dGs[i])
    end

    return kinetic_grad, F, G
end

"""
Single leapfrog step with full diagnostics.
"""
function leapfrog_step_debug(θ, r, ε, ssm, K, D, P, prior_prec, obs_indices, u, ℓπ)
    n = length(θ)

    # Half step for momentum
    ll, pot_grad = LogDensityProblems.logdensity_and_gradient(ℓπ, θ)
    kin_grad, F, G = compute_kinetic_grad(θ, r, ssm, K, D, P, prior_prec, obs_indices, u)

    r_half = r - (ε/2) * (-pot_grad + kin_grad)

    # Full step for position
    v = F \ r_half
    θ_new = θ + ε * v

    # Half step for momentum
    ll_new, pot_grad_new = LogDensityProblems.logdensity_and_gradient(ℓπ, θ_new)
    kin_grad_new, F_new, G_new = compute_kinetic_grad(
        θ_new, r_half, ssm, K, D, P, prior_prec, obs_indices, u
    )

    r_new = r_half - (ε/2) * (-pot_grad_new + kin_grad_new)

    # Compute energies
    logdet_G = MCRHMC.logdet(F)
    v_start = F \ r
    kinetic_start = 0.5 * logdet_G + 0.5 * dot(r, v_start)
    potential_start = -ll
    H_start = potential_start + kinetic_start

    logdet_G_new = MCRHMC.logdet(F_new)
    v_end = F_new \ r_new
    kinetic_end = 0.5 * logdet_G_new + 0.5 * dot(r_new, v_end)
    potential_end = -ll_new
    H_end = potential_end + kinetic_end

    diagnostics = (
        pot_grad_norm=norm(pot_grad),
        kin_grad_norm=norm(kin_grad),
        max_pot_grad=maximum(abs.(pot_grad)),
        max_kin_grad=maximum(abs.(kin_grad)),
        position_change=norm(θ_new - θ),
        momentum_change=norm(r_new - r),
        H_start=H_start,
        H_end=H_end,
        ΔH=H_end - H_start,
        ll=ll,
        ll_new=ll_new,
        G_min_eig=minimum(eigvals(Symmetric(Matrix(G)))),
        G_new_min_eig=minimum(eigvals(Symmetric(Matrix(G_new)))),
    )

    return θ_new, r_new, diagnostics
end

"""
Run multiple leapfrog steps and collect trajectory history.
"""
function run_leapfrog_trajectory(
    θ_init, r_init, n_steps, ε, ssm, K, D, P, prior_prec, obs_indices, u, ℓπ
)
    θ = copy(θ_init)
    r = copy(r_init)

    # Store trajectory history
    trajectory_history = [(copy(θ), copy(r))]
    diagnostics_history = []

    for step in 1:n_steps
        try
            θ, r, diag = leapfrog_step_debug(
                θ, r, ε, ssm, K, D, P, prior_prec, obs_indices, u, ℓπ
            )
            push!(trajectory_history, (copy(θ), copy(r)))
            push!(diagnostics_history, diag)

            # Check for numerical issues
            if any(isnan.(θ)) || any(isinf.(θ))
                println("NaN/Inf detected at step $step!")
                break
            end
            if abs(diag.ΔH) > 1e10
                println("Huge energy error at step $step: ΔH = $(diag.ΔH)")
                break
            end
        catch e
            println("Error at step $step: $e")
            break
        end
    end

    return trajectory_history, diagnostics_history
end

# ============================================================================
# Run and Visualize
# ============================================================================

struct LogTargetDensityParamSparse{D,P,M,V,PM,PV,OI}
    dim::Int
    ssm::M
    ys::V
    prior_mean::PM
    prior_prec::PV
    obs_indices::OI
end

function LogTargetDensityParamSparse(
    K::Int, D::Int, P::Int, ssm, ys, prior_mean, prior_var, obs_indices
)
    dim = K * D + P
    prior_prec = SVector{P,Float64}(1 ./ prior_var)
    return LogTargetDensityParamSparse{
        D,P,typeof(ssm),typeof(ys),typeof(prior_mean),typeof(prior_prec),typeof(obs_indices)
    }(
        dim, ssm, ys, prior_mean, prior_prec, obs_indices
    )
end

function LogDensityProblems.logdensity(p::LogTargetDensityParamSparse{D,P}, θ) where {D,P}
    θ_blocks = to_bordered_block_vector(θ, Val(D), Val(P))
    return calc_ll_param(
        θ_blocks, p.ys, p.ssm, p.prior_mean, p.prior_prec; obs_indices=p.obs_indices
    )
end

function LogDensityProblems.logdensity_and_gradient(
    p::LogTargetDensityParamSparse{D,P}, θ
) where {D,P}
    θ_blocks = to_bordered_block_vector(θ, Val(D), Val(P))
    ll = calc_ll_param(
        θ_blocks, p.ys, p.ssm, p.prior_mean, p.prior_prec; obs_indices=p.obs_indices
    )
    grad = calc_ll_grad_param(
        θ_blocks, p.ys, p.ssm, p.prior_mean, p.prior_prec; obs_indices=p.obs_indices
    )
    return ll, from_bordered_block_vector(grad)
end

LogDensityProblems.dimension(p::LogTargetDensityParamSparse) = p.dim
function LogDensityProblems.capabilities(::Type{<:LogTargetDensityParamSparse})
    LogDensityProblems.LogDensityOrder{1}()
end

ℓπ = LogTargetDensityParamSparse(K, D, P, ssm, ys, prior_mean, prior_var, obs_indices)

# Settings
u = fill(10.0, n)
ε = 0.1  # Step size
n_steps = 20  # Number of leapfrog steps

println("="^70)
println("LEAPFROG TRAJECTORY VISUALIZATION")
println("="^70)
println("\nSettings: u=$(u[1]), ε=$ε, n_steps=$n_steps")

# Initialize with random momentum
Random.seed!(123)
r_init = randn(n)

println("\nRunning leapfrog integration...")
traj_hist, diag_hist = run_leapfrog_trajectory(
    initial_θ, r_init, n_steps, ε, ssm, K, D, P, prior_prec, obs_indices, u, ℓπ
)

println("\n--- Step-by-step diagnostics ---")
println("Step | ΔH           | max|kin_grad| | max|pot_grad| | G_min_eig    | |Δθ|")
println("-"^85)

for (i, diag) in enumerate(diag_hist)
    @printf(
        "%4d | %12.4e | %13.4e | %13.4e | %12.4e | %10.4e\n",
        i,
        diag.ΔH,
        diag.max_kin_grad,
        diag.max_pot_grad,
        diag.G_min_eig,
        diag.position_change
    )
end

# ============================================================================
# Plotting
# ============================================================================

println("\nGenerating trajectory plots...")

# Create color gradient for steps
n_traj = length(traj_hist)
colors = cgrad(:viridis, n_traj)

# Plot 1: Full phase space trajectory (u vs v for first few time points)
p1 = plot(;
    title="Leapfrog: States 1-5 (u vs v)", xlabel="u", ylabel="v", legend=:outerright
)

# Plot true trajectory
us_true, vs_true = extract_trajectory(initial_θ, K, D)
plot!(
    p1, us_true[1:5], vs_true[1:5]; lw=3, color=:black, label="True", marker=:circle, ms=4
)

# Plot leapfrog evolution
for (step, (θ, r)) in enumerate(traj_hist)
    us, vs = extract_trajectory(θ, K, D)
    plot!(
        p1,
        us[1:5],
        vs[1:5];
        lw=1,
        color=colors[step],
        alpha=0.7,
        label=(step == 1 ? "Step 0" : (step == n_traj ? "Step $(n_traj-1)" : "")),
    )
end

# Plot 2: State 1 trajectory over leapfrog steps
p2 = plot(;
    title="State 1: u₁ and v₁ over leapfrog steps", xlabel="Leapfrog step", legend=:topright
)

u1_traj = [traj_hist[i][1][1] for i in 1:n_traj]
v1_traj = [traj_hist[i][1][2] for i in 1:n_traj]
u1_true = initial_θ[1]
v1_true = initial_θ[2]

plot!(p2, 0:(n_traj - 1), u1_traj; lw=2, label="u₁", color=:blue)
plot!(p2, 0:(n_traj - 1), v1_traj; lw=2, label="v₁", color=:red)
hline!(p2, [u1_true]; ls=:dash, color=:blue, alpha=0.5, label="u₁ true")
hline!(p2, [v1_true]; ls=:dash, color=:red, alpha=0.5, label="v₁ true")

# Plot 2b: Parameter (log μ) over leapfrog steps
p2b = plot(;
    title="Parameter: log(μ) over leapfrog steps", xlabel="Leapfrog step", legend=:topright
)

param_idx = K * D + 1  # Index of parameter in state vector
logmu_traj = [traj_hist[i][1][param_idx] for i in 1:n_traj]
logmu_true = initial_θ[param_idx]
mu_traj = exp.(logmu_traj)
mu_true_val = exp(logmu_true)

plot!(p2b, 0:(n_traj - 1), logmu_traj; lw=2, label="log(μ)", color=:orange)
hline!(p2b, [logmu_true]; ls=:dash, color=:orange, alpha=0.5, label="log(μ) true")

# Also show μ on secondary y-axis (as text annotation)
annotate!(
    p2b, n_traj-1, logmu_traj[end], text("μ=$(round(mu_traj[end], digits=3))", 8, :left)
)

# Plot 3: Hamiltonian energy over steps
p3 = plot(; title="Hamiltonian Energy", xlabel="Leapfrog step", ylabel="H")

H_vals = [diag_hist[1].H_start]
append!(H_vals, [d.H_end for d in diag_hist])
plot!(p3, 0:(length(H_vals) - 1), H_vals; lw=2, marker=:circle, ms=3, label="H")

# Plot 4: Energy error ΔH
p4 = plot(; title="Energy Error ΔH per step", xlabel="Leapfrog step", ylabel="ΔH")
ΔH_vals = [d.ΔH for d in diag_hist]
plot!(p4, 1:length(ΔH_vals), ΔH_vals; lw=2, marker=:circle, ms=3, label="ΔH", color=:red)
hline!(p4, [0]; ls=:dash, color=:black, label="")

# Plot 5: Gradient magnitudes
p5 = plot(;
    title="Gradient Magnitudes", xlabel="Leapfrog step", ylabel="max |grad|", yscale=:log10
)
kin_grads = [d.max_kin_grad for d in diag_hist]
pot_grads = [d.max_pot_grad for d in diag_hist]
plot!(p5, 1:length(kin_grads), kin_grads; lw=2, label="Kinetic", color=:blue)
plot!(p5, 1:length(pot_grads), pot_grads; lw=2, label="Potential", color=:green)

# Plot 6: Hessian minimum eigenvalue
p6 = plot(; title="Observed Hessian min eigenvalue", xlabel="Leapfrog step", ylabel="λ_min")
min_eigs = [d.G_min_eig for d in diag_hist]
plot!(
    p6,
    1:length(min_eigs),
    min_eigs;
    lw=2,
    marker=:circle,
    ms=3,
    label="λ_min",
    color=:purple,
)
hline!(p6, [0]; ls=:dash, color=:red, label="Zero")

# Combine plots (now with parameter plot)
p_combined = plot(p1, p2, p2b, p3, p4, p5, p6; layout=(4, 2), size=(1200, 1200))
display(p_combined)

# Also plot full trajectory evolution in phase space
println("\nPlotting full trajectory evolution...")

p_full = plot(;
    title="Full Trajectory Evolution (colored by step)",
    xlabel="u",
    ylabel="v",
    legend=false,
    size=(800, 600),
)

# True trajectory
us_true, vs_true = extract_trajectory(initial_θ, K, D)
plot!(p_full, us_true, vs_true; lw=2, color=:black, alpha=0.3)

# Evolution
for (step, (θ, r)) in enumerate(traj_hist)
    us, vs = extract_trajectory(θ, K, D)
    plot!(p_full, us, vs; lw=0.5, color=colors[step], alpha=0.8)
end

# Mark observations
scatter!(p_full, only.(ys), vs_true[obs_indices]; color=:red, ms=5, marker=:star)

display(p_full)

println("\nDone!")
