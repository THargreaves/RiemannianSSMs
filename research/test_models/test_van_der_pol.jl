"""
Test script for the unified RHMC interface with Van der Pol oscillator.

This replicates test_vdp_benchmark_param.jl using the new unified interface.
"""

using Distributions
using LinearAlgebra
using Random
using Plots
using StaticArrays

using AbstractMCMC
using AdvancedHMC
using MCMCDiagnosticTools

using RiemannianSSMs

# ============================================================================
# Configuration
# ============================================================================

SEED = 42
rng = MersenneTwister(SEED)

# Time discretization
δt = 0.1           # Euler step size
K_y = 10           # Latent steps between observations
N_obs = 20         # Number of observations
K = N_obs * K_y    # Total number of latent states (200)

# Observation indices (1-indexed, every K_y-th state starting from K_y)
obs_indices = collect(K_y:K_y:K)

# ============================================================================
# Model Setup
# ============================================================================

# True parameter value
μ_true = 1.0
θ_true = @SVector [log(μ_true)]

# Create model using new interface
model = VanDerPolModel(;
    δt=δt,
    σ_u=0.01,
    σ_v=0.1,
    σ_obs=0.5,
    μ0=SVector{2,Float64}(1.0, 0.0),
    Σ0_diag=SVector{2,Float64}(0.1, 0.1),
)

# Dimensions
Dx = state_dim(model)
Dy = obs_dim(model)
Dp = param_dim(model)

println("Model: VanDerPolModel")
println("State dim: $Dx, Obs dim: $Dy, Param dim: $Dp")

# ============================================================================
# Ground truth simulation
# ============================================================================

println("\nSimulating Van der Pol oscillator with μ = $(μ_true)...")
println("Total latent states: K = $K")
println("Number of observations: $(length(obs_indices))")
println("Observations at indices: $(obs_indices[1]):$(K_y):$(obs_indices[end])")

zs_true, ys = simulate(rng, model, θ_true, K; obs_indices=obs_indices)
zs_true_block = BlockVector{Float64,2}(zs_true)

# Plot the trajectory
p1 = plot(;
    title="Van der Pol Oscillator (μ=$(μ_true))",
    xlabel="u",
    ylabel="v",
    legend=:topright,
    size=(800, 600),
    aspect_ratio=1,
)
plot!(
    p1, [z[1] for z in zs_true], [z[2] for z in zs_true]; label="Truth", lw=2, color=:black
)

# Plot observed u coordinate with true v
scatter!(
    p1,
    only.(ys),
    getindex.(zs_true[obs_indices], 2);
    label="Observations",
    color=:blue,
    ms=4,
)

# Add dashed line between true and observed points
for (k, y) in zip(obs_indices, ys)
    plot!(
        p1,
        [zs_true[k][1], y[1]],
        [zs_true[k][2], zs_true[k][2]];
        l=:dot,
        color=:black,
        alpha=0.8,
        label="",
    )
end

display(p1)

# ============================================================================
# LogDensityProblems Model (Joint state + parameter inference)
# ============================================================================

# Parameter prior (diffuse Gaussian on log scale)
prior_mean = @SVector [log(μ_true)]  # Centered near true value
prior_var = @SVector [4.0]           # Wide variance

ℓπ = RHMCLogDensity(
    model, ys, K; prior_mean=prior_mean, prior_var=prior_var, obs_indices=obs_indices
)
adv_model = AdvancedHMC.LogDensityModel(ℓπ)

# Initial state: true states + true parameters
initial_θ = vcat(from_block_vector(zs_true_block), collect(θ_true))

# ============================================================================
# RHMC Sampling (Joint state + parameter)
# ============================================================================

println("\nSetting up RHMC sampler with unified interface...")
metric = RiemannianMetric(model, ys, K, prior_mean, prior_var; obs_indices=obs_indices)
hamiltonian = Hamiltonian(metric, ℓπ)
initial_ϵ = 0.005
integrator = AdaptiveGeneralizedLeapfrog(initial_ϵ; max_iters=7)
kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
adaptor = StepSizeAdaptor(0.9, integrator)
rhmc = HMCSampler(kernel, metric, adaptor)

N_samples = 5000
N_adapt = 2000

println("Running RHMC with joint state+parameter inference...")
chains = AbstractMCMC.sample(
    adv_model,
    rhmc,
    N_samples;
    n_adapts=N_adapt,
    initial_params=initial_θ,
    verbose=false,
    progress=true,
);
samples = [s.z.θ for s in chains];

# Extract parameter samples (last Dp elements)
μ_samples = [exp(s[K * Dx + 1]) for s in samples]

# Compute ESS for all dimensions
total_dim = Dx * K + Dp
rhmc_samples = Array{Float64}(undef, N_samples, 1, total_dim)
for i in 1:N_samples
    for j in 1:total_dim
        rhmc_samples[i, 1, j] = samples[i][j]
    end
end
rhmc_ess = ess(rhmc_samples) ./ N_samples

println("\n=== RHMC State ESS Statistics ===")
state_ess = rhmc_ess[1:(Dx * K)]
println("Minimum State ESS: ", minimum(state_ess))
println("Median State ESS: ", median(state_ess))
println("Mean State ESS: ", mean(state_ess))

println("\n=== RHMC Parameter ESS ===")
param_ess = rhmc_ess[(Dx * K + 1):end]
println("log(μ) ESS: ", param_ess[1])

println("\n=== RHMC Parameter Posterior Summary ===")
burn_in = N_adapt
μ_post = μ_samples[(burn_in + 1):end]
println("μ: true=$(μ_true), posterior mean=$(mean(μ_post)), std=$(std(μ_post))")

# ============================================================================
# Plots
# ============================================================================

# Trace plot for parameter
p_μ = plot(; title="μ trace (RHMC Unified)", xlabel="Sample", ylabel="μ", legend=:topright)
plot!(p_μ, μ_samples; label="Samples", lw=1, color=:blue, alpha=0.5)
hline!(p_μ, [μ_true]; label="True", lw=2, color=:red)

display(p_μ)

# Histogram of posterior parameter
p_μ_hist = histogram(
    μ_post;
    title="μ posterior (RHMC Unified)",
    xlabel="μ",
    ylabel="Count",
    label="",
    bins=50,
    color=:blue,
    alpha=0.7,
)
vline!(p_μ_hist, [μ_true]; label="True", lw=2, color=:red)

display(p_μ_hist)

# Thin plot samples for trajectory
n_plot_samples = 200
plot_idxs = round.(Int, LinRange(burn_in + 1, N_samples, n_plot_samples))
plot_samples = samples[plot_idxs]

p2 = plot(;
    title="Posterior Trajectories (RHMC Unified)",
    xlabel="u",
    ylabel="v",
    legend=:topright,
    size=(800, 600),
)
for i in 1:n_plot_samples
    s = plot_samples[i]
    plot!(
        p2,
        [s[1 + (k - 1) * Dx] for k in 1:K],
        [s[2 + (k - 1) * Dx] for k in 1:K];
        label="",
        lw=1,
        alpha=0.05,
        color=:blue,
    )
end
plot!(
    p2, [z[1] for z in zs_true], [z[2] for z in zs_true]; label="Truth", lw=2, color=:black
)

display(p2)

# ============================================================================
# HMC Sampling (for comparison)
# ============================================================================

println("\nRunning HMC with joint state+parameter inference...")
hmc = NUTS(0.8)
hmc_chains = AbstractMCMC.sample(
    adv_model,
    hmc,
    N_samples;
    n_adapts=N_adapt,
    initial_params=initial_θ,
    verbose=false,
    progress=true,
);
hmc_samples_raw = [s.z.θ for s in hmc_chains];

# Extract parameter samples
μ_samples_hmc = [exp(s[K * Dx + 1]) for s in hmc_samples_raw]

# Compute ESS
hmc_samples = Array{Float64}(undef, N_samples, 1, total_dim)
for i in 1:N_samples
    for j in 1:total_dim
        hmc_samples[i, 1, j] = hmc_samples_raw[i][j]
    end
end
hmc_ess = ess(hmc_samples) ./ N_samples

println("\n=== HMC State ESS Statistics ===")
hmc_state_ess = hmc_ess[1:(Dx * K)]
println("Minimum State ESS: ", minimum(hmc_state_ess))
println("Median State ESS: ", median(hmc_state_ess))
println("Mean State ESS: ", mean(hmc_state_ess))

println("\n=== HMC Parameter ESS ===")
hmc_param_ess = hmc_ess[(Dx * K + 1):end]
println("log(μ) ESS: ", hmc_param_ess[1])

println("\n=== HMC Parameter Posterior Summary ===")
μ_post_hmc = μ_samples_hmc[(burn_in + 1):end]
println("μ: true=$(μ_true), posterior mean=$(mean(μ_post_hmc)), std=$(std(μ_post_hmc))")

# ============================================================================
# Comparison Summary
# ============================================================================

println("\n" * "="^60)
println("COMPARISON SUMMARY")
println("="^60)

println("\n--- State ESS ---")
println(
    "RHMC: min=$(round(minimum(state_ess), digits=4)), median=$(round(median(state_ess), digits=4)), mean=$(round(mean(state_ess), digits=4))",
)
println(
    "HMC:  min=$(round(minimum(hmc_state_ess), digits=4)), median=$(round(median(hmc_state_ess), digits=4)), mean=$(round(mean(hmc_state_ess), digits=4))",
)

println("\n--- Parameter ESS ---")
println("RHMC: log(μ)=$(round(param_ess[1], digits=4))")
println("HMC:  log(μ)=$(round(hmc_param_ess[1], digits=4))")

# Comparison trace plots for parameter
p_μ_compare = plot(;
    title="μ trace comparison", xlabel="Sample", ylabel="μ", legend=:topright
)
plot!(p_μ_compare, μ_samples; label="RHMC", lw=1, color=:blue, alpha=0.5)
plot!(p_μ_compare, μ_samples_hmc; label="HMC", lw=1, color=:green, alpha=0.5)
hline!(p_μ_compare, [μ_true]; label="True", lw=2, color=:red)

display(p_μ_compare)

println("\nDone!")
