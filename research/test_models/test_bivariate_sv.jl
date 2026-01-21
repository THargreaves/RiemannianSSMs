"""
Test script for RHMC with Bivariate Stochastic Volatility model.

This benchmark uses:
- Nonlinear AR dynamics with quadratic damping: h_{t+1} = ϕ·h_t - κ·h_t² + σ·η_t
- Zero-mean Gaussian observations with stochastic variance: y_t ~ N(0, diag(exp(h_t)))

This model is designed to challenge HMC:
- Near-unit-root persistence (ϕ ≈ 0.98) creates long-range correlations
- Small process noise (σ = 0.05) creates narrow posterior tubes
- Nonlinear dynamics create position-dependent curvature
- State-dependent observation variance couples latent states and data

Parameters to infer:
- θ₁ = atanh(ϕ) where ϕ ∈ (-1, 1) is persistence
- θ₂ = log(κ) where κ > 0 is nonlinear damping coefficient
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
K_y = 5            # Latent steps between observations
N_obs = 40         # Number of observations
K = N_obs * K_y    # Total number of latent states (200)

# Observation indices (1-indexed, every K_y-th state starting from K_y)
obs_indices = collect(K_y:K_y:K)

# ============================================================================
# Model Setup - Parameters chosen to make HMC struggle
# ============================================================================

# True parameter values (challenging regime)
ϕ_true = 0.97      # Near unit root - creates long-range correlations
κ_true = 0.5       # Cubic damping coefficient (keeps h in reasonable range)

# Transform to unconstrained parameterization
θ_true = @SVector [atanh(ϕ_true), log(κ_true)]

# Process noise - small for near-deterministic dynamics
σ = 0.05

# Create model
model = BivariateSVModel(;
    σ=σ, h0=SVector{2,Float64}(0.0, 0.0), Σ0_diag=SVector{2,Float64}(1.0, 1.0)
)

# Dimensions
Dx = state_dim(model)
Dy = obs_dim(model)
Dp = param_dim(model)

println("Model: BivariateSVModel")
println("State dim: $Dx, Obs dim: $Dy, Param dim: $Dp")
println("\nTrue parameters:")
println("  ϕ = $ϕ_true (persistence, near unit root)")
println("  κ = $κ_true (nonlinear damping)")
println("  σ = $σ (process noise, fixed)")

# ============================================================================
# Ground truth simulation
# ============================================================================

println("\nSimulating bivariate stochastic volatility...")
println("Total latent states: K = $K")
println("Number of observations: $(length(obs_indices))")
println("Observations at indices: $(obs_indices[1]):$(K_y):$(obs_indices[end])")

hs_true, ys = simulate(rng, model, θ_true, K; obs_indices=obs_indices)
hs_true_block = BlockVector{Float64,2}(hs_true)

# Plot log-volatility trajectories
p1 = plot(;
    title="Log-Volatility Trajectories (h_t)",
    xlabel="Time step",
    ylabel="Log-volatility",
    legend=:topright,
    size=(900, 400),
)
plot!(p1, 1:K, [h[1] for h in hs_true]; label="h₁", lw=1.5, color=:blue)
plot!(p1, 1:K, [h[2] for h in hs_true]; label="h₂", lw=1.5, color=:red)
vline!(p1, obs_indices; label="", color=:gray, alpha=0.2, linestyle=:dash)

display(p1)

# Plot volatility (exp(h/2) = σ)
p2 = plot(;
    title="Volatility Trajectories (σ_t = exp(h_t/2))",
    xlabel="Time step",
    ylabel="Volatility",
    legend=:topright,
    size=(900, 400),
)
plot!(p2, 1:K, [exp(h[1] / 2) for h in hs_true]; label="σ₁", lw=1.5, color=:blue)
plot!(p2, 1:K, [exp(h[2] / 2) for h in hs_true]; label="σ₂", lw=1.5, color=:red)
vline!(p2, obs_indices; label="", color=:gray, alpha=0.2, linestyle=:dash)

display(p2)

# Plot observations
p3 = plot(;
    title="Observations (y_t ~ N(0, exp(h_t)))",
    xlabel="Observation index",
    ylabel="y",
    legend=:topright,
    size=(900, 400),
)
scatter!(p3, 1:N_obs, [y[1] for y in ys]; label="y₁", color=:blue, ms=4, alpha=0.7)
scatter!(p3, 1:N_obs, [y[2] for y in ys]; label="y₂", color=:red, ms=4, alpha=0.7)

display(p3)

# Observation statistics
println("\nObservation statistics:")
println(
    "  y₁: mean=$(round(mean([y[1] for y in ys]), digits=3)), std=$(round(std([y[1] for y in ys]), digits=3))",
)
println(
    "  y₂: mean=$(round(mean([y[2] for y in ys]), digits=3)), std=$(round(std([y[2] for y in ys]), digits=3))",
)

# Check for bursts (large |y|)
y1_abs = abs.([y[1] for y in ys])
y2_abs = abs.([y[2] for y in ys])
println("  Max |y₁|: $(round(maximum(y1_abs), digits=3))")
println("  Max |y₂|: $(round(maximum(y2_abs), digits=3))")

# ============================================================================
# LogDensityProblems Model (Joint state + parameter inference)
# ============================================================================

# Parameter prior (weakly informative, centered near true values)
prior_mean = @SVector [atanh(0.9), log(0.2)]  # Prior centered away from truth
prior_var = @SVector [1.0, 1.0]               # Moderate variance

ℓπ = RHMCLogDensity(
    model, ys, K; prior_mean=prior_mean, prior_var=prior_var, obs_indices=obs_indices
)
adv_model = AdvancedHMC.LogDensityModel(ℓπ)

# Initial state: true states + true parameters (warm start)
initial_θ = vcat(from_block_vector(hs_true_block), collect(θ_true))

# ============================================================================
# RHMC Sampling (Joint state + parameter)
# ============================================================================

println("\nSetting up RHMC sampler...")

metric = RiemannianMetric(model, ys, K, prior_mean, prior_var; obs_indices=obs_indices)
hamiltonian = Hamiltonian(metric, ℓπ)
initial_ϵ = 0.01
integrator = AdaptiveGeneralizedLeapfrog(initial_ϵ; max_iters=7)
kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
adaptor = StepSizeAdaptor(0.9, integrator)
rhmc = HMCSampler(kernel, metric, adaptor)

N_samples = 3000
N_adapt = 1000

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
ϕ_samples = [tanh(s[K * Dx + 1]) for s in samples]
κ_samples = [exp(s[K * Dx + 2]) for s in samples]

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
println("atanh(ϕ) ESS: ", param_ess[1])
println("log(κ) ESS: ", param_ess[2])

println("\n=== RHMC Parameter Posterior Summary ===")
burn_in = N_adapt
ϕ_post = ϕ_samples[(burn_in + 1):end]
κ_post = κ_samples[(burn_in + 1):end]
println(
    "ϕ: true=$ϕ_true, posterior mean=$(round(mean(ϕ_post), digits=4)), std=$(round(std(ϕ_post), digits=4))",
)
println(
    "κ: true=$κ_true, posterior mean=$(round(mean(κ_post), digits=4)), std=$(round(std(κ_post), digits=4))",
)

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
ϕ_samples_hmc = [tanh(s[K * Dx + 1]) for s in hmc_samples_raw]
κ_samples_hmc = [exp(s[K * Dx + 2]) for s in hmc_samples_raw]

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
println("atanh(ϕ) ESS: ", hmc_param_ess[1])
println("log(κ) ESS: ", hmc_param_ess[2])

println("\n=== HMC Parameter Posterior Summary ===")
ϕ_post_hmc = ϕ_samples_hmc[(burn_in + 1):end]
κ_post_hmc = κ_samples_hmc[(burn_in + 1):end]
println(
    "ϕ: true=$ϕ_true, posterior mean=$(round(mean(ϕ_post_hmc), digits=4)), std=$(round(std(ϕ_post_hmc), digits=4))",
)
println(
    "κ: true=$κ_true, posterior mean=$(round(mean(κ_post_hmc), digits=4)), std=$(round(std(κ_post_hmc), digits=4))",
)

# ============================================================================
# Comparison Summary
# ============================================================================

println("\n" * "="^60)
println("COMPARISON SUMMARY: RHMC vs HMC for Bivariate SV")
println("="^60)

println("\n--- State ESS ---")
println(
    "RHMC: min=$(round(minimum(state_ess), digits=4)), median=$(round(median(state_ess), digits=4)), mean=$(round(mean(state_ess), digits=4))",
)
println(
    "HMC:  min=$(round(minimum(hmc_state_ess), digits=4)), median=$(round(median(hmc_state_ess), digits=4)), mean=$(round(mean(hmc_state_ess), digits=4))",
)

println("\n--- Parameter ESS ---")
println(
    "RHMC: atanh(ϕ)=$(round(param_ess[1], digits=4)), log(κ)=$(round(param_ess[2], digits=4))",
)
println(
    "HMC:  atanh(ϕ)=$(round(hmc_param_ess[1], digits=4)), log(κ)=$(round(hmc_param_ess[2], digits=4))",
)

# ============================================================================
# Plots
# ============================================================================

# Parameter trace plots
p_trace = plot(; layout=(2, 1), size=(800, 600), legend=:topright)

plot!(
    p_trace[1],
    ϕ_samples;
    subplot=1,
    label="RHMC",
    lw=1,
    color=:blue,
    alpha=0.5,
    title="ϕ trace",
)
plot!(p_trace[1], ϕ_samples_hmc; subplot=1, label="HMC", lw=1, color=:green, alpha=0.5)
hline!(p_trace[1], [ϕ_true]; subplot=1, label="True", lw=2, color=:red)

plot!(
    p_trace[2],
    κ_samples;
    subplot=2,
    label="RHMC",
    lw=1,
    color=:blue,
    alpha=0.5,
    title="κ trace",
    xlabel="Sample",
)
plot!(p_trace[2], κ_samples_hmc; subplot=2, label="HMC", lw=1, color=:green, alpha=0.5)
hline!(p_trace[2], [κ_true]; subplot=2, label="True", lw=2, color=:red)

display(p_trace)

# Parameter posterior histograms
p_hist = plot(; layout=(1, 2), size=(800, 400))

histogram!(
    p_hist[1], ϕ_post; subplot=1, label="RHMC", alpha=0.5, color=:blue, title="ϕ posterior"
)
histogram!(p_hist[1], ϕ_post_hmc; subplot=1, label="HMC", alpha=0.5, color=:green)
vline!(p_hist[1], [ϕ_true]; subplot=1, label="True", lw=2, color=:red)

histogram!(
    p_hist[2], κ_post; subplot=2, label="RHMC", alpha=0.5, color=:blue, title="κ posterior"
)
histogram!(p_hist[2], κ_post_hmc; subplot=2, label="HMC", alpha=0.5, color=:green)
vline!(p_hist[2], [κ_true]; subplot=2, label="True", lw=2, color=:red)

display(p_hist)

# State trajectory posterior (thin samples for visualization)
n_plot_samples = 100
plot_idxs = round.(Int, LinRange(burn_in + 1, N_samples, n_plot_samples))

p_traj = plot(; layout=(2, 1), size=(900, 600), legend=:topright)

# RHMC trajectories for h_1
for i in 1:n_plot_samples
    s = samples[plot_idxs[i]]
    plot!(
        p_traj[1],
        1:K,
        [s[1 + (k - 1) * Dx] for k in 1:K];
        subplot=1,
        label="",
        lw=0.5,
        alpha=0.1,
        color=:blue,
    )
end
plot!(
    p_traj[1],
    1:K,
    [h[1] for h in hs_true];
    subplot=1,
    label="Truth",
    lw=2,
    color=:black,
    title="h₁ posterior (RHMC)",
)

# RHMC trajectories for h_2
for i in 1:n_plot_samples
    s = samples[plot_idxs[i]]
    plot!(
        p_traj[2],
        1:K,
        [s[2 + (k - 1) * Dx] for k in 1:K];
        subplot=2,
        label="",
        lw=0.5,
        alpha=0.1,
        color=:blue,
    )
end
plot!(
    p_traj[2],
    1:K,
    [h[2] for h in hs_true];
    subplot=2,
    label="Truth",
    lw=2,
    color=:black,
    title="h₂ posterior (RHMC)",
    xlabel="Time step",
)

display(p_traj)

println("\nDone!")
