"""
Test script for RHMC with spatiotemporal Bernoulli observations.

This benchmark uses:
- Nonlinear diffusion dynamics with kernel Laplacian coupling
- Bernoulli observations with logistic link at each node

Configuration:
- d = 6 nodes with positions sampled from N(0, I)
- K = 100 total time steps
- n_obs = 10 observations (every 10th step)
- Parameters: κ (coupling), g (reaction), a (sensitivity), b (intercept)
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

# Spatial configuration
d = 5              # Number of nodes
p = 2              # Spatial dimension (2D positions)

# Time discretization
Δt = 0.1           # Time step
K_y = 10           # Latent steps between observations
N_obs = 10         # Number of observations
K = N_obs * K_y    # Total number of latent states (100)

# Observation indices (1-indexed, every K_y-th state starting from K_y)
obs_indices = collect(K_y:K_y:K)

# Process noise (fixed)
σ = 0.05

# Initial state prior scale
s0 = 1.0

# ============================================================================
# Generate Node Positions
# ============================================================================

println("Generating $d node positions in $p dimensions...")
positions = [SVector{p,Float64}(randn(rng, p)) for _ in 1:d]

# Compute kernel lengthscale as 2× median nearest-neighbor distance
r_nn = median_nearest_neighbor_distance(positions)
ℓ = 2.0 * r_nn
println("Median nearest-neighbor distance: $(round(r_nn, digits=4))")
println("Kernel lengthscale ℓ = $(round(ℓ, digits=4))")

# ============================================================================
# Model Setup
# ============================================================================

# True parameter values (reasonable for the stabilized dynamics)
κ_true = 2.0       # Coupling strength
g_true = 1.5       # Reaction strength (stabilizing toward zero)
a_true = 10.0       # Measurement sensitivity
b_true = 0.0       # Baseline intercept (p = 0.5 when x = 0)

# θ = [log(κ), log(g), log(a), b]
θ_true = @SVector [log(κ_true), log(g_true), log(a_true), b_true]

# Create model
model = SpatiotemporalBernoulliModel(positions, ℓ; Δt=Δt, σ=σ, s0=s0)

# Dimensions
Dx = state_dim(model)
Dy = obs_dim(model)
Dp = param_dim(model)

println("\nModel: SpatiotemporalBernoulliModel")
println("State dim: $Dx, Obs dim: $Dy, Param dim: $Dp")

# ============================================================================
# Ground truth simulation
# ============================================================================

println("\nSimulating spatiotemporal dynamics...")
println("True parameters: κ=$(κ_true), g=$(g_true), a=$(a_true), b=$(b_true)")
println("Total latent states: K = $K")
println("Number of observations: $(length(obs_indices))")
println("Observations at indices: $(obs_indices[1]):$(K_y):$(obs_indices[end])")

zs_true, ys = simulate(rng, model, θ_true, K; obs_indices=obs_indices)
zs_true_block = BlockVector{Float64,Dx}(zs_true)

# Plot state trajectories
p1 = plot(;
    title="Latent State Trajectories",
    xlabel="Time step",
    ylabel="State value",
    legend=:outerright,
    size=(900, 500),
)
for i in 1:Dx
    plot!(p1, 1:K, [z[i] for z in zs_true]; label="x_$i", lw=1.5, alpha=0.8)
end
# Mark observation times
vline!(p1, obs_indices; label="", color=:gray, alpha=0.3, linestyle=:dash)

display(p1)

# Plot observation probabilities at all times and actual observations at obs times
η_all = [η(model, zs_true[k], θ_true) for k in 1:K]
p_all = [[1 / (1 + exp(-η_k[i])) for i in 1:Dx] for η_k in η_all]

p2 = plot(;
    title="Bernoulli Probabilities and Observations",
    xlabel="Time step",
    ylabel="Probability / Observation",
    legend=:outerright,
    size=(900, 500),
)
for i in 1:Dx
    probs = [p_k[i] for p_k in p_all]
    obs_i = [y[i] for y in ys]
    plot!(p2, 1:K, probs; label="p_$i", lw=1.5, alpha=0.7)
    scatter!(
        p2, obs_indices, obs_i; label="", color=i, ms=5, alpha=0.8, markerstrokewidth=0
    )
end
vline!(p2, obs_indices; label="", color=:gray, alpha=0.2, linestyle=:dash)

display(p2)

println("\nObservation statistics:")
total_obs = sum(sum(y) for y in ys)
println(
    "  Total 1s: $total_obs / $(N_obs * Dx) = $(round(total_obs / (N_obs * Dx), digits=3))"
)

# ============================================================================
# LogDensityProblems Model (Joint state + parameter inference)
# ============================================================================

# Parameter prior (diffuse Gaussian on log/real scale)
prior_mean = @SVector [log(κ_true), log(g_true), log(a_true), b_true]  # Centered near true
prior_var = @SVector [2.0, 2.0, 2.0, 4.0]  # Moderate variance

ℓπ = RHMCLogDensity(
    model, ys, K; prior_mean=prior_mean, prior_var=prior_var, obs_indices=obs_indices
)
adv_model = AdvancedHMC.LogDensityModel(ℓπ)

# Initial state: true states + true parameters
initial_θ = vcat(from_block_vector(zs_true_block), collect(θ_true))

# ============================================================================
# RHMC Sampling (Joint state + parameter with Bernoulli)
# ============================================================================

println("\nSetting up RHMC sampler...")

metric = RiemannianMetric(model, ys, K, prior_mean, prior_var; obs_indices=obs_indices)
hamiltonian = Hamiltonian(metric, ℓπ)
initial_ϵ = 0.01
integrator = AdaptiveGeneralizedLeapfrog(initial_ϵ; max_iters=7)
kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
adaptor = StepSizeAdaptor(0.9, integrator)
rhmc = HMCSampler(kernel, metric, adaptor)

N_samples = 1000
N_adapt = 500

println("Running RHMC with joint state+parameter inference (Bernoulli)...")
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
κ_samples = [exp(s[K * Dx + 1]) for s in samples]
g_samples = [exp(s[K * Dx + 2]) for s in samples]
a_samples = [exp(s[K * Dx + 3]) for s in samples]
b_samples = [s[K * Dx + 4] for s in samples]

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
println("log(κ) ESS: ", param_ess[1])
println("log(g) ESS: ", param_ess[2])
println("log(a) ESS: ", param_ess[3])
println("b ESS: ", param_ess[4])

println("\n=== RHMC Parameter Posterior Summary ===")
burn_in = N_adapt
κ_post = κ_samples[(burn_in + 1):end]
g_post = g_samples[(burn_in + 1):end]
a_post = a_samples[(burn_in + 1):end]
b_post = b_samples[(burn_in + 1):end]
println(
    "κ: true=$(κ_true), posterior mean=$(round(mean(κ_post), digits=4)), std=$(round(std(κ_post), digits=4))",
)
println(
    "g: true=$(g_true), posterior mean=$(round(mean(g_post), digits=4)), std=$(round(std(g_post), digits=4))",
)
println(
    "a: true=$(a_true), posterior mean=$(round(mean(a_post), digits=4)), std=$(round(std(a_post), digits=4))",
)
println(
    "b: true=$(b_true), posterior mean=$(round(mean(b_post), digits=4)), std=$(round(std(b_post), digits=4))",
)

# ============================================================================
# HMC Sampling (for comparison)
# ============================================================================

println("\nRunning HMC with joint state+parameter inference (Bernoulli)...")
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
κ_samples_hmc = [exp(s[K * Dx + 1]) for s in hmc_samples_raw]
g_samples_hmc = [exp(s[K * Dx + 2]) for s in hmc_samples_raw]
a_samples_hmc = [exp(s[K * Dx + 3]) for s in hmc_samples_raw]
b_samples_hmc = [s[K * Dx + 4] for s in hmc_samples_raw]

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
println("log(κ) ESS: ", hmc_param_ess[1])
println("log(g) ESS: ", hmc_param_ess[2])
println("log(a) ESS: ", hmc_param_ess[3])
println("b ESS: ", hmc_param_ess[4])

println("\n=== HMC Parameter Posterior Summary ===")
κ_post_hmc = κ_samples_hmc[(burn_in + 1):end]
g_post_hmc = g_samples_hmc[(burn_in + 1):end]
a_post_hmc = a_samples_hmc[(burn_in + 1):end]
b_post_hmc = b_samples_hmc[(burn_in + 1):end]
println(
    "κ: true=$(κ_true), posterior mean=$(round(mean(κ_post_hmc), digits=4)), std=$(round(std(κ_post_hmc), digits=4))",
)
println(
    "g: true=$(g_true), posterior mean=$(round(mean(g_post_hmc), digits=4)), std=$(round(std(g_post_hmc), digits=4))",
)
println(
    "a: true=$(a_true), posterior mean=$(round(mean(a_post_hmc), digits=4)), std=$(round(std(a_post_hmc), digits=4))",
)
println(
    "b: true=$(b_true), posterior mean=$(round(mean(b_post_hmc), digits=4)), std=$(round(std(b_post_hmc), digits=4))",
)

# ============================================================================
# Comparison Summary
# ============================================================================

println("\n" * "="^60)
println("COMPARISON SUMMARY: RHMC vs HMC for Spatiotemporal Bernoulli SSM")
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
    "RHMC: κ=$(round(param_ess[1], digits=4)), g=$(round(param_ess[2], digits=4)), a=$(round(param_ess[3], digits=4)), b=$(round(param_ess[4], digits=4))",
)
println(
    "HMC:  κ=$(round(hmc_param_ess[1], digits=4)), g=$(round(hmc_param_ess[2], digits=4)), a=$(round(hmc_param_ess[3], digits=4)), b=$(round(hmc_param_ess[4], digits=4))",
)

# ============================================================================
# Plots
# ============================================================================

# Parameter trace plots
p_trace = plot(; layout=(4, 1), size=(800, 900), legend=:topright)

plot!(
    p_trace[1],
    κ_samples;
    subplot=1,
    label="RHMC",
    lw=1,
    color=:blue,
    alpha=0.5,
    title="κ trace",
)
plot!(p_trace[1], κ_samples_hmc; subplot=1, label="HMC", lw=1, color=:green, alpha=0.5)
hline!(p_trace[1], [κ_true]; subplot=1, label="True", lw=2, color=:red)

plot!(
    p_trace[2],
    g_samples;
    subplot=2,
    label="RHMC",
    lw=1,
    color=:blue,
    alpha=0.5,
    title="g trace",
)
plot!(p_trace[2], g_samples_hmc; subplot=2, label="HMC", lw=1, color=:green, alpha=0.5)
hline!(p_trace[2], [g_true]; subplot=2, label="True", lw=2, color=:red)

plot!(
    p_trace[3],
    a_samples;
    subplot=3,
    label="RHMC",
    lw=1,
    color=:blue,
    alpha=0.5,
    title="a trace",
)
plot!(p_trace[3], a_samples_hmc; subplot=3, label="HMC", lw=1, color=:green, alpha=0.5)
hline!(p_trace[3], [a_true]; subplot=3, label="True", lw=2, color=:red)

plot!(
    p_trace[4],
    b_samples;
    subplot=4,
    label="RHMC",
    lw=1,
    color=:blue,
    alpha=0.5,
    title="b trace",
    xlabel="Sample",
)
plot!(p_trace[4], b_samples_hmc; subplot=4, label="HMC", lw=1, color=:green, alpha=0.5)
hline!(p_trace[4], [b_true]; subplot=4, label="True", lw=2, color=:red)

display(p_trace)

# Parameter posterior histograms
p_hist = plot(; layout=(2, 2), size=(800, 600))

histogram!(
    p_hist[1], κ_post; subplot=1, label="RHMC", alpha=0.5, color=:blue, title="κ posterior"
)
histogram!(p_hist[1], κ_post_hmc; subplot=1, label="HMC", alpha=0.5, color=:green)
vline!(p_hist[1], [κ_true]; subplot=1, label="True", lw=2, color=:red)

histogram!(
    p_hist[2], g_post; subplot=2, label="RHMC", alpha=0.5, color=:blue, title="g posterior"
)
histogram!(p_hist[2], g_post_hmc; subplot=2, label="HMC", alpha=0.5, color=:green)
vline!(p_hist[2], [g_true]; subplot=2, label="True", lw=2, color=:red)

histogram!(
    p_hist[3], a_post; subplot=3, label="RHMC", alpha=0.5, color=:blue, title="a posterior"
)
histogram!(p_hist[3], a_post_hmc; subplot=3, label="HMC", alpha=0.5, color=:green)
vline!(p_hist[3], [a_true]; subplot=3, label="True", lw=2, color=:red)

histogram!(
    p_hist[4], b_post; subplot=4, label="RHMC", alpha=0.5, color=:blue, title="b posterior"
)
histogram!(p_hist[4], b_post_hmc; subplot=4, label="HMC", alpha=0.5, color=:green)
vline!(p_hist[4], [b_true]; subplot=4, label="True", lw=2, color=:red)

display(p_hist)

println("\nDone!")
