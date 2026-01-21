"""
Test script for RHMC with linear LGCP AR(p) model.

This benchmark uses:
- Linear AR(p) latent dynamics: x_{t+1} = μ(1-Σφ) + Σφᵢx_{t-i+1} + σε_t
- Poisson observations: y_t ~ Poisson(exp(b + x_t))

This is a simplified version without nonlinear self-regulation, where the
GN approximation equals the exact observed Hessian for the transition term.

The model creates challenging geometry because:
- Poisson Fisher information exp(η) creates state-dependent observation curvature
- Near-persistent dynamics (Σφᵢ ≈ 1) create long-range correlations

Configuration:
- p = AR order (configurable)
- K = total time steps
- Parameters: φ₁,...,φₚ (AR coeffs), μ (drift), b (intercept)
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

# AR order
P = 3

# Time discretization
K_y = 5            # Latent steps between observations
N_obs = 40         # Number of observations
K = N_obs * K_y    # Total number of latent states

# Observation indices (1-indexed, every K_y-th state starting from K_y)
obs_indices = collect(K_y:K_y:K)

# Process noise
σ = 0.15

# ============================================================================
# Model Setup
# ============================================================================

# True parameter values
# Near-persistent AR(3): coefficients sum to ~0.95
φ_true = SVector{P,Float64}(0.6, 0.25, 0.1)  # Sum = 0.95
μ_true = 0.5        # Drift / long-run mean
b_true = 1.0        # Observation intercept (baseline rate ~ exp(1) ≈ 2.7)

# θ = [φ₁, ..., φₚ, μ, b]
θ_true = SVector{P + 2,Float64}(φ_true..., μ_true, b_true)

# Create model
model = LGCPARpLinearModel{P}(; σ=σ, s0=1.0)

# Dimensions
Dx = state_dim(model)
Dy = obs_dim(model)
Dp = param_dim(model)

println("Model: LGCPARpLinearModel (Linear AR($P))")
println("State dim: $Dx, Obs dim: $Dy, Param dim: $Dp")
println("True parameters:")
println("  φ = $φ_true (sum = $(sum(φ_true)))")
println("  μ = $μ_true")
println("  b = $b_true")

# ============================================================================
# Ground truth simulation
# ============================================================================

println("\nSimulating LGCP with linear AR($P) dynamics...")
println("Total latent states: K = $K")
println("Number of observations: $(length(obs_indices))")

zs_true, ys = simulate(rng, model, θ_true, K; obs_indices=obs_indices)
zs_true_block = BlockVector{Float64,Dx}(zs_true)

# Extract the first component (current log-intensity) for plotting
x_true = [z[1] for z in zs_true]

# Plot latent state trajectory (first component only)
p1 = plot(;
    title="Latent Log-Intensity x_t (Linear AR($P))",
    xlabel="Time step",
    ylabel="x_t",
    legend=:topright,
    size=(900, 400),
)
plot!(p1, 1:K, x_true; label="True x_t", lw=1.5, color=:black)
vline!(p1, obs_indices; label="", color=:gray, alpha=0.2, linestyle=:dash)

display(p1)

# Plot observation counts vs. intensity
η_true = [η(model, zs_true[k], θ_true)[1] for k in obs_indices]
λ_true = exp.(η_true)
counts = [y[1] for y in ys]

p2 = plot(;
    title="Poisson Observations vs. True Intensity",
    xlabel="Time index",
    ylabel="Count / Rate",
    legend=:topright,
)
plot!(p2, obs_indices, λ_true; label="True λ = exp(b + x)", lw=2, color=:red)
scatter!(p2, obs_indices, counts; label="Observed counts", color=:blue, ms=4, alpha=0.7)

display(p2)

println("\nObservation statistics:")
println("  Mean count: $(mean(counts))")
println("  Mean λ: $(mean(λ_true))")
println("  Min/Max λ: $(minimum(λ_true)) / $(maximum(λ_true))")
println("  Min/Max x: $(minimum(x_true)) / $(maximum(x_true))")

# ============================================================================
# LogDensityProblems Model (Joint state + parameter inference)
# ============================================================================

# Parameter prior (diffuse Gaussian)
# θ = [φ₁, ..., φₚ, μ, b]
prior_mean = SVector{P + 2,Float64}(ntuple(i -> i <= P ? 0.3 : 0.0, Val(P + 2)))
prior_var = SVector{P + 2,Float64}(ntuple(_ -> 4.0, Val(P + 2)))

ℓπ = RHMCLogDensity(
    model, ys, K; prior_mean=prior_mean, prior_var=prior_var, obs_indices=obs_indices
)
adv_model = AdvancedHMC.LogDensityModel(ℓπ)

# Initial state: true states + perturbed parameters
initial_θ_params = θ_true .+ 0.1 * randn(rng, P + 2)
initial_θ = vcat(from_block_vector(zs_true_block), collect(initial_θ_params))

# ============================================================================
# RHMC Sampling
# ============================================================================

println("\nSetting up RHMC sampler...")

metric = RiemannianMetric(model, ys, K, prior_mean, prior_var; obs_indices=obs_indices)
hamiltonian = Hamiltonian(metric, ℓπ)
initial_ϵ = 0.01
integrator = AdaptiveGeneralizedLeapfrog(initial_ϵ; max_iters=7)
kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
adaptor = StepSizeAdaptor(0.9, integrator)
rhmc = HMCSampler(kernel, metric, adaptor)

N_samples = 2000
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
param_offset = K * Dx
φ_samples = [[s[param_offset + i] for s in samples] for i in 1:P]
μ_samples = [s[param_offset + P + 1] for s in samples]
b_samples = [s[param_offset + P + 2] for s in samples]

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
for i in 1:P
    println("φ_$i ESS: ", param_ess[i])
end
println("μ ESS: ", param_ess[P + 1])
println("b ESS: ", param_ess[P + 2])

println("\n=== RHMC Parameter Posterior Summary ===")
burn_in = N_adapt
for i in 1:P
    φ_post = φ_samples[i][(burn_in + 1):end]
    println(
        "φ_$i: true=$(φ_true[i]), posterior mean=$(round(mean(φ_post), digits=4)), std=$(round(std(φ_post), digits=4))",
    )
end
μ_post = μ_samples[(burn_in + 1):end]
b_post = b_samples[(burn_in + 1):end]
println(
    "μ: true=$(μ_true), posterior mean=$(round(mean(μ_post), digits=4)), std=$(round(std(μ_post), digits=4))",
)
println(
    "b: true=$(b_true), posterior mean=$(round(mean(b_post), digits=4)), std=$(round(std(b_post), digits=4))",
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
φ_samples_hmc = [[s[param_offset + i] for s in hmc_samples_raw] for i in 1:P]
μ_samples_hmc = [s[param_offset + P + 1] for s in hmc_samples_raw]
b_samples_hmc = [s[param_offset + P + 2] for s in hmc_samples_raw]

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
for i in 1:P
    println("φ_$i ESS: ", hmc_param_ess[i])
end
println("μ ESS: ", hmc_param_ess[P + 1])
println("b ESS: ", hmc_param_ess[P + 2])

println("\n=== HMC Parameter Posterior Summary ===")
for i in 1:P
    φ_post_hmc = φ_samples_hmc[i][(burn_in + 1):end]
    println(
        "φ_$i: true=$(φ_true[i]), posterior mean=$(round(mean(φ_post_hmc), digits=4)), std=$(round(std(φ_post_hmc), digits=4))",
    )
end
μ_post_hmc = μ_samples_hmc[(burn_in + 1):end]
b_post_hmc = b_samples_hmc[(burn_in + 1):end]
println(
    "μ: true=$(μ_true), posterior mean=$(round(mean(μ_post_hmc), digits=4)), std=$(round(std(μ_post_hmc), digits=4))",
)
println(
    "b: true=$(b_true), posterior mean=$(round(mean(b_post_hmc), digits=4)), std=$(round(std(b_post_hmc), digits=4))",
)

# ============================================================================
# Comparison Summary
# ============================================================================

println("\n" * "="^60)
println("COMPARISON SUMMARY: RHMC vs HMC for Linear LGCP AR($P)")
println("="^60)

println("\n--- State ESS ---")
println(
    "RHMC: min=$(round(minimum(state_ess), digits=4)), median=$(round(median(state_ess), digits=4)), mean=$(round(mean(state_ess), digits=4))",
)
println(
    "HMC:  min=$(round(minimum(hmc_state_ess), digits=4)), median=$(round(median(hmc_state_ess), digits=4)), mean=$(round(mean(hmc_state_ess), digits=4))",
)

println("\n--- Parameter ESS ---")
print("RHMC: ")
for i in 1:P
    print("φ_$i=$(round(param_ess[i], digits=4)), ")
end
println("μ=$(round(param_ess[P + 1], digits=4)), b=$(round(param_ess[P + 2], digits=4))")

print("HMC:  ")
for i in 1:P
    print("φ_$i=$(round(hmc_param_ess[i], digits=4)), ")
end
println(
    "μ=$(round(hmc_param_ess[P + 1], digits=4)), b=$(round(hmc_param_ess[P + 2], digits=4))"
)

# ============================================================================
# Plots
# ============================================================================

# Parameter trace plots
n_params = P + 2
p_trace = plot(; layout=(n_params, 1), size=(800, 150 * n_params), legend=:topright)

for i in 1:P
    plot!(
        p_trace[i],
        φ_samples[i];
        subplot=i,
        label="RHMC",
        lw=1,
        color=:blue,
        alpha=0.5,
        title="φ_$i trace",
    )
    plot!(
        p_trace[i], φ_samples_hmc[i]; subplot=i, label="HMC", lw=1, color=:green, alpha=0.5
    )
    hline!(p_trace[i], [φ_true[i]]; subplot=i, label="True", lw=2, color=:red)
end

plot!(
    p_trace[P + 1],
    μ_samples;
    subplot=P + 1,
    label="RHMC",
    lw=1,
    color=:blue,
    alpha=0.5,
    title="μ trace",
)
plot!(
    p_trace[P + 1], μ_samples_hmc; subplot=P + 1, label="HMC", lw=1, color=:green, alpha=0.5
)
hline!(p_trace[P + 1], [μ_true]; subplot=P + 1, label="True", lw=2, color=:red)

plot!(
    p_trace[P + 2],
    b_samples;
    subplot=P + 2,
    label="RHMC",
    lw=1,
    color=:blue,
    alpha=0.5,
    title="b trace",
    xlabel="Sample",
)
plot!(
    p_trace[P + 2], b_samples_hmc; subplot=P + 2, label="HMC", lw=1, color=:green, alpha=0.5
)
hline!(p_trace[P + 2], [b_true]; subplot=P + 2, label="True", lw=2, color=:red)

display(p_trace)

# Posterior state trajectory comparison
println("\nPlotting posterior state trajectories...")
n_plot_samples = 100
plot_idxs = round.(Int, LinRange(burn_in + 1, N_samples, n_plot_samples))

p_traj = plot(;
    title="Posterior x_t Trajectories (Linear AR($P))",
    xlabel="Time step",
    ylabel="x_t",
    legend=:topright,
    size=(900, 400),
)

# Plot RHMC samples
for idx in plot_idxs
    s = samples[idx]
    x_sample = [s[(k - 1) * Dx + 1] for k in 1:K]
    plot!(p_traj, 1:K, x_sample; label="", lw=0.5, alpha=0.1, color=:blue)
end

# Plot HMC samples
for idx in plot_idxs
    s = hmc_samples_raw[idx]
    x_sample = [s[(k - 1) * Dx + 1] for k in 1:K]
    plot!(p_traj, 1:K, x_sample; label="", lw=0.5, alpha=0.1, color=:green)
end

# Plot truth on top
plot!(p_traj, 1:K, x_true; label="Truth", lw=2, color=:black)

# Add dummy lines for legend
plot!(p_traj, [NaN], [NaN]; label="RHMC", color=:blue, lw=2)
plot!(p_traj, [NaN], [NaN]; label="HMC", color=:green, lw=2)

display(p_traj)

println("\nDone!")
