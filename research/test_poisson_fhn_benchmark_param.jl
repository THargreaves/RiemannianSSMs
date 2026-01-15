"""
FitzHugh-Nagumo + Multi-Channel Poisson Benchmark for RHMC vs HMC

This benchmark follows the recommendations from poisson_model_2.md to create
a test case where RHMC should outperform HMC:

1. Excitable FHN dynamics produce occasional spikes separated by calm periods
2. Two-channel Poisson observations with parameter-dependent scaling
3. Small process noise creates a "tight tube" posterior
4. Heterogeneous curvature across time (high λ during spikes, low otherwise)

The key insight is that Euclidean HMC must use a globally small step size
dictated by the worst (highest curvature) time points, while RHMC can adapt
locally via the Fisher/GGN metric.
"""

using Distributions
using LinearAlgebra
using Random
using Plots
using StaticArrays

using AbstractMCMC
using AdvancedHMC
using LogDensityProblems
using MCMCDiagnosticTools

using RiemannianSSMs

# ============================================================================
# Configuration (following poisson_model_2.md recommendations)
# ============================================================================

SEED = 42
rng = MersenneTwister(SEED)

# Model dimensions
D = 2      # State dimension (u, v)
P = 1      # Parameter dimension (θ = log scale)

# Time discretization
δt = 0.05          # Euler step size
K = 800            # Total number of latent states (T = 800 as recommended)

# Observation indices (observe at every time step for maximum curvature variation)
obs_indices = collect(1:K)

# FitzHugh-Nagumo parameters (fixed, not inferred)
fhn_a = 0.7
fhn_b = 0.8
fhn_τ = 12.0
fhn_I = 0.5

# Process noise (small for "tight tube" posterior)
σ_u = sqrt(1.0e-4)   # σ_u² = 1e-4
σ_v = sqrt(2.5e-4)   # σ_v² = 2.5e-4

# Observation model coefficients
# η₁ = 1.0 · exp(θ) · u + b₁
# η₂ = 0.6 · exp(θ) · v + b₂
A = @SMatrix [1.0 0.0; 0.0 1.0]  # Identity projection
c = @SVector [1.0, 0.6]          # Per-channel scales
b = @SVector [-1.8, -1.2]        # Per-channel offsets

# ============================================================================
# Ground truth simulation
# ============================================================================

# Prior for initial state
μ0 = @SVector [-1.0, 1.0]  # Start in calm region
Σ0 = Diagonal(@SVector([0.01, 0.01]))  # Tight initial prior
prior = MvNormal(μ0, Σ0)

# True parameter value
θ_true_val = 0.8  # s(θ) = exp(0.8) ≈ 2.23
θ_true = @SVector [θ_true_val]

# Dynamics
dyn = FitzHughNagumoParam(σ_u, σ_v, δt, fhn_a, fhn_b, fhn_τ, fhn_I)

# Observations
obs = PoissonMultiChannelParam(A, c, b)

ssm = SSMParam(prior, dyn, obs)

# Parameter prior (Gaussian on log scale)
prior_mean = @SVector [0.0]   # Weakly informative
prior_var = @SVector [1.0]    # Moderate variance

function simulate(rng::AbstractRNG, ssm, θ, K::Int, obs_indices)
    zs = Vector{SVector{2,Float64}}(undef, K)
    ys = Vector{SVector{2,Int64}}(undef, length(obs_indices))

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
            η = h_param(ssm.sensor, z, θ)
            λ = exp.(η)
            # Clamp λ to avoid numerical issues
            λ_clamped = clamp.(λ, 1e-10, 1e6)
            y1 = rand(rng, Poisson(λ_clamped[1]))
            y2 = rand(rng, Poisson(λ_clamped[2]))
            ys[obs_idx] = @SVector [y1, y2]
            obs_idx += 1
        end
    end

    return zs, ys
end

println("Simulating FitzHugh-Nagumo with Poisson observations...")
println("FHN parameters: a=$(fhn_a), b=$(fhn_b), τ=$(fhn_τ), I=$(fhn_I)")
println("True θ = $(θ_true_val), s(θ) = $(exp(θ_true_val))")
println("Total latent states: K = $K")
println("Number of observations: $(length(obs_indices))")

zs_true, ys = simulate(rng, ssm, θ_true, K, obs_indices)
zs_true_block = BlockVector{Float64,2}(zs_true)

# Plot the FHN trajectory
p1 = plot(;
    title="FitzHugh-Nagumo State Trajectory",
    xlabel="u (fast variable)",
    ylabel="v (slow variable)",
    legend=:topright,
    size=(800, 600),
)
plot!(
    p1,
    [z[1] for z in zs_true],
    [z[2] for z in zs_true];
    label="True trajectory",
    lw=1,
    color=:black,
    alpha=0.7,
)
scatter!(
    p1, [zs_true[1][1]], [zs_true[1][2]]; label="Start", color=:green, ms=8, marker=:circle
)
scatter!(
    p1, [zs_true[end][1]], [zs_true[end][2]]; label="End", color=:red, ms=8, marker=:square
)
display(p1)

# Plot time series of u and v
p_ts = plot(; layout=(2, 1), size=(1000, 600), legend=:topright)
plot!(p_ts[1], 1:K, [z[1] for z in zs_true]; label="u(t)", lw=1, color=:blue)
ylabel!(p_ts[1], "u")
title!(p_ts[1], "FHN State Time Series")
plot!(p_ts[2], 1:K, [z[2] for z in zs_true]; label="v(t)", lw=1, color=:red)
xlabel!(p_ts[2], "Time step")
ylabel!(p_ts[2], "v")
display(p_ts)

# Plot observation statistics
η_true = [h_param(ssm.sensor, zs_true[k], θ_true) for k in obs_indices]
λ_true = [exp.(η) for η in η_true]
p2 = plot(; layout=(2, 1), size=(1000, 600), legend=:topright)

plot!(p2[1], obs_indices, [λ[1] for λ in λ_true]; label="λ₁ (channel 1)", lw=1, color=:blue)
scatter!(
    p2[1],
    obs_indices,
    [y[1] for y in ys];
    label="y₁ (counts)",
    color=:blue,
    ms=2,
    alpha=0.5,
)
ylabel!(p2[1], "Rate / Count")
title!(p2[1], "Poisson Observations vs True Intensity")

plot!(p2[2], obs_indices, [λ[2] for λ in λ_true]; label="λ₂ (channel 2)", lw=1, color=:red)
scatter!(
    p2[2], obs_indices, [y[2] for y in ys]; label="y₂ (counts)", color=:red, ms=2, alpha=0.5
)
xlabel!(p2[2], "Time step")
ylabel!(p2[2], "Rate / Count")
display(p2)

println("\nObservation statistics:")
println(
    "  Channel 1: mean λ = $(mean([λ[1] for λ in λ_true])), range = $(extrema([λ[1] for λ in λ_true]))",
)
println(
    "  Channel 2: mean λ = $(mean([λ[2] for λ in λ_true])), range = $(extrema([λ[2] for λ in λ_true]))",
)
println("  η₁ range: $(extrema([η[1] for η in η_true]))")
println("  η₂ range: $(extrema([η[2] for η in η_true]))")

# ============================================================================
# LogDensityProblems Model
# ============================================================================

struct LogTargetDensityFHNPoisson{D,P,M,V,PM,PV,OI}
    dim::Int
    ssm::M
    ys::V
    prior_mean::PM
    prior_prec::PV
    obs_indices::OI
end

function LogTargetDensityFHNPoisson(
    K::Int, D::Int, P::Int, ssm, ys, prior_mean, prior_var, obs_indices
)
    dim = K * D + P
    prior_prec = SVector{P,Float64}(1 ./ prior_var)
    return LogTargetDensityFHNPoisson{
        D,P,typeof(ssm),typeof(ys),typeof(prior_mean),typeof(prior_prec),typeof(obs_indices)
    }(
        dim, ssm, ys, prior_mean, prior_prec, obs_indices
    )
end

function LogDensityProblems.logdensity(p::LogTargetDensityFHNPoisson{D,P}, θ) where {D,P}
    θ_blocks = to_bordered_block_vector(θ, Val(D), Val(P))
    return calc_ll_param_poisson(
        θ_blocks, p.ys, p.ssm, p.prior_mean, p.prior_prec; obs_indices=p.obs_indices
    )
end

function LogDensityProblems.logdensity_and_gradient(
    p::LogTargetDensityFHNPoisson{D,P}, θ
) where {D,P}
    θ_blocks = to_bordered_block_vector(θ, Val(D), Val(P))
    ll = calc_ll_param_poisson(
        θ_blocks, p.ys, p.ssm, p.prior_mean, p.prior_prec; obs_indices=p.obs_indices
    )
    grad = calc_ll_grad_param_poisson(
        θ_blocks, p.ys, p.ssm, p.prior_mean, p.prior_prec; obs_indices=p.obs_indices
    )
    return ll, from_bordered_block_vector(grad)
end

LogDensityProblems.dimension(p::LogTargetDensityFHNPoisson) = p.dim

function LogDensityProblems.capabilities(::Type{<:LogTargetDensityFHNPoisson})
    return LogDensityProblems.LogDensityOrder{1}()
end

ℓπ = LogTargetDensityFHNPoisson(K, D, P, ssm, ys, prior_mean, prior_var, obs_indices)
model = AdvancedHMC.LogDensityModel(ℓπ)

# Initial state: true states + perturbed parameter
initial_θ_param = θ_true .+ 0.1 * randn(rng, 1)
initial_θ = vcat(from_block_vector(zs_true_block), collect(initial_θ_param))

# ============================================================================
# RHMC Sampling
# ============================================================================

println("\nSetting up RHMC sampler for FHN + Poisson SSM...")

metric = PoissonRiemannianMetric(ssm, ys, D, P, K, prior_mean, prior_var, obs_indices)
hamiltonian = Hamiltonian(metric, ℓπ)
initial_ϵ = 0.005
integrator = AdaptiveGeneralizedLeapfrog(initial_ϵ; max_iters=7)
kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
adaptor = StepSizeAdaptor(0.95, integrator)
rhmc = HMCSampler(kernel, metric, adaptor)

N_samples = 1000
N_adapt = 250

println("Running RHMC...")
rhmc_start = time()
chains = AbstractMCMC.sample(
    model,
    rhmc,
    N_samples;
    n_adapts=N_adapt,
    initial_params=initial_θ,
    verbose=false,
    progress=true,
);
rhmc_time = time() - rhmc_start
samples = [s.z.θ for s in chains];

# Extract parameter samples
θ_samples = [s[K * D + 1] for s in samples]

# Compute ESS
total_dim = D * K + P
rhmc_samples_arr = Array{Float64}(undef, N_samples, 1, total_dim)
for i in 1:N_samples
    for j in 1:total_dim
        rhmc_samples_arr[i, 1, j] = samples[i][j]
    end
end
rhmc_ess = ess(rhmc_samples_arr) ./ N_samples

println("\n=== RHMC Results ===")
println("Wall time: $(round(rhmc_time, digits=2)) seconds")

state_ess = rhmc_ess[1:(D * K)]
println(
    "State ESS: min=$(round(minimum(state_ess), digits=4)), median=$(round(median(state_ess), digits=4)), mean=$(round(mean(state_ess), digits=4))",
)

param_ess = rhmc_ess[(D * K + 1):end]
println("θ ESS: $(round(param_ess[1], digits=4))")

burn_in = N_adapt
θ_post = θ_samples[(burn_in + 1):end]
println(
    "θ posterior: true=$(θ_true_val), mean=$(round(mean(θ_post), digits=4)), std=$(round(std(θ_post), digits=4))",
)

# ============================================================================
# HMC Sampling (for comparison)
# ============================================================================

println("\nRunning HMC (NUTS)...")
# hmc = NUTS(0.8)
hmc = NUTS(0.8; metric=:dense)
hmc_start = time()
hmc_chains = AbstractMCMC.sample(
    model,
    hmc,
    N_samples;
    n_adapts=N_adapt,
    initial_params=initial_θ,
    verbose=false,
    progress=true,
);
hmc_time = time() - hmc_start
hmc_samples_raw = [s.z.θ for s in hmc_chains];

θ_samples_hmc = [s[K * D + 1] for s in hmc_samples_raw]

hmc_samples_arr = Array{Float64}(undef, N_samples, 1, total_dim)
for i in 1:N_samples
    for j in 1:total_dim
        hmc_samples_arr[i, 1, j] = hmc_samples_raw[i][j]
    end
end
hmc_ess = ess(hmc_samples_arr) ./ N_samples

println("\n=== HMC Results ===")
println("Wall time: $(round(hmc_time, digits=2)) seconds")

hmc_state_ess = hmc_ess[1:(D * K)]
println(
    "State ESS: min=$(round(minimum(hmc_state_ess), digits=4)), median=$(round(median(hmc_state_ess), digits=4)), mean=$(round(mean(hmc_state_ess), digits=4))",
)

hmc_param_ess = hmc_ess[(D * K + 1):end]
println("θ ESS: $(round(hmc_param_ess[1], digits=4))")

θ_post_hmc = θ_samples_hmc[(burn_in + 1):end]
println(
    "θ posterior: true=$(θ_true_val), mean=$(round(mean(θ_post_hmc), digits=4)), std=$(round(std(θ_post_hmc), digits=4))",
)

# ============================================================================
# Comparison Summary
# ============================================================================

println("\n" * "="^70)
println("COMPARISON SUMMARY: RHMC vs HMC for FHN + Poisson SSM")
println("="^70)

println("\n--- Timing ---")
println("RHMC: $(round(rhmc_time, digits=2)) seconds")
println("HMC:  $(round(hmc_time, digits=2)) seconds")

println("\n--- State ESS ---")
println(
    "RHMC: min=$(round(minimum(state_ess), digits=4)), median=$(round(median(state_ess), digits=4)), mean=$(round(mean(state_ess), digits=4))",
)
println(
    "HMC:  min=$(round(minimum(hmc_state_ess), digits=4)), median=$(round(median(hmc_state_ess), digits=4)), mean=$(round(mean(hmc_state_ess), digits=4))",
)

println("\n--- Parameter ESS ---")
println("RHMC: θ=$(round(param_ess[1], digits=4))")
println("HMC:  θ=$(round(hmc_param_ess[1], digits=4))")

println("\n--- ESS per second ---")
println(
    "RHMC: state mean=$(round(mean(state_ess) * N_samples / rhmc_time, digits=2)), θ=$(round(param_ess[1] * N_samples / rhmc_time, digits=2))",
)
println(
    "HMC:  state mean=$(round(mean(hmc_state_ess) * N_samples / hmc_time, digits=2)), θ=$(round(hmc_param_ess[1] * N_samples / hmc_time, digits=2))",
)

# ============================================================================
# Comparison Plots
# ============================================================================

# Parameter trace plot
p_θ = plot(; title="θ trace comparison", xlabel="Sample", ylabel="θ", legend=:topright)
plot!(p_θ, θ_samples; label="RHMC", lw=1, color=:blue, alpha=0.5)
plot!(p_θ, θ_samples_hmc; label="HMC", lw=1, color=:green, alpha=0.5)
hline!(p_θ, [θ_true_val]; label="True", lw=2, color=:red)
display(p_θ)

# Parameter histogram
p_θ_hist = plot(; layout=(1, 2), size=(1000, 400), legend=:topright)
histogram!(
    p_θ_hist[1],
    θ_post;
    title="RHMC θ posterior",
    xlabel="θ",
    ylabel="Count",
    label="",
    bins=50,
    color=:blue,
    alpha=0.7,
)
vline!(p_θ_hist[1], [θ_true_val]; label="True", lw=2, color=:red)

histogram!(
    p_θ_hist[2],
    θ_post_hmc;
    title="HMC θ posterior",
    xlabel="θ",
    ylabel="Count",
    label="",
    bins=50,
    color=:green,
    alpha=0.7,
)
vline!(p_θ_hist[2], [θ_true_val]; label="True", lw=2, color=:red)
display(p_θ_hist)

# ESS comparison
p_ess = plot(;
    title="ESS per dimension", xlabel="Dimension", ylabel="ESS", legend=:topright
)
plot!(p_ess, 1:(D * K), state_ess; label="RHMC states", lw=1, color=:blue, alpha=0.7)
plot!(p_ess, 1:(D * K), hmc_state_ess; label="HMC states", lw=1, color=:green, alpha=0.7)
display(p_ess)

println("\nDone!")
