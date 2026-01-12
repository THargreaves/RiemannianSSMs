using Distributions
using LinearAlgebra
using Random
using Plots
using StaticArrays

using AbstractMCMC
using AdvancedHMC
using LogDensityProblems
using MCMCDiagnosticTools

# Load the package (will need to add includes for new files)
using RiemannianSSMs

# ============================================================================
# Configuration
# ============================================================================

SEED = 42
rng = MersenneTwister(SEED)

# Model dimensions
D = 2      # State dimension (x_1, x_2)
P = 3      # Parameter dimension (a_1, a_2, b)

# Time discretization
K_y = 5           # Latent steps between observations
N_obs = 40         # Number of observations
K = N_obs * K_y    # Total number of latent states (200)

# Observation indices (1-indexed, every K_y-th state starting from K_y)
obs_indices = collect(K_y:K_y:K)

# ============================================================================
# Ground truth simulation
# ============================================================================

# Prior for initial state
μ0 = @SVector [1.0, 2.0]  # Start at origin
Σ0 = Diagonal(@SVector([1.0, 1.0]))
prior = MvNormal(μ0, Σ0)

# True parameter values
a_1_true = -3.0
a_2_true = 2.5
b_true = -2.0
θ_true = @SVector [a_1_true, a_2_true, b_true]

# Dynamics (random walk)
σ_1 = 0.01  # Process noise on x_1
σ_2 = 0.1  # Process noise on x_2
dyn = RandomWalkDynamicsParam(σ_1, σ_2)

# Observations (Poisson with log-linear predictor)
obs = PoissonObservationParam()

ssm = SSMParam(prior, dyn, obs)

# Parameter prior (diffuse Gaussian)
prior_mean = @SVector [0.0, 0.0, 0.0]  # Weakly informative
prior_var = @SVector [4.0, 4.0, 4.0]   # Wide variance

function simulate(rng::AbstractRNG, ssm, θ, K::Int, obs_indices)
    zs = Vector{SVector{2,Float64}}(undef, K)
    ys = Vector{SVector{1,Int64}}(undef, length(obs_indices))

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

        # Only observe at specified indices
        if k in obs_indices
            η = h_param(ssm.sensor, z, θ)[1]
            λ = exp(η)
            y = rand(rng, Poisson(λ))
            ys[obs_idx] = @SVector [y]
            obs_idx += 1
        end
    end

    return zs, ys
end

println("Simulating random walk with Poisson observations...")
println("True parameters: a_1=$(a_1_true), a_2=$(a_2_true), b=$(b_true)")
println("Total latent states: K = $K")
println("Number of observations: $(length(obs_indices))")
println("Observations at indices: $(obs_indices[1]):$(K_y):$(obs_indices[end])")

zs_true, ys = simulate(rng, ssm, θ_true, K, obs_indices)
zs_true_block = BlockVector{Float64,2}(zs_true)

# Plot the trajectory
p1 = plot(;
    title="Random Walk State Trajectory",
    xlabel="x_1",
    ylabel="x_2",
    legend=:topright,
    size=(800, 600),
    aspect_ratio=1,
)
plot!(
    p1,
    [z[1] for z in zs_true],
    [z[2] for z in zs_true];
    label="True states",
    lw=2,
    color=:black,
)
scatter!(
    p1,
    [zs_true[k][1] for k in obs_indices],
    [zs_true[k][2] for k in obs_indices];
    label="Observed states",
    color=:blue,
    ms=4,
)
display(p1)

# Plot observation counts vs. intensity
η_true = [h_param(ssm.sensor, zs_true[k], θ_true)[1] for k in obs_indices]
λ_true = exp.(η_true)
p2 = plot(;
    title="Poisson Observations vs. True Intensity",
    xlabel="Time index",
    ylabel="Count / Rate",
)
plot!(p2, obs_indices, λ_true; label="True λ", lw=2, color=:red)
scatter!(
    p2,
    obs_indices,
    [y[1] for y in ys];
    label="Observed counts",
    color=:blue,
    ms=4,
    alpha=0.7,
)
display(p2)

println("\nObservation statistics:")
println("  Mean count: $(mean([y[1] for y in ys]))")
println("  Mean λ: $(mean(λ_true))")
println("  Min/Max λ: $(minimum(λ_true)) / $(maximum(λ_true))")

# ============================================================================
# LogDensityProblems Model (Joint state + parameter inference)
# ============================================================================

struct LogTargetDensityPoissonParam{D,P,M,V,PM,PV,OI}
    dim::Int
    ssm::M
    ys::V
    prior_mean::PM
    prior_prec::PV
    obs_indices::OI
end

function LogTargetDensityPoissonParam(
    K::Int, D::Int, P::Int, ssm, ys, prior_mean, prior_var, obs_indices
)
    dim = K * D + P
    prior_prec = SVector{P,Float64}(1 ./ prior_var)
    return LogTargetDensityPoissonParam{
        D,P,typeof(ssm),typeof(ys),typeof(prior_mean),typeof(prior_prec),typeof(obs_indices)
    }(
        dim, ssm, ys, prior_mean, prior_prec, obs_indices
    )
end

function LogDensityProblems.logdensity(p::LogTargetDensityPoissonParam{D,P}, θ) where {D,P}
    θ_blocks = to_bordered_block_vector(θ, Val(D), Val(P))
    return calc_ll_param_poisson(
        θ_blocks, p.ys, p.ssm, p.prior_mean, p.prior_prec; obs_indices=p.obs_indices
    )
end

function LogDensityProblems.logdensity_and_gradient(
    p::LogTargetDensityPoissonParam{D,P}, θ
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

LogDensityProblems.dimension(p::LogTargetDensityPoissonParam) = p.dim

function LogDensityProblems.capabilities(::Type{<:LogTargetDensityPoissonParam})
    return LogDensityProblems.LogDensityOrder{1}()
end

ℓπ = LogTargetDensityPoissonParam(K, D, P, ssm, ys, prior_mean, prior_var, obs_indices)
model = AdvancedHMC.LogDensityModel(ℓπ)

# Initial state: true states + perturbed parameters
initial_θ_params = θ_true .+ 0.1 * randn(rng, 3)  # Start near truth
initial_θ = vcat(from_block_vector(zs_true_block), collect(initial_θ_params))

# ============================================================================
# RHMC Sampling (Joint state + parameter with Poisson)
# ============================================================================

println("\nSetting up RHMC sampler for Poisson SSM...")

# Use the PoissonRiemannianMetric exported from RiemannianSSMs
metric = PoissonRiemannianMetric(ssm, ys, D, P, K, prior_mean, prior_var, obs_indices)
hamiltonian = Hamiltonian(metric, ℓπ)
initial_ϵ = 0.01
integrator = AdaptiveGeneralizedLeapfrog(initial_ϵ; max_iters=7)
kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
adaptor = StepSizeAdaptor(0.95, integrator)
rhmc = HMCSampler(kernel, metric, adaptor)

N_samples = 3000
N_adapt = 1000

println("Running RHMC with joint state+parameter inference (Poisson)...")
chains = AbstractMCMC.sample(
    model,
    rhmc,
    N_samples;
    n_adapts=N_adapt,
    initial_params=initial_θ,
    verbose=false,
    progress=true,
);
samples = [s.z.θ for s in chains];

# Extract parameter samples (last P elements)
a_1_samples = [s[K * D + 1] for s in samples]
a_2_samples = [s[K * D + 2] for s in samples]
b_samples = [s[K * D + 3] for s in samples]

# Compute ESS for all dimensions
total_dim = D * K + P
rhmc_samples = Array{Float64}(undef, N_samples, 1, total_dim)
for i in 1:N_samples
    for j in 1:total_dim
        rhmc_samples[i, 1, j] = samples[i][j]
    end
end
rhmc_ess = ess(rhmc_samples) ./ N_samples

println("\n=== RHMC State ESS Statistics ===")
state_ess = rhmc_ess[1:(D * K)]
println("Minimum State ESS: ", minimum(state_ess))
println("Median State ESS: ", median(state_ess))
println("Mean State ESS: ", mean(state_ess))

println("\n=== RHMC Parameter ESS ===")
param_ess = rhmc_ess[(D * K + 1):end]
println("a_1 ESS: ", param_ess[1])
println("a_2 ESS: ", param_ess[2])
println("b ESS: ", param_ess[3])

println("\n=== RHMC Parameter Posterior Summary ===")
burn_in = N_adapt
a_1_post = a_1_samples[(burn_in + 1):end]
a_2_post = a_2_samples[(burn_in + 1):end]
b_post = b_samples[(burn_in + 1):end]
println("a_1: true=$(a_1_true), posterior mean=$(mean(a_1_post)), std=$(std(a_1_post))")
println("a_2: true=$(a_2_true), posterior mean=$(mean(a_2_post)), std=$(std(a_2_post))")
println("b: true=$(b_true), posterior mean=$(mean(b_post)), std=$(std(b_post))")

# ============================================================================
# HMC Sampling (for comparison)
# ============================================================================

println("\nRunning HMC with joint state+parameter inference (Poisson)...")
hmc = NUTS(0.8)
hmc_chains = AbstractMCMC.sample(
    model,
    hmc,
    N_samples;
    n_adapts=N_adapt,
    initial_params=initial_θ,
    verbose=false,
    progress=true,
);
hmc_samples_raw = [s.z.θ for s in hmc_chains];

# Extract parameter samples
a_1_samples_hmc = [s[K * D + 1] for s in hmc_samples_raw]
a_2_samples_hmc = [s[K * D + 2] for s in hmc_samples_raw]
b_samples_hmc = [s[K * D + 3] for s in hmc_samples_raw]

# Compute ESS
hmc_samples = Array{Float64}(undef, N_samples, 1, total_dim)
for i in 1:N_samples
    for j in 1:total_dim
        hmc_samples[i, 1, j] = hmc_samples_raw[i][j]
    end
end
hmc_ess = ess(hmc_samples) ./ N_samples

println("\n=== HMC State ESS Statistics ===")
hmc_state_ess = hmc_ess[1:(D * K)]
println("Minimum State ESS: ", minimum(hmc_state_ess))
println("Median State ESS: ", median(hmc_state_ess))
println("Mean State ESS: ", mean(hmc_state_ess))

println("\n=== HMC Parameter ESS ===")
hmc_param_ess = hmc_ess[(D * K + 1):end]
println("a_1 ESS: ", hmc_param_ess[1])
println("a_2 ESS: ", hmc_param_ess[2])
println("b ESS: ", hmc_param_ess[3])

println("\n=== HMC Parameter Posterior Summary ===")
a_1_post_hmc = a_1_samples_hmc[(burn_in + 1):end]
a_2_post_hmc = a_2_samples_hmc[(burn_in + 1):end]
b_post_hmc = b_samples_hmc[(burn_in + 1):end]
println(
    "a_1: true=$(a_1_true), posterior mean=$(mean(a_1_post_hmc)), std=$(std(a_1_post_hmc))"
)
println(
    "a_2: true=$(a_2_true), posterior mean=$(mean(a_2_post_hmc)), std=$(std(a_2_post_hmc))"
)
println("b: true=$(b_true), posterior mean=$(mean(b_post_hmc)), std=$(std(b_post_hmc))")

# ============================================================================
# Comparison Summary
# ============================================================================

println("\n" * "="^60)
println("COMPARISON SUMMARY: RHMC vs HMC for Poisson SSM")
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
    "RHMC: a_1=$(round(param_ess[1], digits=4)), a_2=$(round(param_ess[2], digits=4)), b=$(round(param_ess[3], digits=4))",
)
println(
    "HMC:  a_1=$(round(hmc_param_ess[1], digits=4)), a_2=$(round(hmc_param_ess[2], digits=4)), b=$(round(hmc_param_ess[3], digits=4))",
)

# Comparison plots
p_compare = plot(; layout=(3, 1), size=(800, 900), legend=:topright)

plot!(
    p_compare[1],
    a_1_samples;
    subplot=1,
    label="RHMC",
    lw=1,
    color=:blue,
    alpha=0.5,
    title="a_1 trace",
)
plot!(p_compare[1], a_1_samples_hmc; subplot=1, label="HMC", lw=1, color=:green, alpha=0.5)
hline!(p_compare[1], [a_1_true]; subplot=1, label="True", lw=2, color=:red)

plot!(
    p_compare[2],
    a_2_samples;
    subplot=2,
    label="RHMC",
    lw=1,
    color=:blue,
    alpha=0.5,
    title="a_2 trace",
)
plot!(p_compare[2], a_2_samples_hmc; subplot=2, label="HMC", lw=1, color=:green, alpha=0.5)
hline!(p_compare[2], [a_2_true]; subplot=2, label="True", lw=2, color=:red)

plot!(
    p_compare[3],
    b_samples;
    subplot=3,
    label="RHMC",
    lw=1,
    color=:blue,
    alpha=0.5,
    title="b trace",
    xlabel="Sample",
)
plot!(p_compare[3], b_samples_hmc; subplot=3, label="HMC", lw=1, color=:green, alpha=0.5)
hline!(p_compare[3], [b_true]; subplot=3, label="True", lw=2, color=:red)

display(p_compare)

println("\nDone!")
