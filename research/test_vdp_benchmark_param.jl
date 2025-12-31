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
# Configuration
# ============================================================================

SEED = 42
rng = MersenneTwister(SEED)

# Model dimensions
D = 2      # State dimension (u, v)
P = 1      # Parameter dimension (log μ)

# Time discretization
δt = 0.1           # Euler step size
K_y = 10           # Latent steps between observations
N_obs = 20         # Number of observations
K = N_obs * K_y    # Total number of latent states (200)

# Observation indices (1-indexed, every K_y-th state starting from K_y)
obs_indices = collect(K_y:K_y:K)

# ============================================================================
# Ground truth simulation
# ============================================================================

# Prior for initial state
μ0 = @SVector [1.0, 0.0]  # Start near the limit cycle
Σ0 = Diagonal(@SVector([0.1, 0.1]))
prior = MvNormal(μ0, Σ0)

# True parameter value (stiff oscillator)
μ_true = 1.0
θ_true = @SVector [log(μ_true)]

# Dynamics
σ_u = 0.01  # Small noise on u for numerical stability
σ_v = 0.1   # Main process noise on v
dyn = VanDerPolDynamicsParam(σ_u, σ_v, δt)

# Observations (partial linear: y = u + noise)
σ_obs = 0.5
obs = PartialLinearObservationParam(σ_obs)

ssm = SSMParam(prior, dyn, obs)

# Parameter prior (diffuse Gaussian on log scale)
prior_mean = @SVector [log(μ_true)]  # Centered near true value
prior_var = @SVector [4.0]        # Wide variance

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

        # Only observe at specified indices
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

println("Simulating Van der Pol oscillator with μ = $(μ_true)...")
println("Total latent states: K = $K")
println("Number of observations: $(length(obs_indices))")
println("Observations at indices: $(obs_indices[1]):$(K_y):$(obs_indices[end])")

zs_true, ys = simulate(rng, ssm, θ_true, K, obs_indices)
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
    return LogDensityProblems.LogDensityOrder{1}()
end

ℓπ = LogTargetDensityParamSparse(K, D, P, ssm, ys, prior_mean, prior_var, obs_indices)
model = AdvancedHMC.LogDensityModel(ℓπ)

# Initial state: true states + true parameters
initial_θ = vcat(from_block_vector(zs_true_block), collect(θ_true))

# ============================================================================
# RHMC Sampling (Joint state + parameter)
# ============================================================================

println("\nSetting up RHMC sampler...")
metric = BorderedBlockTridiagonalRiemannianMetric(
    ssm, ys, D, P, K, prior_mean, prior_var, obs_indices
)
hamiltonian = Hamiltonian(metric, ℓπ)
initial_ϵ = 0.005
integrator = AdaptiveGeneralizedLeapfrog(initial_ϵ; max_iters=7)
kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
adaptor = StepSizeAdaptor(0.95, integrator)
rhmc = HMCSampler(kernel, metric, adaptor)

N_samples = 5000
N_adapt = 2000

println("Running RHMC with joint state+parameter inference...")
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
μ_samples = [exp(s[K * D + 1]) for s in samples]

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
println("log(μ) ESS: ", param_ess[1])

println("\n=== RHMC Parameter Posterior Summary ===")
burn_in = N_adapt
μ_post = μ_samples[(burn_in + 1):end]
println("μ: true=$(μ_true), posterior mean=$(mean(μ_post)), std=$(std(μ_post))")

# ============================================================================
# Plots
# ============================================================================

# Trace plot for parameter
p_μ = plot(; title="μ trace (RHMC)", xlabel="Sample", ylabel="μ", legend=:topright)
plot!(p_μ, μ_samples; label="Samples", lw=1, color=:blue, alpha=0.5)
hline!(p_μ, [μ_true]; label="True", lw=2, color=:red)

display(p_μ)

# Histogram of posterior parameter
p_μ_hist = histogram(
    μ_post;
    title="μ posterior (RHMC)",
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
    title="Posterior Trajectories (RHMC)",
    xlabel="u",
    ylabel="v",
    legend=:topright,
    size=(800, 600),
)
for i in 1:n_plot_samples
    s = plot_samples[i]
    plot!(
        p2,
        [s[1 + (k - 1) * D] for k in 1:K],
        [s[2 + (k - 1) * D] for k in 1:K];
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
μ_samples_hmc = [exp(s[K * D + 1]) for s in hmc_samples_raw]

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
