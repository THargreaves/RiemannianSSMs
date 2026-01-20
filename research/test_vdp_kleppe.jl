"""
Test Kleppe's observed Hessian metric for RMHMC on Van der Pol oscillator.

This compares the ObservedHessianMetric (using modified Cholesky on the observed Hessian)
against the existing BorderedBlockTridiagonalRiemannianMetric (Fisher/GGN approximation).
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
σ_u = 0.1  # Small noise on u for numerical stability
σ_v = 0.1   # Main process noise on v
dyn = VanDerPolDynamicsParam(σ_u, σ_v, δt)

# Observations (partial linear: y = u + noise)
σ_obs = 0.05
obs = PartialLinearObservationParam(σ_obs)

ssm = SSMParam(prior, dyn, obs)

# Parameter prior (diffuse Gaussian on log scale)
prior_mean = @SVector [log(μ_true)]  # Centered near true value
prior_var = @SVector [4.0]           # Wide variance

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
# Kleppe's Observed Hessian RHMC
# ============================================================================

println("\n" * "="^60)
println("KLEPPE'S OBSERVED HESSIAN RHMC")
println("="^60)

# Regularization parameter for modified Cholesky
# u_i sets a lower bound on diagonal D_ii >= u_i
# Needs to be large enough to handle negative eigenvalues from observed Hessian
n = K * D + P
u = fill(1e1, n)  # Larger regularization for stability

println("\nSetting up Kleppe's Observed Hessian metric...")
kleppe_metric = ObservedHessianMetric(
    ssm, ys, D, P, K, prior_mean, prior_var, obs_indices, u
)
println(kleppe_metric)

# Build Hamiltonian with the observed Hessian metric
kleppe_hamiltonian = Hamiltonian(kleppe_metric, ℓπ)

# Use the same integrator settings as the Fisher/GGN approach
initial_ϵ = 0.1
kleppe_integrator = AdaptiveGeneralizedLeapfrog(initial_ϵ; max_iters=7)
kleppe_kernel = HMCKernel(
    Trajectory{MultinomialTS}(kleppe_integrator, GeneralisedNoUTurn())
)
# Fixed integration length
# kleppe_kernel = HMCKernel(
#     Trajectory{MultinomialTS}(kleppe_integrator, FixedNSteps(10))
# )
kleppe_adaptor = StepSizeAdaptor(0.8, kleppe_integrator)
# kleppe_adaptor = NoAdaptation()
kleppe_sampler = HMCSampler(kleppe_kernel, kleppe_metric, kleppe_adaptor)

# For quick testing, use small sample sizes
N_samples = 1000
N_adapt = 500

println("\nRunning Kleppe RHMC (N_samples=$N_samples, N_adapt=$N_adapt)...")
kleppe_chains = AbstractMCMC.sample(
    model,
    kleppe_sampler,
    N_samples;
    n_adapts=N_adapt,
    initial_params=initial_θ,
    verbose=false,
    progress=true,
);
kleppe_samples = [s.z.θ for s in kleppe_chains];

# Print total acceptance rate
accept = [s.stat.is_accept for s in kleppe_chains]
println("Total Acceptance Rate: ", mean(accept))

accept_prop = [s.stat.acceptance_rate for s in kleppe_chains]
println("Mean Acceptance Probability: ", mean(accept_prop))

# Extract parameter samples (last P elements)
kleppe_μ_samples = [exp(s[K * D + 1]) for s in kleppe_samples]

# Compute ESS for all dimensions
total_dim = D * K + P
kleppe_samples_arr = Array{Float64}(undef, N_samples, 1, total_dim)
for i in 1:N_samples
    for j in 1:total_dim
        kleppe_samples_arr[i, 1, j] = kleppe_samples[i][j]
    end
end
kleppe_ess = ess(kleppe_samples_arr) ./ N_samples

println("\n=== Kleppe RHMC State ESS Statistics ===")
kleppe_state_ess = kleppe_ess[1:(D * K)]
println("Minimum State ESS: ", minimum(kleppe_state_ess))
println("Median State ESS: ", median(kleppe_state_ess))
println("Mean State ESS: ", mean(kleppe_state_ess))

println("\n=== Kleppe RHMC Parameter ESS ===")
kleppe_param_ess = kleppe_ess[(D * K + 1):end]
println("log(μ) ESS: ", kleppe_param_ess[1])

println("\n=== Kleppe RHMC Parameter Posterior Summary ===")
burn_in = N_adapt
kleppe_μ_post = kleppe_μ_samples[(burn_in + 1):end]
println(
    "μ: true=$(μ_true), posterior mean=$(mean(kleppe_μ_post)), std=$(std(kleppe_μ_post))"
)

# ============================================================================
# Fisher/GGN RHMC (for comparison)
# ============================================================================

println("\n" * "="^60)
println("FISHER/GGN RHMC (COMPARISON)")
println("="^60)

println("\nSetting up Fisher/GGN RHMC sampler...")
fisher_metric = BorderedBlockTridiagonalRiemannianMetric(
    ssm, ys, D, P, K, prior_mean, prior_var, obs_indices
)
fisher_hamiltonian = Hamiltonian(fisher_metric, ℓπ)
fisher_integrator = AdaptiveGeneralizedLeapfrog(initial_ϵ; max_iters=7)
fisher_kernel = HMCKernel(
    Trajectory{MultinomialTS}(fisher_integrator, GeneralisedNoUTurn())
)
fisher_adaptor = StepSizeAdaptor(0.8, fisher_integrator)
fisher_sampler = HMCSampler(fisher_kernel, fisher_metric, fisher_adaptor)

println("Running Fisher/GGN RHMC (N_samples=$N_samples, N_adapt=$N_adapt)...")
fisher_chains = AbstractMCMC.sample(
    model,
    fisher_sampler,
    N_samples;
    n_adapts=N_adapt,
    initial_params=initial_θ,
    verbose=false,
    progress=true,
);
fisher_samples = [s.z.θ for s in fisher_chains];

# Extract parameter samples
fisher_μ_samples = [exp(s[K * D + 1]) for s in fisher_samples]

# Compute ESS
fisher_samples_arr = Array{Float64}(undef, N_samples, 1, total_dim)
for i in 1:N_samples
    for j in 1:total_dim
        fisher_samples_arr[i, 1, j] = fisher_samples[i][j]
    end
end
fisher_ess = ess(fisher_samples_arr) ./ N_samples

println("\n=== Fisher/GGN RHMC State ESS Statistics ===")
fisher_state_ess = fisher_ess[1:(D * K)]
println("Minimum State ESS: ", minimum(fisher_state_ess))
println("Median State ESS: ", median(fisher_state_ess))
println("Mean State ESS: ", mean(fisher_state_ess))

println("\n=== Fisher/GGN RHMC Parameter ESS ===")
fisher_param_ess = fisher_ess[(D * K + 1):end]
println("log(μ) ESS: ", fisher_param_ess[1])

println("\n=== Fisher/GGN RHMC Parameter Posterior Summary ===")
fisher_μ_post = fisher_μ_samples[(burn_in + 1):end]
println(
    "μ: true=$(μ_true), posterior mean=$(mean(fisher_μ_post)), std=$(std(fisher_μ_post))"
)

# ============================================================================
# Comparison Summary
# ============================================================================

println("\n" * "="^60)
println("COMPARISON SUMMARY")
println("="^60)

println("\n--- State ESS ---")
println(
    "Kleppe (Observed Hessian): min=$(round(minimum(kleppe_state_ess), digits=4)), median=$(round(median(kleppe_state_ess), digits=4)), mean=$(round(mean(kleppe_state_ess), digits=4))",
)
println(
    "Fisher/GGN:               min=$(round(minimum(fisher_state_ess), digits=4)), median=$(round(median(fisher_state_ess), digits=4)), mean=$(round(mean(fisher_state_ess), digits=4))",
)

println("\n--- Parameter ESS ---")
println("Kleppe (Observed Hessian): log(μ)=$(round(kleppe_param_ess[1], digits=4))")
println("Fisher/GGN:               log(μ)=$(round(fisher_param_ess[1], digits=4))")

println("\n--- Parameter Posterior ---")
println(
    "Kleppe (Observed Hessian): μ mean=$(round(mean(kleppe_μ_post), digits=4)), std=$(round(std(kleppe_μ_post), digits=4))",
)
println(
    "Fisher/GGN:               μ mean=$(round(mean(fisher_μ_post), digits=4)), std=$(round(std(fisher_μ_post), digits=4))",
)

# Print total number of steps for each method
total_kleppe_steps = sum(s.stat.n_steps for s in kleppe_chains)
total_fisher_steps = sum(s.stat.n_steps for s in fisher_chains)

# ESSes weighted by total number of steps
weighted_kleppe_state_ess = kleppe_state_ess .* N_samples ./ total_kleppe_steps
weighted_fisher_state_ess = fisher_state_ess .* N_samples ./ total_fisher_steps

println("\n--- Weighted State ESS (by total steps) ---")
println(
    "Kleppe (Observed Hessian): min=$(round(minimum(weighted_kleppe_state_ess), digits=6)), median=$(round(median(weighted_kleppe_state_ess), digits=6)), mean=$(round(mean(weighted_kleppe_state_ess), digits=6))",
)
println(
    "Fisher/GGN:               min=$(round(minimum(weighted_fisher_state_ess), digits=6)), median=$(round(median(weighted_fisher_state_ess), digits =6)), mean=$(round(mean(weighted_fisher_state_ess), digits=6))",
)


# ============================================================================
# Plots
# ============================================================================

# Trace plot comparison for parameter
p_μ_compare = plot(;
    title="μ trace comparison",
    xlabel="Sample",
    ylabel="μ",
    legend=:topright,
    size=(800, 400),
)
plot!(
    p_μ_compare,
    kleppe_μ_samples;
    label="Kleppe (Obs. Hessian)",
    lw=1,
    color=:blue,
    alpha=0.7,
)
plot!(p_μ_compare, fisher_μ_samples; label="Fisher/GGN", lw=1, color=:green, alpha=0.7)
hline!(p_μ_compare, [μ_true]; label="True", lw=2, color=:red)

display(p_μ_compare)

# Histogram comparison of posterior parameter (post burn-in)
p_μ_hist = plot(; layout=(1, 2), size=(1000, 400))

histogram!(
    p_μ_hist[1],
    kleppe_μ_post;
    title="Kleppe (Obs. Hessian)",
    xlabel="μ",
    ylabel="Count",
    label="",
    bins=20,
    color=:blue,
    alpha=0.7,
)
vline!(p_μ_hist[1], [μ_true]; label="True", lw=2, color=:red)

histogram!(
    p_μ_hist[2],
    fisher_μ_post;
    title="Fisher/GGN",
    xlabel="μ",
    ylabel="Count",
    label="",
    bins=20,
    color=:green,
    alpha=0.7,
)
vline!(p_μ_hist[2], [μ_true]; label="True", lw=2, color=:red)

display(p_μ_hist)

# Thin plot samples for trajectory comparison
n_plot_samples = min(50, length(kleppe_μ_post))
plot_idxs = round.(Int, LinRange(burn_in + 1, N_samples, n_plot_samples))

p_traj = plot(; layout=(1, 2), size=(1200, 500))

# Kleppe trajectories
for i in plot_idxs
    s = kleppe_samples[i]
    plot!(
        p_traj[1],
        [s[1 + (k - 1) * D] for k in 1:K],
        [s[2 + (k - 1) * D] for k in 1:K];
        label="",
        lw=1,
        alpha=0.1,
        color=:blue,
    )
end
plot!(
    p_traj[1],
    [z[1] for z in zs_true],
    [z[2] for z in zs_true];
    label="Truth",
    lw=2,
    color=:black,
    title="Kleppe (Obs. Hessian)",
    xlabel="u",
    ylabel="v",
)

# Fisher/GGN trajectories
for i in plot_idxs
    s = fisher_samples[i]
    plot!(
        p_traj[2],
        [s[1 + (k - 1) * D] for k in 1:K],
        [s[2 + (k - 1) * D] for k in 1:K];
        label="",
        lw=1,
        alpha=0.1,
        color=:green,
    )
end
plot!(
    p_traj[2],
    [z[1] for z in zs_true],
    [z[2] for z in zs_true];
    label="Truth",
    lw=2,
    color=:black,
    title="Fisher/GGN",
    xlabel="u",
    ylabel="v",
)

display(p_traj)

println("\nDone!")
