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
# Ground truth simulation
# ============================================================================

SEED = 4
rng = MersenneTwister(SEED)

D = 4
P = 2
K = 100
δt = 1.0

# Prior for initial state
μ0 = @SVector [0.0, 0.0, 0.0, 0.0]
Σ0 = Diagonal(@SVector([0.5, 0.5, 0.1, 0.1]))
prior = MvNormal(μ0, Σ0)

# True parameter values
α_true = 0.01
β_true = 0.1
θ_true = @SVector [log(α_true), log(β_true)]

# Dynamics (γ is fixed, α and β will be inferred)
γ = 0.005
σ_p = 0.1
σ_v = 0.5
dyn = VariableRestoringForceDynamicsParam(γ, σ_p, σ_v, δt)

# Observations
a1, b1 = -1.0, -3.0
a2, b2 = 5.0, -1.5
σ1 = 0.5
σ2 = 0.5
obs = TwoLandmarkMeasurementModelParam(a1, b1, a2, b2, σ1, σ2)

ssm = SSMParam(prior, dyn, obs)

# Parameter prior (diffuse Gaussian on log scale)
prior_mean = @SVector [log(0.01), log(0.1)]  # Centered near true values for testing
prior_var = @SVector [4.0, 4.0]  # Large variance for diffuse prior

function simulate(rng::AbstractRNG, ssm, θ, K::Int)
    zs = Vector{SVector{4,Float64}}(undef, K)
    ys = Vector{SVector{2,Float64}}(undef, K)

    for k in 1:K
        if k == 1
            z = SVector{4,Float64}(rand(rng, ssm.prior))
        else
            z =
                f_param(ssm.dyn, zs[k - 1], θ) +
                rand(rng, MvNormal(zeros(4), calc_Q_param(ssm.dyn)))
        end
        zs[k] = z

        y =
            h_param(ssm.sensor, z, θ) +
            rand(rng, MvNormal(zeros(2), calc_R_param(ssm.sensor)))
        ys[k] = y
    end

    return zs, ys
end

zs_true, ys = simulate(rng, ssm, θ_true, K)
zs_true_block = BlockVector{Float64,4}(zs_true)

p1 = plot(;
    title="Position",
    xlabel="x",
    ylabel="y",
    legend=:topright,
    size=(800, 600),
    aspect_ratio=1,
)
plot!(
    p1,
    [z[1] for z in zs_true_block.blocks],
    [z[2] for z in zs_true_block.blocks];
    label="Truth",
    lw=2,
    color=:black,
)
scatter!(p1, [a1, a2], [b1, b2]; label="Sensors", color=:red, ms=8, marker=:star5)

# ============================================================================
# LogDensityProblems Model (Joint state + parameter inference)
# ============================================================================

struct LogTargetDensityParam{D,P,M,V,PM,PV}
    dim::Int
    ssm::M
    ys::V
    prior_mean::PM
    prior_prec::PV
end

function LogTargetDensityParam(K::Int, D::Int, P::Int, ssm, ys, prior_mean, prior_var)
    dim = K * D + P
    prior_prec = SVector{P,Float64}(1 ./ prior_var)
    return LogTargetDensityParam{
        D,P,typeof(ssm),typeof(ys),typeof(prior_mean),typeof(prior_prec)
    }(
        dim, ssm, ys, prior_mean, prior_prec
    )
end

function LogDensityProblems.logdensity(p::LogTargetDensityParam{D,P}, θ) where {D,P}
    θ_blocks = to_bordered_block_vector(θ, Val(D), Val(P))
    return calc_ll_param(θ_blocks, p.ys, p.ssm, p.prior_mean, p.prior_prec)
end

function LogDensityProblems.logdensity_and_gradient(
    p::LogTargetDensityParam{D,P}, θ
) where {D,P}
    θ_blocks = to_bordered_block_vector(θ, Val(D), Val(P))
    ll = calc_ll_param(θ_blocks, p.ys, p.ssm, p.prior_mean, p.prior_prec)
    grad = calc_ll_grad_param(θ_blocks, p.ys, p.ssm, p.prior_mean, p.prior_prec)
    return ll, from_bordered_block_vector(grad)
end

LogDensityProblems.dimension(p::LogTargetDensityParam) = p.dim

function LogDensityProblems.capabilities(::Type{<:LogTargetDensityParam})
    return LogDensityProblems.LogDensityOrder{1}()
end

ℓπ = LogTargetDensityParam(K, D, P, ssm, ys, prior_mean, prior_var)
model = AdvancedHMC.LogDensityModel(ℓπ)

# Initial state: true states + true parameters (for debugging, start near truth)
initial_θ = vcat(from_block_vector(zs_true_block), collect(θ_true))

# ============================================================================
# RHMC Sampling (Joint state + parameter)
# ============================================================================

metric = BorderedBlockTridiagonalRiemannianMetric(ssm, ys, D, P, K, prior_mean, prior_var)
hamiltonian = Hamiltonian(metric, ℓπ)
initial_ϵ = 0.05
integrator = GeneralizedLeapfrog(initial_ϵ, 7)
integrator = AdaptiveGeneralizedLeapfrog(initial_ϵ; max_iters=7)
# integrator = AdaptiveImplicitMidpoint(initial_ϵ; max_iters=8)
kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
adaptor = StepSizeAdaptor(0.9, integrator)
rhmc = HMCSampler(kernel, metric, adaptor)

N_samples = 5000
N_adapt = 2000

# @profview AbstractMCMC.sample(
#     model, rhmc, 200; n_adapts=100, initial_params=initial_θ, verbose=false, progress=true
# );

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
α_samples = [exp(s[K * D + 1]) for s in samples]
β_samples = [exp(s[K * D + 2]) for s in samples]

# Compute ESS for all dimensions
total_dim = D * K + P
rhmc_samples = Array{Float64}(undef, N_samples, 1, total_dim)
for i in 1:N_samples
    for j in 1:total_dim
        rhmc_samples[i, 1, j] = samples[i][j]
    end
end
rhmc_ess = ess(rhmc_samples) ./ N_samples

println("\n=== State ESS Statistics ===")
state_ess = rhmc_ess[1:(D * K)]
println("Minimum State ESS: ", minimum(state_ess))
println("Median State ESS: ", median(state_ess))
println("Mean State ESS: ", mean(state_ess))

println("\n=== Parameter ESS ===")
param_ess = rhmc_ess[(D * K + 1):end]
println("log(α) ESS: ", param_ess[1])
println("log(β) ESS: ", param_ess[2])

println("\n=== Parameter Posterior Summary ===")
burn_in = N_adapt
α_post = α_samples[(burn_in + 1):end]
β_post = β_samples[(burn_in + 1):end]

println("α: true=$(α_true), posterior mean=$(mean(α_post)), std=$(std(α_post))")
println("β: true=$(β_true), posterior mean=$(mean(β_post)), std=$(std(β_post))")

# ============================================================================
# Plots
# ============================================================================

# Trace plots for parameters
p_α = plot(; title="α trace", xlabel="Sample", ylabel="α", legend=:topright)
plot!(p_α, α_samples; label="Samples", lw=1, color=:blue, alpha=0.5)
hline!(p_α, [α_true]; label="True", lw=2, color=:red)

p_β = plot(; title="β trace", xlabel="Sample", ylabel="β", legend=:topright)
plot!(p_β, β_samples; label="Samples", lw=1, color=:blue, alpha=0.5)
hline!(p_β, [β_true]; label="True", lw=2, color=:red)

display(plot(p_α, p_β; layout=(2, 1), size=(800, 600)))

# Histograms of posterior parameters
p_α_hist = histogram(
    α_post;
    title="α posterior",
    xlabel="α",
    ylabel="Count",
    label="",
    bins=50,
    color=:blue,
    alpha=0.7,
)
vline!(p_α_hist, [α_true]; label="True", lw=2, color=:red)

p_β_hist = histogram(
    β_post;
    title="β posterior",
    xlabel="β",
    ylabel="Count",
    label="",
    bins=50,
    color=:blue,
    alpha=0.7,
)
vline!(p_β_hist, [β_true]; label="True", lw=2, color=:red)

display(plot(p_α_hist, p_β_hist; layout=(2, 1), size=(800, 600)))

# Thin plot samples for trajectory
n_plot_samples = 500
plot_idxs = round.(Int, LinRange(burn_in + 1, N_samples, n_plot_samples))
plot_samples = samples[plot_idxs]

for i in 1:n_plot_samples
    s = plot_samples[i]
    plot!(
        p1,
        [s[1 + (k - 1) * D] for k in 1:K],
        [s[2 + (k - 1) * D] for k in 1:K];
        label="",
        lw=1,
        alpha=0.05,
        color=:blue,
    )
end

display(p1)

# State trace plots (last time step)
ps = []
for d in 1:4
    push!(ps, plot(; xlabel="Sample", ylabel="Dimension $d", legend=false))
    plot!(ps[end], [samples[i][4 * (K - 1) + d] for i in 1:N_samples]; lw=1, color=:blue)
    hline!(ps[end], [zs_true_block.blocks[K][d]]; lw=2, color=:black, label="True")
end
display(plot(ps...; layout=(4, 1), size=(600, 800)))

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
α_samples_hmc = [exp(s[K * D + 1]) for s in hmc_samples_raw]
β_samples_hmc = [exp(s[K * D + 2]) for s in hmc_samples_raw]

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
println("log(α) ESS: ", hmc_param_ess[1])
println("log(β) ESS: ", hmc_param_ess[2])

println("\n=== HMC Parameter Posterior Summary ===")
α_post_hmc = α_samples_hmc[(burn_in + 1):end]
β_post_hmc = β_samples_hmc[(burn_in + 1):end]

println("α: true=$(α_true), posterior mean=$(mean(α_post_hmc)), std=$(std(α_post_hmc))")
println("β: true=$(β_true), posterior mean=$(mean(β_post_hmc)), std=$(std(β_post_hmc))")

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
println(
    "RHMC: log(α)=$(round(param_ess[1], digits=4)), log(β)=$(round(param_ess[2], digits=4))"
)
println(
    "HMC:  log(α)=$(round(hmc_param_ess[1], digits=4)), log(β)=$(round(hmc_param_ess[2], digits=4))",
)

# Comparison trace plots for parameters
p_α_compare = plot(;
    title="α trace comparison", xlabel="Sample", ylabel="α", legend=:topright
)
plot!(p_α_compare, α_samples; label="RHMC", lw=1, color=:blue, alpha=0.5)
plot!(p_α_compare, α_samples_hmc; label="HMC", lw=1, color=:green, alpha=0.5)
hline!(p_α_compare, [α_true]; label="True", lw=2, color=:red)

p_β_compare = plot(;
    title="β trace comparison", xlabel="Sample", ylabel="β", legend=:topright
)
plot!(p_β_compare, β_samples; label="RHMC", lw=1, color=:blue, alpha=0.5)
plot!(p_β_compare, β_samples_hmc; label="HMC", lw=1, color=:green, alpha=0.5)
hline!(p_β_compare, [β_true]; label="True", lw=2, color=:red)

display(plot(p_α_compare, p_β_compare; layout=(2, 1), size=(800, 600)))

println("\nDone!")
