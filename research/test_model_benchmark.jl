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
K = 100
δt = 1.0

# Prior
μ0 = @SVector [0.0, 0.0, 0.0, 0.0]
Σ0 = Diagonal(@SVector([0.5, 0.5, 0.1, 0.1]))
prior = MvNormal(μ0, Σ0)

# Dynamics
α = 0.01
β = 0.1
γ = 0.005
σ_p = 0.1
σ_v = 0.5
dyn = VariableRestoringForceDynamics(α, β, γ, σ_p, σ_v, δt)

# Observations
a1, b1 = -1.0, -3.0
a2, b2 = 5.0, -1.5
σ1 = 0.5
σ2 = 0.5
obs = TwoLandmarkMeasurementModel(a1, b1, a2, b2, σ1, σ2)

ssm = SSM(prior, dyn, obs)

function simulate(rng::AbstractRNG, ssm, K::Int)
    zs = Vector{SVector{4,Float64}}(undef, K)
    ys = Vector{SVector{2,Float64}}(undef, K)

    for k in 1:K
        if k == 1
            z = SVector{4,Float64}(rand(rng, ssm.prior))
        else
            z = f(ssm.dyn, zs[k - 1]) + rand(rng, MvNormal(zeros(4), calc_Q(ssm.dyn)))
        end
        zs[k] = z

        y = h(ssm.sensor, z) + rand(rng, MvNormal(zeros(2), calc_R(ssm.sensor)))
        ys[k] = y
    end

    return zs, ys
end

zs_true, ys = simulate(rng, ssm, K)
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
# LogDensityProblems Model
# ============================================================================

struct LogTargetDensity{D,M,V}
    dim::Int
    ssm::M
    ys::V
end

function LogTargetDensity(K::Int, D::Int, ssm, ys)
    dim = K * D
    return LogTargetDensity{D,typeof(ssm),typeof(ys)}(dim, ssm, ys)
end

function LogDensityProblems.logdensity(p::LogTargetDensity{D}, θ) where {D}
    θ = to_block_vector(θ, Val(D))
    return calc_ll(θ, p.ys, p.ssm)
end
function LogDensityProblems.logdensity_and_gradient(p::LogTargetDensity{D}, θ) where {D}
    θ_blocks = to_block_vector(θ, Val(D))
    ll = calc_ll(θ_blocks, p.ys, p.ssm)
    grad = calc_ll_grad(θ_blocks, p.ys, p.ssm)
    return ll, from_block_vector(grad)
end
LogDensityProblems.dimension(p::LogTargetDensity) = p.dim
function LogDensityProblems.capabilities(::Type{<:LogTargetDensity})
    return LogDensityProblems.LogDensityOrder{1}()
end

ℓπ = LogTargetDensity(K, D, ssm, ys)
model = AdvancedHMC.LogDensityModel(ℓπ)
initial_θ = from_block_vector(zs_true_block)

# ============================================================================
# RHMC Sampling
# ============================================================================

metric = BlockTridiagonalRiemannianMetric(ssm, ys, D, K)
hamiltonian = Hamiltonian(metric, ℓπ)
initial_ϵ = 0.05
integrator = GeneralizedLeapfrog(initial_ϵ, 7)
kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
adaptor = StepSizeAdaptor(0.95, integrator)
rhmc = HMCSampler(kernel, metric, adaptor)

N_samples = 5000
N_adapt = 2000

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

# Shape into (draws, [chains[, parameters...]])
rhmc_samples = Array{Float64}(undef, N_samples, 1, 4 * K)
for i in 1:N_samples
    for k in 1:K
        for d in 1:4
            rhmc_samples[i, 1, 4 * (k - 1) + d] = samples[i][4 * (k - 1) + d]
        end
    end
end
rhmc_ess = ess(rhmc_samples) ./ N_samples

println("Minimum RHMC ESS: ", minimum(rhmc_ess))
println("Median RHMC ESS: ", median(rhmc_ess))
println("Mean RHMC ESS: ", mean(rhmc_ess))

# ============================================================================
# Plots
# ============================================================================

# Thin plot samples
n_plot_samples = 500
plot_idxs = round.(Int, LinRange(1, N_samples, n_plot_samples))
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

# Trace plots
ps = []
for d in 1:4
    push!(ps, plot(; xlabel="Sample", ylabel="Dimension $d", legend=false))
    plot!(ps[end], [samples[i][4 * (K - 1) + d] for i in 1:N_samples]; lw=1, color=:blue)
    hline!(ps[end], [zs_true_block.blocks[K][d]]; lw=2, color=:black, label="True")
end
display(plot(ps...; layout=(4, 1), size=(600, 800)))

# ============================================================================
# HMC Sampling
# ============================================================================

hmc = NUTS(0.8)
chains = AbstractMCMC.sample(
    model,
    hmc,
    N_samples;
    n_adapts=N_adapt,
    initial_params=initial_θ,
    verbose=false,
    progress=true,
);
samples = [s.z.θ for s in chains];

hmc_samples = Array{Float64}(undef, N_samples, 1, 4 * K)
for i in 1:N_samples
    for k in 1:K
        for d in 1:4
            hmc_samples[i, 1, 4 * (k - 1) + d] = samples[i][4 * (k - 1) + d]
        end
    end
end
hmc_ess = ess(hmc_samples) ./ N_samples

println("Minimum HMC ESS: ", minimum(hmc_ess))
println("Median HMC ESS: ", median(hmc_ess))
println("Mean HMC ESS: ", mean(hmc_ess))
