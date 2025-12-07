"""
Compatibility layer for using AdvancedHMC.jl with RiemannianSSMs block tridiagonal computations.

This file demonstrates how to connect the fast block tridiagonal metric tensor computations
to AdvancedHMC's Riemannian HMC implementation.
"""

using Distributions
using LinearAlgebra
using Random
using StaticArrays

using RiemannianSSMs

# Load AdvancedHMC (should already be configured in the research environment to use local installation)
using AdvancedHMC

# Import what we need to extend
import AdvancedHMC: ∂H∂θ, ∂H∂r, neg_energy, rand_momentum, phasepoint
import AdvancedHMC: DualValue, PhasePoint, Hamiltonian, AbstractMetric
import AdvancedHMC: AbstractRiemannianMetric

# ============================================================================
# Custom Block Tridiagonal Riemannian Metric
# ============================================================================

"""
A Riemannian metric for state space models with block tridiagonal structure.

This metric uses the optimized block tridiagonal operations from RiemannianSSMs
to efficiently compute the metric tensor G(θ) and its derivatives.

# Fields
- `size`: Tuple indicating the dimension of the parameter space
- `ssm`: The state space model containing dynamics and observation models
- `ys`: The observed data
- `D`: Block dimension
- `K`: Number of time steps
"""
struct BlockTridiagonalRiemannianMetric{T,A,SSM,YS} <: AbstractRiemannianMetric
    size::A
    ssm::SSM
    ys::YS
    D::Int
    K::Int
end

function BlockTridiagonalRiemannianMetric(ssm, ys, D::Int, K::Int)
    size = (D * K,)
    return BlockTridiagonalRiemannianMetric{Float64,typeof(size),typeof(ssm),typeof(ys)}(
        size, ssm, ys, D, K
    )
end

Base.size(m::BlockTridiagonalRiemannianMetric) = m.size
Base.size(m::BlockTridiagonalRiemannianMetric, dim::Int) = m.size[dim]
Base.eltype(::BlockTridiagonalRiemannianMetric{T}) where {T} = T

function Base.show(io::IO, m::BlockTridiagonalRiemannianMetric)
    return print(io, "BlockTridiagonalRiemannianMetric(K=$(m.K), D=$(m.D))")
end

# ============================================================================
# Conversion utilities between Vector and BlockVector
# ============================================================================

"""Convert a standard vector to a BlockVector."""
function to_block_vector(θ::AbstractVector{T}, D::Int) where {T}
    blocks = reinterpret(SVector{D,T}, θ)
    return BlockVector{T,D}(blocks)
end

"""Convert a BlockVector to a standard vector."""
function from_block_vector(θ_blocks::BlockVector{T,D}) where {T,D}
    K = length(θ_blocks.blocks)
    θ = Vector{T}(undef, K * D)
    for k in 1:K
        θ[(k - 1) * D .+ (1:D)] .= θ_blocks.blocks[k]
    end
    return θ
end

# ============================================================================
# Metric tensor and derivative computation
# ============================================================================

"""Compute the metric tensor G(θ) as a block tridiagonal matrix."""
function compute_G(metric::BlockTridiagonalRiemannianMetric, θ::AbstractVector)
    θ_blocks = to_block_vector(θ, metric.D)
    return calc_G(θ_blocks, metric.ssm)
end

"""
Compute derivatives of G with respect to θ.

Returns a 3D array where ∂G∂θ[:,:,i] is the derivative of G w.r.t. θ[i].
This is converted from the block tridiagonal format to dense format for AdvancedHMC.
"""
function compute_dGdθ(metric::BlockTridiagonalRiemannianMetric, θ::AbstractVector)
    throw(NotImplementedError("Function not tested"))

    θ_blocks = to_block_vector(θ, metric.D)
    dGs_block = calc_dGs(θ_blocks, metric.ssm)  # Vector of block tridiagonal matrices

    # Convert to dense 3D array format expected by AdvancedHMC
    n = length(θ)
    ∂G∂θ = zeros(n, n, n)

    for idx in 1:n
        k = div(idx - 1, metric.D) + 1  # which block
        d = mod(idx - 1, metric.D) + 1   # which dimension within block

        # The derivative lives in dGs_block[d]
        # We need to convert this block tridiagonal matrix to dense
        G_deriv = dGs_block[d]

        for i in 1:n, j in 1:n
            ∂G∂θ[i, j, idx] = G_deriv[i, j]
        end
    end

    return ∂G∂θ
end

# ============================================================================
# Log probability and gradient computation
# ============================================================================

"""
Compute log probability and its gradient for the state space model.

This wraps the log-likelihood computation from RiemannianSSMs.
"""
function log_prob_and_grad(metric::BlockTridiagonalRiemannianMetric, θ::AbstractVector)
    θ_blocks = to_block_vector(θ, metric.D)

    # Compute log probability
    ll = calc_ll(θ_blocks, metric.ys, metric.ssm)

    # Compute gradient
    grad = calc_ll_grad(θ_blocks, metric.ys, metric.ssm)

    return ll, grad
end

# ============================================================================
# AdvancedHMC interface implementations
# ============================================================================

"""
Sample momentum from the Riemannian metric at position θ.

Uses the efficient block tridiagonal Cholesky decomposition.
"""
function rand_momentum(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    metric::BlockTridiagonalRiemannianMetric{T},
    kinetic,
    θ::AbstractVecOrMat,
) where {T}
    # Compute G at current position
    G = compute_G(metric, θ)

    # Efficient Cholesky decomposition using block structure
    G_chol = cholesky(G)

    # Sample white noise
    U = BlockVector{Float64,metric.D}([
        SVector{metric.D,Float64}(randn(rng, metric.D)) for k in 1:(metric.K)
    ])

    # Transform to get momentum: r ~ N(0, G^{-1})
    # We want G^{-1/2} * U, which is (L^{-T}) * U where G = LL^T
    # TODO: this is messy
    r_blocks = G_chol.U.data' * U
    r = from_block_vector(r_blocks)

    return r
end

"""
Compute negative kinetic energy: -log p(r|θ) - normalizing_constant.

For Riemannian HMC: K(r,θ) = (1/2) r^T G(θ)^{-1} r + (1/2) log|G(θ)| + const
So: -K(r,θ) = -(1/2) r^T G(θ)^{-1} r - (1/2) log|G(θ)| - const
"""
function neg_energy(
    h::Hamiltonian{<:BlockTridiagonalRiemannianMetric}, r::T, θ::T
) where {T<:AbstractVecOrMat}
    metric = h.metric

    # Compute G at current position
    G = compute_G(metric, θ)

    # Efficient Cholesky decomposition
    G_chol = cholesky(G)

    # # Compute log determinant
    # log_det = logdet(G_chol)

    # # Compute quadratic form r^T G^{-1} r efficiently
    # r_blocks = to_block_vector(r, metric.D)
    # w = G_chol.factors' \ r_blocks  # Whiten by solving L^T w = r
    # pTG_inv_p = sum(abs2, w)

    # # Negative kinetic energy (including normalizing constant)
    # D = length(θ)
    # return -(D * log(2π) + log_det + pTG_inv_p) / 2

    logZ = 1 / 2 * (length(θ) * log(2π) + logdet(G_chol))
    w = G_chol.factors' \ to_block_vector(r, metric.D)
    rTG_inv_r = sum(abs2, w)
    return -logZ - rTG_inv_r / 2
end

# # Negative kinetic energy
# #! Eq (13) of Girolami & Calderhead (2011)
# function neg_energy(
#     h::Hamiltonian{<:DenseRiemannianMetric}, r::T, θ::T
# ) where {T<:AbstractVecOrMat}
#     G = h.metric.map(h.metric.G(θ))
#     D = size(G, 1)
#     # Need to consider the normalizing term as it is no longer same for different θs
#     logZ = 1 / 2 * (D * log(2π) + logdet(G)) # it will be user's responsibility to make sure G is SPD and logdet(G) is defined
#     mul!(h.metric._temp, inv(G), r)
#     return -logZ - dot(r, h.metric._temp) / 2
# end

"""
Compute ∂H/∂θ = -∂ℓπ/∂θ + ∂K/∂θ where K is the kinetic energy.

For Riemannian HMC, this includes correction terms from the position-dependent metric.
"""
function ∂H∂θ(
    h::Hamiltonian{<:BlockTridiagonalRiemannianMetric{T}},
    θ::AbstractVecOrMat{T},
    r::AbstractVecOrMat{T},
) where {T}
    metric = h.metric

    # Get log probability and gradient
    ℓπ, ∂ℓπ∂θ = log_prob_and_grad(metric, θ)

    # Convert to block format for efficient operations
    θ_blocks = to_block_vector(θ, metric.D)
    r_blocks = to_block_vector(r, metric.D)

    # Compute G and its derivatives
    G = calc_G(θ_blocks, metric.ssm)
    G_chol = cholesky(G)
    G_inv = block_tridiag_selected_inv(G)
    dGs = calc_dGs(θ_blocks, metric.ssm)

    # Compute G^{-1} * r using efficient block operations
    G_inv_r = G_chol \ r_blocks

    # Compute gradient contributions
    grad = Vector{T}(undef, length(θ))

    for idx in 1:length(θ)
        k = div(idx - 1, metric.D) + 1  # which block
        d = mod(idx - 1, metric.D) + 1   # which dimension within block

        v = -∂ℓπ∂θ.blocks[k][d]

        # Trace term: (1/2) tr(G^{-1} ∂G/∂θ_i)
        v -= -0.5 * sum(G_inv.diag_blocks[k] .* dGs[d].diag_blocks[k])
        if k < metric.K
            v -= -sum(G_inv.super_blocks[k] .* dGs[d].super_blocks[k])
        end

        # Quadratic form term: -(1/2) (G^{-1}r)^T (∂G/∂θ_i) (G^{-1}r)
        v -= 0.5 * G_inv_r.blocks[k]' * dGs[d].diag_blocks[k] * G_inv_r.blocks[k]
        if k < metric.K
            v -= G_inv_r.blocks[k]' * dGs[d].super_blocks[k] * G_inv_r.blocks[k + 1]
        end

        grad[idx] = v
    end

    H = RiemannianSSMs._calc_hamiltonian(θ_blocks, r_blocks, ℓπ, G)

    return DualValue(H, grad)
end

"""
Compute ∂H/∂r = G(θ)^{-1} r.

This is the "generalized velocity" in Riemannian HMC.
"""
function ∂H∂r(
    h::Hamiltonian{<:BlockTridiagonalRiemannianMetric},
    θ::AbstractVecOrMat,
    r::AbstractVecOrMat,
)
    metric = h.metric

    # Compute G at current position
    G = compute_G(metric, θ)

    # Efficient solve using block structure
    G_chol = try
        cholesky(G)
    catch e
        @warn "Cholesky decomposition failed in ∂H∂r with θ=$(θ)"
        rethrow(e)
    end
    r_blocks = to_block_vector(r, metric.D)
    result_blocks = G_chol \ r_blocks

    return from_block_vector(result_blocks)
end

"""
Specialized phasepoint constructor for BlockTridiagonalRiemannianMetric.

Ensures we pass θ to both ∂H∂θ and ∂H∂r since they depend on position.
"""
function phasepoint(
    h::Hamiltonian{<:BlockTridiagonalRiemannianMetric},
    θ::T,
    r::T;
    ℓπ=∂H∂θ(h, θ, r),
    ℓκ=DualValue(neg_energy(h, r, θ), ∂H∂r(h, θ, r)),
) where {T<:AbstractVecOrMat}
    return PhasePoint(θ, r, ℓπ, ℓκ)
end

# ============================================================================
# Helper struct for state space model
# ============================================================================

struct SSM{PT,DY,OM}
    prior::PT
    dyn::DY
    sensor::OM
end

# ============================================================================
# Ground truth simulation
# ============================================================================

SEED = 4
rng = MersenneTwister(SEED)

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
# σ1 = 0.5
# σ2 = 0.5
σ1 = 2.0
σ2 = 2.0
obs = TwoLandmarkMeasurementModel(a1, b1, a2, b2, σ1, σ2)

# State space model
ssm = SSM(prior, dyn, obs)

# Simulate data
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

using Plots

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

# Add sensor locations
scatter!(p1, [a1, a2], [b1, b2]; label="Sensors", color=:red, ms=8, marker=:star5)

display(p1)

# ============================================================================
# RHMC Test Run
# ============================================================================

D = 4  # state dimension
metric = BlockTridiagonalRiemannianMetric(ssm, ys, D, K)

using LogDensityProblems, AbstractMCMC

struct LogTargetDensity{M,V}
    dim::Int
    ys::V
    ssm::M
end
function LogDensityProblems.logdensity(p::LogTargetDensity, θ)
    θ = BlockVector{Float64,4}(reinterpret(SVector{4,Float64}, θ))
    return RiemannianSSMs.calc_ll(θ, p.ys, p.ssm)
end
function LogDensityProblems.logdensity_and_gradient(p::LogTargetDensity, θ)
    θ = BlockVector{Float64,4}(reinterpret(SVector{4,Float64}, θ))
    ll = RiemannianSSMs.calc_ll(θ, p.ys, p.ssm)
    grad = RiemannianSSMs.calc_ll_grad(θ, p.ys, p.ssm)
    return ll, reinterpret(Float64, grad.blocks)
end
LogDensityProblems.dimension(p::LogTargetDensity) = p.dim
function LogDensityProblems.capabilities(::Type{<:LogTargetDensity})
    return LogDensityProblems.LogDensityOrder{1}()
end

ℓπ = LogTargetDensity(4 * K, ys, ssm)
model = AdvancedHMC.LogDensityModel(ℓπ)

initial_θ = from_block_vector(zs_true_block)

hamiltonian = Hamiltonian(metric, ℓπ)
initial_ϵ = 0.01
integrator = GeneralizedLeapfrog(initial_ϵ, 7)
# kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
# No NUTS on multinomial
kernel = HMCKernel(Trajectory{EndPointTS}(integrator, FixedNSteps(5)))
adaptor = StepSizeAdaptor(0.8, integrator)
# adaptor = AdvancedHMC.NoAdaptation()
rhmc = HMCSampler(kernel, metric, adaptor)

N_samples = 5000
N_adapt = 1000

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

# Plot true and inferred trajectories

# Thin plot samples
n_plot_samples = 200
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
        alpha=0.1,
        color=:blue,
    )
end

display(p1)
