"""Integration with AdvancedHMC.jl's implementation of Riemannian HMC."""

using AdvancedHMC
using Distributions

import AdvancedHMC: DualValue

export BlockTridiagonalRiemannianMetric, calc_G, calc_dGs, calc_ll, calc_ll_grad, SSM

struct BlockTridiagonalRiemannianMetric{T,D,SSM,YT} <: AdvancedHMC.AbstractRiemannianMetric
    ssm::SSM
    ys::YT
    K::Int
end

function BlockTridiagonalRiemannianMetric(ssm, ys, D::Int, K::Int)
    return BlockTridiagonalRiemannianMetric{Float64,D,typeof(ssm),typeof(ys)}(ssm, ys, K)
end

Base.size(m::BlockTridiagonalRiemannianMetric{T,D}) where {T,D} = (m.K * D,)
Base.size(m::BlockTridiagonalRiemannianMetric{T,D}, ::Int) where {T,D} = m.K * D
Base.eltype(::BlockTridiagonalRiemannianMetric{T}) where {T} = T

function Base.show(io::IO, m::BlockTridiagonalRiemannianMetric{T,D}) where {T,D}
    return print(io, "BlockTridiagonalRiemannianMetric(K=$(m.K), D=$D)")
end

function calc_G(
    metric::BlockTridiagonalRiemannianMetric{T,D}, θ::AbstractVector
) where {T,D}
    θ_blocks = to_block_vector(θ, Val(D))
    return calc_G(θ_blocks, metric.ssm)
end

function calc_G_fast(
    metric::BlockTridiagonalRiemannianMetric{T,D}, θ::AbstractVector
) where {T,D}
    θ_blocks = to_block_vector(θ, Val(D))
    return calc_G_fast(θ_blocks, metric.ssm)
end

function AdvancedHMC.rand_momentum(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    metric::BlockTridiagonalRiemannianMetric{T,D},
    kinetic,
    θ::AbstractVecOrMat,
) where {T,D}
    G = calc_G(metric, θ)
    G_chol = cholesky!(G)

    # Sample Z ~ N(0, I) as a block vector
    Z = BlockVector{Float64,D}([(@SVector randn(rng, D)) for k in 1:(metric.K)])

    # Transform to r ~ N(0, G)
    # TODO: clean up how the U factor is accessed
    r_blocks = G_chol.U.data' * Z
    r = from_block_vector(r_blocks)

    return r
end

"""
Compute negative kinetic energy: -log p(r|θ) - normalizing_constant.

For Riemannian HMC: K(r,θ) = (1/2) r^T G(θ)^{-1} r + (1/2) log|G(θ)| + const
So: -K(r,θ) = -(1/2) r^T G(θ)^{-1} r - (1/2) log|G(θ)| - const
"""
# TODO: combine with ∂H∂r to avoid recomputing G
function AdvancedHMC.neg_energy(
    h::Hamiltonian{<:BlockTridiagonalRiemannianMetric{T,D}}, r::V, θ::V
) where {V<:AbstractVecOrMat,T,D}
    metric = h.metric

    G = calc_G(metric, θ)
    G_chol = cholesky!(G)

    logZ = 1 / 2 * (length(θ) * log(2π) + logdet(G_chol))
    w = G_chol.factors' \ to_block_vector(r, Val(D))
    rTG_inv_r = sum(abs2, w)

    return -logZ - rTG_inv_r / 2
end

function AdvancedHMC.∂H∂θ(
    h::Hamiltonian{<:BlockTridiagonalRiemannianMetric{T,D}},
    θ::AbstractVecOrMat{T},
    r::AbstractVecOrMat{T},
) where {T,D}
    return AdvancedHMC.∂H∂θ_cache(h, θ, r)
end

function AdvancedHMC.∂H∂θ_cache(
    h::Hamiltonian{<:BlockTridiagonalRiemannianMetric{T,D}},
    θ::AbstractVecOrMat{T},
    r::AbstractVecOrMat{T};
    return_cache=false,
    cache=nothing,
) where {T,D}
    metric = h.metric

    # Terms that only depend on θ can be cached
    if isnothing(cache)
        θ_blocks = to_block_vector(θ, Val(D))
        ℓπ = calc_ll(θ_blocks, metric.ys, metric.ssm)
        ∂ℓπ∂θ = calc_ll_grad(θ_blocks, metric.ys, metric.ssm)

        G = calc_G(θ_blocks, metric.ssm)
        G_inv = block_tridiag_selected_inv(G)  # sparse inverse of G with only tridiagonal blocks
        G_chol = cholesky!(G)  # warning: overwrites G
        dGs = calc_dGs(θ_blocks, metric.ssm)

        # Pre-compute gradient contributions that are independent of r and can be cached
        # This is the gradient from -∂ℓπ/∂θ and the trace term (1/2) tr(G^{-1} ∂G/∂θ_i)
        cached_grad = Vector{T}(undef, length(θ))

        @inbounds for idx in 1:length(θ)
            k = div(idx - 1, D) + 1  # which block
            d = mod(idx - 1, D) + 1   # which dimension within block

            v = -∂ℓπ∂θ.blocks[k][d]

            # Trace term: (1/2) tr(G^{-1} ∂G/∂θ_i)
            v -= -0.5 * sum(G_inv.diag_blocks[k] .* dGs[d].diag_blocks[k])
            if k < metric.K
                v -= -sum(G_inv.super_blocks[k] .* dGs[d].super_blocks[k])
            end

            cached_grad[idx] = v
        end
    else
        ℓπ, cached_grad, G_chol, dGs = cache
    end

    r_blocks = to_block_vector(r, Val(D))

    # Compute G^{-1} * r
    G_inv_r = G_chol \ r_blocks

    # Compute gradient contributions starting from cached contributions
    grad = copy(cached_grad)

    @inbounds for idx in 1:length(θ)
        k = div(idx - 1, D) + 1  # which block
        d = mod(idx - 1, D) + 1   # which dimension within block

        # Quadratic form term: -(1/2) (G^{-1}r)^T (∂G/∂θ_i) (G^{-1}r)
        v = 0.5 * G_inv_r.blocks[k]' * dGs[d].diag_blocks[k] * G_inv_r.blocks[k]
        if k < metric.K
            v += G_inv_r.blocks[k]' * dGs[d].super_blocks[k] * G_inv_r.blocks[k + 1]
        end

        grad[idx] -= v
    end

    dv = DualValue(ℓπ, grad)
    return return_cache ? (dv, (; ℓπ, cached_grad, G_chol, dGs)) : dv
end

function AdvancedHMC.∂H∂r(
    h::Hamiltonian{<:BlockTridiagonalRiemannianMetric{T,D}},
    θ::AbstractVecOrMat,
    r::AbstractVecOrMat,
) where {T,D}
    metric = h.metric

    G = calc_G(metric, θ)
    G_chol = cholesky!(G)

    r_blocks = to_block_vector(r, Val(D))
    result_blocks = G_chol \ r_blocks

    return from_block_vector(result_blocks)
end

#####################
#### DERIVATIVES ####
#####################

struct SSM{PT,DY,OM}
    prior::PT
    dyn::DY
    sensor::OM
end

function calc_ll(zs::BlockVector{T,D}, ys, ssm) where {T,D}
    K = length(ys)

    # Prior
    ll = logpdf(ssm.prior, zs.blocks[1])

    # Dynamics
    @inbounds for k in 2:K
        ll += logpdf(MvNormal(f(ssm.dyn, zs.blocks[k - 1]), calc_Q(ssm.dyn)), zs.blocks[k])
    end

    # Likelihood
    @inbounds for k in 1:K
        ll += logpdf(MvNormal(h(ssm.sensor, zs.blocks[k]), calc_R(ssm.sensor)), ys[k])
    end

    return ll
end

function calc_ll_grad(zs::BlockVector{T,D}, ys, ssm) where {T,D}
    K = length(zs.blocks)
    grads = Vector{SVector{D,T}}(undef, K)

    # Incoming dynamics
    @inbounds grads[1] = -inv(ssm.prior.Σ) * (zs.blocks[1] - ssm.prior.μ)
    @inbounds for k in 2:K
        grads[k] = -calc_Qinv(ssm.dyn) * (zs.blocks[k] - f(ssm.dyn, zs.blocks[k - 1]))
    end

    # Observation term
    @inbounds for k in 1:K
        Jh = calc_Jh(ssm.sensor, zs.blocks[k])
        R_inv = calc_Rinv(ssm.sensor)
        grads[k] += Jh' * R_inv * (ys[k] - h(ssm.sensor, zs.blocks[k]))
    end

    # Transition term
    @inbounds for k in 2:K
        Jf = calc_Jf(ssm.dyn, zs.blocks[k - 1])
        grads[k - 1] +=
            Jf' * calc_Qinv(ssm.dyn) * (zs.blocks[k] - f(ssm.dyn, zs.blocks[k - 1]))
    end

    return BlockVector{T,D}(grads)
end

function calc_G(zs::BlockVector{T,D}, ssm) where {T,D}
    K = length(zs.blocks)
    # TODO: again, allocation is expensive. Should do in-place
    G = SymPSDBlockTridiag{T,D}(
        Vector{SMatrix{D,D,T,D^2}}(undef, K), Vector{SMatrix{D,D,T,D^2}}(undef, K - 1)
    )

    # Compute diagonal
    @inbounds for k in 1:K
        # Base term
        Λ = k == 1 ? inv(ssm.prior.Σ) : calc_Qinv(ssm.dyn)
        # Observation term
        Jh = calc_Jh(ssm.sensor, zs.blocks[k])
        Λ += Jh' * calc_Rinv(ssm.sensor) * Jh
        # Dynamics term
        if k < K
            Jf = calc_Jf(ssm.dyn, zs.blocks[k])
            Λ += Jf' * calc_Qinv(ssm.dyn) * Jf
        end

        # Force to be positive definite
        Λ = (Λ + Λ') / 2 + 1e-6 * I

        G.diag_blocks[k] = Λ
    end

    # Compute off-diagonal term
    @inbounds for k in 1:(K - 1)
        Jf = calc_Jf(ssm.dyn, zs.blocks[k])
        Λ = -Jf' * calc_Qinv(ssm.dyn)
        G.super_blocks[k] = Λ
    end

    return G
end

function calc_dGs(zs::BlockVector{T,D}, ssm) where {T,D}
    dGs = Vector{SymPSDBlockTridiag{T,D,D^2,Vector{SMatrix{D,D,T,D^2}}}}(undef, D)
    K = length(zs.blocks)

    for d in 1:D
        # TODO: this allocation is quite expensive, maybe do in-place?
        diagonal_hessians = Vector{SMatrix{D,D,T,D^2}}(undef, K)
        off_diagonal_hessians = Vector{SMatrix{D,D,T,D^2}}(undef, K - 1)

        @inbounds for k in 1:K
            Jf = calc_Jf(ssm.dyn, zs.blocks[k])
            Jh = calc_Jh(ssm.sensor, zs.blocks[k])
            Hfs = calc_Hfs(ssm.dyn, zs.blocks[k])
            Hhs = calc_Hhs(ssm.sensor, zs.blocks[k])  # latter pair are zeros but include for generality
            Q_inv = calc_Qinv(ssm.dyn)
            R_inv = calc_Rinv(ssm.sensor)

            # Cache matrices
            Mh = R_inv * Jh
            Mf = Q_inv * Jf

            diagonal_hessians[k] = Hhs[d]' * Mh + Mh' * Hhs[d]
            if k < K
                diagonal_hessians[k] += Hfs[d]' * Mf + Mf' * Hfs[d]
            end
        end

        @inbounds for k in 1:(K - 1)
            Hfs = calc_Hfs(ssm.dyn, zs.blocks[k])
            Q_inv = calc_Qinv(ssm.dyn)

            off_diagonal_hessians[k] = -Hfs[d]' * Q_inv
        end

        dGs[d] = SymPSDBlockTridiag{T,D}(diagonal_hessians, off_diagonal_hessians)
    end

    return dGs
end
