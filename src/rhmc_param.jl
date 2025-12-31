"""Integration with AdvancedHMC.jl's implementation of Riemannian HMC for joint state+parameter inference."""

using AdvancedHMC
using Distributions

import AdvancedHMC: DualValue

export BorderedBlockTridiagonalRiemannianMetric, SSMParam
export calc_G_param, calc_dGs_state, calc_dGs_param, calc_ll_param, calc_ll_grad_param

struct BorderedBlockTridiagonalRiemannianMetric{T,D,P,SSM,YT,OBS} <:
       AdvancedHMC.AbstractRiemannianMetric
    ssm::SSM
    ys::YT
    obs_indices::OBS           # Vector{Int} or Nothing - which latent state indices have observations
    K::Int                     # Total number of latent states
    prior_mean::SVector{P,T}   # prior mean for log-parameters
    prior_prec::SVector{P,T}   # prior precision (1/variance) for diagonal prior
end

# Backward-compatible constructor (observations at every time step)
function BorderedBlockTridiagonalRiemannianMetric(
    ssm, ys, D::Int, P::Int, K::Int, prior_mean, prior_var
)
    prior_prec = 1 ./ prior_var
    T = eltype(prior_mean)
    return BorderedBlockTridiagonalRiemannianMetric{T,D,P,typeof(ssm),typeof(ys),Nothing}(
        ssm, ys, nothing, K, prior_mean, prior_prec
    )
end

# New constructor with sparse observations
function BorderedBlockTridiagonalRiemannianMetric(
    ssm, ys, D::Int, P::Int, K::Int, prior_mean, prior_var, obs_indices::Vector{Int}
)
    prior_prec = 1 ./ prior_var
    T = eltype(prior_mean)
    return BorderedBlockTridiagonalRiemannianMetric{
        T,D,P,typeof(ssm),typeof(ys),Vector{Int}
    }(
        ssm, ys, obs_indices, K, prior_mean, prior_prec
    )
end

Base.size(m::BorderedBlockTridiagonalRiemannianMetric{T,D,P}) where {T,D,P} = (m.K * D + P,)
function Base.size(m::BorderedBlockTridiagonalRiemannianMetric{T,D,P}, ::Int) where {T,D,P}
    return m.K * D + P
end
Base.eltype(::BorderedBlockTridiagonalRiemannianMetric{T}) where {T} = T

function Base.show(io::IO, m::BorderedBlockTridiagonalRiemannianMetric{T,D,P}) where {T,D,P}
    return print(io, "BorderedBlockTridiagonalRiemannianMetric(K=$(m.K), D=$D, P=$P)")
end

function calc_G(
    metric::BorderedBlockTridiagonalRiemannianMetric{T,D,P}, z::AbstractVector; λ=1e-7
) where {T,D,P}
    z_blocks = to_bordered_block_vector(z, Val(D), Val(P))
    return calc_G_param(
        z_blocks, metric.ssm, metric.prior_prec; λ=λ, obs_indices=metric.obs_indices
    )
end

function AdvancedHMC.rand_momentum(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    metric::BorderedBlockTridiagonalRiemannianMetric{T,D,P},
    kinetic,
    z::AbstractVecOrMat,
) where {T,D,P}
    G = calc_G(metric, z)
    G_chol = cholesky(G)

    # Sample Z ~ N(0, I) as a bordered block vector
    state_blocks = [(@SVector randn(rng, D)) for k in 1:(metric.K)]
    param_block = @SVector randn(rng, P)
    Z = BorderedBlockVector{Float64,D,P}(state_blocks, param_block)

    # Transform to r ~ N(0, G)
    r_blocks = G_chol.factors' * Z
    r = from_bordered_block_vector(r_blocks)

    return r
end

"""
Compute negative kinetic energy: -log p(r|z) - normalizing_constant.
"""
function AdvancedHMC.neg_energy(
    h::Hamiltonian{<:BorderedBlockTridiagonalRiemannianMetric{T,D,P}}, r::V, z::V
) where {V<:AbstractVecOrMat,T,D,P}
    metric = h.metric

    G = calc_G(metric, z)
    G_chol = cholesky(G)

    logZ = 1 / 2 * (length(z) * log(2π) + logdet(G_chol))
    w = G_chol.factors' \ to_bordered_block_vector(r, Val(D), Val(P))
    rTG_inv_r = sum(abs2, from_bordered_block_vector(w))

    return -logZ - rTG_inv_r / 2
end

function AdvancedHMC.∂H∂θ(
    h::Hamiltonian{<:BorderedBlockTridiagonalRiemannianMetric{T,D,P}},
    z::AbstractVecOrMat{T},
    r::AbstractVecOrMat{T},
) where {T,D,P}
    return AdvancedHMC.∂H∂θ_cache(h, z, r)
end

function AdvancedHMC.∂H∂θ_cache(
    h::Hamiltonian{<:BorderedBlockTridiagonalRiemannianMetric{T,D,P}},
    z::AbstractVecOrMat{T},
    r::AbstractVecOrMat{T};
    return_cache=false,
    cache=nothing,
) where {T,D,P}
    metric = h.metric

    # Terms that only depend on z can be cached
    if isnothing(cache)
        z_blocks = to_bordered_block_vector(z, Val(D), Val(P))
        ℓπ = calc_ll_param(
            z_blocks,
            metric.ys,
            metric.ssm,
            metric.prior_mean,
            metric.prior_prec;
            obs_indices=metric.obs_indices,
        )
        ∂ℓπ∂z = calc_ll_grad_param(
            z_blocks,
            metric.ys,
            metric.ssm,
            metric.prior_mean,
            metric.prior_prec;
            obs_indices=metric.obs_indices,
        )

        G = calc_G_param(
            z_blocks, metric.ssm, metric.prior_prec; obs_indices=metric.obs_indices
        )
        G_inv_parts = bordered_block_selected_inv(G)
        G_chol = cholesky(G)
        dGs_state = calc_dGs_state(z_blocks, metric.ssm; obs_indices=metric.obs_indices)
        dGs_param = calc_dGs_param(z_blocks, metric.ssm; obs_indices=metric.obs_indices)

        # Pre-compute gradient contributions that are independent of r
        # This is the gradient from -∂ℓπ/∂z and the trace term (1/2) tr(G^{-1} ∂G/∂z_i)
        cached_grad = Vector{T}(undef, length(z))

        # State gradient components
        @inbounds for k in 1:(metric.K)
            for d in 1:D
                idx = (k - 1) * D + d
                v = -∂ℓπ∂z.state_blocks[k][d]

                # Trace term: (1/2) tr(G^{-1} ∂G/∂z_k^{(d)})
                dG = dGs_state[k, d]
                v -= -0.5 * tr_product(G_inv_parts, dG)

                cached_grad[idx] = v
            end
        end

        # Parameter gradient components
        @inbounds for p in 1:P
            idx = metric.K * D + p
            v = -∂ℓπ∂z.param_block[p]

            # Trace term: (1/2) tr(G^{-1} ∂G/∂θ^{(p)})
            dG = dGs_param[p]
            v -= -0.5 * tr_product(G_inv_parts, dG)

            cached_grad[idx] = v
        end
    else
        ℓπ, cached_grad, G_chol, dGs_state, dGs_param = cache
    end

    r_blocks = to_bordered_block_vector(r, Val(D), Val(P))

    # Compute G^{-1} * r
    G_inv_r = G_chol \ r_blocks

    # Compute gradient contributions starting from cached contributions
    grad = copy(cached_grad)

    # State gradient components (quadratic form term)
    @inbounds for k in 1:(metric.K)
        for d in 1:D
            idx = (k - 1) * D + d
            dG = dGs_state[k, d]

            # Quadratic form term: -(1/2) (G^{-1}r)^T (∂G/∂z_i) (G^{-1}r)
            v = 0.5 * quad_form(G_inv_r, dG)
            grad[idx] -= v
        end
    end

    # Parameter gradient components (quadratic form term)
    @inbounds for p in 1:P
        idx = metric.K * D + p
        dG = dGs_param[p]

        # Quadratic form term
        v = 0.5 * quad_form(G_inv_r, dG)
        grad[idx] -= v
    end

    dv = DualValue(ℓπ, grad)
    return return_cache ? (dv, (; ℓπ, cached_grad, G_chol, dGs_state, dGs_param)) : dv
end

function AdvancedHMC.∂H∂r(
    h::Hamiltonian{<:BorderedBlockTridiagonalRiemannianMetric{T,D,P}},
    z::AbstractVecOrMat,
    r::AbstractVecOrMat,
) where {T,D,P}
    metric = h.metric

    G = calc_G(metric, z)
    G_chol = cholesky(G)

    r_blocks = to_bordered_block_vector(r, Val(D), Val(P))
    result_blocks = G_chol \ r_blocks

    return from_bordered_block_vector(result_blocks)
end

#####################
#### SSM STRUCT #####
#####################

struct SSMParam{PT,DY,OM}
    prior::PT
    dyn::DY
    sensor::OM
end

#####################
#### DERIVATIVES ####
#####################

function calc_ll_param(
    z::BorderedBlockVector{T,D,P},
    ys,
    ssm,
    prior_mean::SVector{P,T},
    prior_prec::SVector{P,T};
    obs_indices=nothing,
) where {T,D,P}
    K = length(z.state_blocks)
    θ = z.param_block

    # Parameter prior (Gaussian with diagonal covariance)
    ll = -0.5 * sum(prior_prec .* (θ - prior_mean) .^ 2)

    # State prior
    ll += logpdf(ssm.prior, z.state_blocks[1])

    # Dynamics
    @inbounds for k in 2:K
        μ = f_param(ssm.dyn, z.state_blocks[k - 1], θ)
        ll += logpdf(MvNormal(μ, calc_Q_param(ssm.dyn)), z.state_blocks[k])
    end

    # Likelihood (only at observed time steps)
    if isnothing(obs_indices)
        # Backward compatible: observations at every time step
        @inbounds for k in 1:K
            μ = h_param(ssm.sensor, z.state_blocks[k], θ)
            ll += logpdf(MvNormal(μ, calc_R_param(ssm.sensor)), ys[k])
        end
    else
        # Sparse observations
        @inbounds for (obs_idx, k) in enumerate(obs_indices)
            μ = h_param(ssm.sensor, z.state_blocks[k], θ)
            ll += logpdf(MvNormal(μ, calc_R_param(ssm.sensor)), ys[obs_idx])
        end
    end

    return ll
end

function calc_ll_grad_param(
    z::BorderedBlockVector{T,D,P},
    ys,
    ssm,
    prior_mean::SVector{P,T},
    prior_prec::SVector{P,T};
    obs_indices=nothing,
) where {T,D,P}
    K = length(z.state_blocks)
    θ = z.param_block

    state_grads = Vector{SVector{D,T}}(undef, K)
    param_grad = -prior_prec .* (θ - prior_mean)  # Parameter prior gradient

    Q_inv = calc_Qinv_param(ssm.dyn)
    R_inv = calc_Rinv_param(ssm.sensor)

    # Incoming dynamics (state prior for k=1, dynamics for k>1)
    @inbounds state_grads[1] = -inv(ssm.prior.Σ) * (z.state_blocks[1] - ssm.prior.μ)
    @inbounds for k in 2:K
        μ = f_param(ssm.dyn, z.state_blocks[k - 1], θ)
        state_grads[k] = -Q_inv * (z.state_blocks[k] - μ)
    end

    # Observation term (only at observed time steps)
    if isnothing(obs_indices)
        # Backward compatible: observations at every time step
        @inbounds for k in 1:K
            Jh = calc_Jh_param(ssm.sensor, z.state_blocks[k], θ)
            μ = h_param(ssm.sensor, z.state_blocks[k], θ)
            state_grads[k] += Jh' * R_inv * (ys[k] - μ)

            # Parameter gradient from observation
            h_θ = calc_h_θ(ssm.sensor, z.state_blocks[k], θ)
            param_grad += h_θ' * R_inv * (ys[k] - μ)
        end
    else
        # Sparse observations
        @inbounds for (obs_idx, k) in enumerate(obs_indices)
            Jh = calc_Jh_param(ssm.sensor, z.state_blocks[k], θ)
            μ = h_param(ssm.sensor, z.state_blocks[k], θ)
            state_grads[k] += Jh' * R_inv * (ys[obs_idx] - μ)

            # Parameter gradient from observation
            h_θ = calc_h_θ(ssm.sensor, z.state_blocks[k], θ)
            param_grad += h_θ' * R_inv * (ys[obs_idx] - μ)
        end
    end

    # Transition term
    @inbounds for k in 2:K
        Jf = calc_Jf_param(ssm.dyn, z.state_blocks[k - 1], θ)
        μ = f_param(ssm.dyn, z.state_blocks[k - 1], θ)
        residual = z.state_blocks[k] - μ
        state_grads[k - 1] += Jf' * Q_inv * residual

        # Parameter gradient from dynamics
        f_θ = calc_f_θ(ssm.dyn, z.state_blocks[k - 1], θ)
        param_grad += f_θ' * Q_inv * residual
    end

    return BorderedBlockVector{T,D,P}(state_grads, param_grad)
end

function calc_G_param(
    z::BorderedBlockVector{T,D,P},
    ssm,
    prior_prec::SVector{P,T};
    λ=1e-7,
    obs_indices=nothing,
) where {T,D,P}
    K = length(z.state_blocks)
    θ = z.param_block

    diag_blocks = Vector{SMatrix{D,D,T,D^2}}(undef, K)
    super_blocks = Vector{SMatrix{D,D,T,D^2}}(undef, K - 1)
    border_blocks = Vector{SMatrix{D,P,T,D * P}}(undef, K)
    corner_block = zeros(SMatrix{P,P,T,P^2})

    Q_inv = calc_Qinv_param(ssm.dyn)
    R_inv = calc_Rinv_param(ssm.sensor)

    # Build set of observed indices for fast lookup
    obs_set = isnothing(obs_indices) ? nothing : Set(obs_indices)

    # Compute diagonal blocks and border blocks
    @inbounds for k in 1:K
        # Base term (prior for k=1, dynamics for k>1)
        Λ = k == 1 ? SMatrix{D,D,T}(inv(ssm.prior.Σ)) : SMatrix{D,D,T}(Q_inv)

        # Check if this time step has an observation
        has_obs = isnothing(obs_set) || (k in obs_set)

        # Observation term (only if observed)
        if has_obs
            Jh = calc_Jh_param(ssm.sensor, z.state_blocks[k], θ)
            Λ += Jh' * R_inv * Jh
        end

        # Dynamics term (contribution from k+1)
        if k < K
            Jf = calc_Jf_param(ssm.dyn, z.state_blocks[k], θ)
            Λ += Jf' * Q_inv * Jf
        end

        # Force to be positive definite
        Λ = (Λ + Λ') / 2 + λ * I
        diag_blocks[k] = Λ

        # Border block G_{k,θ}
        border_k = zeros(SMatrix{D,P,T})

        # Observation contribution to border (only if observed)
        if has_obs
            Jh = calc_Jh_param(ssm.sensor, z.state_blocks[k], θ)
            h_θ = calc_h_θ(ssm.sensor, z.state_blocks[k], θ)
            border_k += Jh' * R_inv * h_θ
        end

        if k > 1
            # -Q⁻¹ f_{θ,k}
            f_θ_k = calc_f_θ(ssm.dyn, z.state_blocks[k - 1], θ)
            border_k += -SMatrix{D,P,T}(Q_inv * f_θ_k)
        end

        if k < K
            # F_{k+1}' Q⁻¹ f_{θ,k+1}
            Jf_k1 = calc_Jf_param(ssm.dyn, z.state_blocks[k], θ)
            f_θ_k1 = calc_f_θ(ssm.dyn, z.state_blocks[k], θ)
            border_k += Jf_k1' * Q_inv * f_θ_k1
        end

        border_blocks[k] = border_k

        # Corner block contributions from observations (only if observed)
        if has_obs
            h_θ_k = calc_h_θ(ssm.sensor, z.state_blocks[k], θ)
            corner_block += h_θ_k' * R_inv * h_θ_k
        end
    end

    # Dynamics contributions to corner block
    @inbounds for k in 2:K
        f_θ_k = calc_f_θ(ssm.dyn, z.state_blocks[k - 1], θ)
        corner_block += f_θ_k' * Q_inv * f_θ_k
    end

    # Add prior precision to corner block
    corner_block += Diagonal(prior_prec)
    corner_block = (corner_block + corner_block') / 2 + λ * I

    # Compute off-diagonal blocks
    @inbounds for k in 1:(K - 1)
        Jf = calc_Jf_param(ssm.dyn, z.state_blocks[k], θ)
        super_blocks[k] = -Jf' * Q_inv
    end

    return SymPSDBorderedBlockTridiag{T,D,P}(
        diag_blocks, super_blocks, border_blocks, corner_block
    )
end

"""
Compute derivatives of G w.r.t. each state coordinate.
Returns a K×D matrix of BlockSparseStateDerivative.
"""
function calc_dGs_state(
    z::BorderedBlockVector{T,D,P}, ssm; obs_indices=nothing
) where {T,D,P}
    K = length(z.state_blocks)
    θ = z.param_block

    dGs = Matrix{BlockSparseStateDerivative{T,D,P,D^2,P^2,D * P}}(undef, K, D)

    Q_inv = calc_Qinv_param(ssm.dyn)
    R_inv = calc_Rinv_param(ssm.sensor)

    # Build set of observed indices for fast lookup
    obs_set = isnothing(obs_indices) ? nothing : Set(obs_indices)

    @inbounds for k in 1:K
        # Check if this time step has an observation
        has_obs = isnothing(obs_set) || (k in obs_set)

        Jf_k1 = k < K ? calc_Jf_param(ssm.dyn, z.state_blocks[k], θ) : zeros(SMatrix{D,D,T})
        Hfs_k1 = if k < K
            calc_Hfs_param(ssm.dyn, z.state_blocks[k], θ)
        else
            ntuple(_ -> zeros(SMatrix{D,D,T}), D)
        end

        f_θ_k1 = k < K ? calc_f_θ(ssm.dyn, z.state_blocks[k], θ) : zeros(SMatrix{D,P,T})
        Hf_θs_k1 = if k < K
            calc_Hf_θs(ssm.dyn, z.state_blocks[k], θ)
        else
            ntuple(_ -> zeros(SMatrix{D,P,T}), D)
        end

        # Only compute observation-related derivatives if observed
        if has_obs
            Jh_k = calc_Jh_param(ssm.sensor, z.state_blocks[k], θ)
            Hhs_k = calc_Hhs_param(ssm.sensor, z.state_blocks[k], θ)
            h_θ_k = calc_h_θ(ssm.sensor, z.state_blocks[k], θ)
            Hh_θs_k = calc_Hh_θs(ssm.sensor, z.state_blocks[k], θ)
            Mh = R_inv * Jh_k
        else
            Hhs_k = ntuple(_ -> zeros(SMatrix{1,D,T}), D)
            h_θ_k = zeros(SMatrix{1,P,T})
            Hh_θs_k = ntuple(_ -> zeros(SMatrix{1,P,T}), D)
            Mh = zeros(SMatrix{1,D,T})
        end

        Mf = Q_inv * Jf_k1

        for d in 1:D
            # Diagonal block ∂G_{kk}/∂x_k^{(d)}
            diag_block = zeros(SMatrix{D,D,T})
            if k < K
                diag_block += Hfs_k1[d]' * Mf + Mf' * Hfs_k1[d]
            end
            if has_obs
                diag_block += Hhs_k[d]' * Mh + Mh' * Hhs_k[d]
            end

            # Off-diagonal block ∂G_{k,k+1}/∂x_k^{(d)}
            super_block = k < K ? -Hfs_k1[d]' * Q_inv : zeros(SMatrix{D,D,T})

            # Border block ∂G_{k,θ}/∂x_k^{(d)}
            border_block_k = zeros(SMatrix{D,P,T})
            if k < K
                border_block_k += Hfs_k1[d]' * Q_inv * f_θ_k1 + Jf_k1' * Q_inv * Hf_θs_k1[d]
            end
            if has_obs
                Jh_k = calc_Jh_param(ssm.sensor, z.state_blocks[k], θ)
                border_block_k += Hhs_k[d]' * R_inv * h_θ_k + Jh_k' * R_inv * Hh_θs_k[d]
            end

            # Border block ∂G_{k+1,θ}/∂x_k^{(d)}
            border_block_k1 = k < K ? -Q_inv * Hf_θs_k1[d] : zeros(SMatrix{D,P,T})

            # Corner block ∂G_{θθ}/∂x_k^{(d)}
            corner_block = zeros(SMatrix{P,P,T})
            if k < K
                corner_block +=
                    Hf_θs_k1[d]' * Q_inv * f_θ_k1 + f_θ_k1' * Q_inv * Hf_θs_k1[d]
            end
            if has_obs
                corner_block += Hh_θs_k[d]' * R_inv * h_θ_k + h_θ_k' * R_inv * Hh_θs_k[d]
            end

            dGs[k, d] = BlockSparseStateDerivative{T,D,P}(
                k, d, diag_block, super_block, border_block_k, border_block_k1, corner_block
            )
        end
    end

    return dGs
end

"""
Compute derivatives of G w.r.t. each parameter coordinate.
Returns a Vector of P SymPSDBorderedBlockTridiag matrices.
"""
function calc_dGs_param(
    z::BorderedBlockVector{T,D,P}, ssm; obs_indices=nothing
) where {T,D,P}
    K = length(z.state_blocks)
    θ = z.param_block

    dGs = Vector{
        SymPSDBorderedBlockTridiag{
            T,D,P,D^2,P^2,D * P,Vector{SMatrix{D,D,T,D^2}},Vector{SMatrix{D,P,T,D * P}}
        },
    }(
        undef, P
    )

    Q_inv = calc_Qinv_param(ssm.dyn)
    R_inv = calc_Rinv_param(ssm.sensor)

    # Build set of observed indices for fast lookup
    obs_set = isnothing(obs_indices) ? nothing : Set(obs_indices)

    for p in 1:P
        diag_blocks = Vector{SMatrix{D,D,T,D^2}}(undef, K)
        super_blocks = Vector{SMatrix{D,D,T,D^2}}(undef, K - 1)
        border_blocks = Vector{SMatrix{D,P,T,D * P}}(undef, K)
        corner_block = zeros(SMatrix{P,P,T,P^2})

        @inbounds for k in 1:K
            # Check if this time step has an observation
            has_obs = isnothing(obs_set) || (k in obs_set)

            Jf_k1 =
                k < K ? calc_Jf_param(ssm.dyn, z.state_blocks[k], θ) : zeros(SMatrix{D,D,T})
            Jf_θ_k1 = if k < K
                calc_Jf_θ(ssm.dyn, z.state_blocks[k], θ)
            else
                ntuple(_ -> zeros(SMatrix{D,D,T}), P)
            end

            f_θ_k1 = k < K ? calc_f_θ(ssm.dyn, z.state_blocks[k], θ) : zeros(SMatrix{D,P,T})
            f_θθ_k1 = if k < K
                calc_f_θθ(ssm.dyn, z.state_blocks[k], θ)
            else
                ntuple(_ -> zeros(SMatrix{D,P,T}), P)
            end

            # Only compute observation-related derivatives if observed
            if has_obs
                Jh_k = calc_Jh_param(ssm.sensor, z.state_blocks[k], θ)
                Jh_θ_k = calc_Jh_θ(ssm.sensor, z.state_blocks[k], θ)
                h_θ_k = calc_h_θ(ssm.sensor, z.state_blocks[k], θ)
                h_θθ_k = calc_h_θθ(ssm.sensor, z.state_blocks[k], θ)
            end

            # Diagonal block ∂G_{kk}/∂θ^{(p)}
            diag_block = zeros(SMatrix{D,D,T})
            if k < K
                diag_block += Jf_θ_k1[p]' * Q_inv * Jf_k1 + Jf_k1' * Q_inv * Jf_θ_k1[p]
            end
            if has_obs
                diag_block += Jh_θ_k[p]' * R_inv * Jh_k + Jh_k' * R_inv * Jh_θ_k[p]
            end
            diag_blocks[k] = diag_block

            # Border block ∂G_{k,θ}/∂θ^{(p)}
            border_block = zeros(SMatrix{D,P,T})
            if k > 1
                f_θ_k = calc_f_θ(ssm.dyn, z.state_blocks[k - 1], θ)
                f_θθ_k = calc_f_θθ(ssm.dyn, z.state_blocks[k - 1], θ)
                border_block += -Q_inv * f_θθ_k[p]
            end
            if k < K
                border_block += Jf_θ_k1[p]' * Q_inv * f_θ_k1 + Jf_k1' * Q_inv * f_θθ_k1[p]
            end
            if has_obs
                border_block += Jh_θ_k[p]' * R_inv * h_θ_k + Jh_k' * R_inv * h_θθ_k[p]
            end
            border_blocks[k] = border_block

            # Corner block contributions from this k (only if observed)
            if has_obs
                corner_block += h_θθ_k[p]' * R_inv * h_θ_k + h_θ_k' * R_inv * h_θθ_k[p]
            end
        end

        # Off-diagonal blocks ∂G_{k,k-1}/∂θ^{(p)}
        @inbounds for k in 1:(K - 1)
            Jf_θ_k1 = calc_Jf_θ(ssm.dyn, z.state_blocks[k], θ)
            super_blocks[k] = -Jf_θ_k1[p]' * Q_inv
        end

        # Corner block contributions from dynamics
        @inbounds for k in 2:K
            f_θ_k = calc_f_θ(ssm.dyn, z.state_blocks[k - 1], θ)
            f_θθ_k = calc_f_θθ(ssm.dyn, z.state_blocks[k - 1], θ)
            corner_block += f_θθ_k[p]' * Q_inv * f_θ_k + f_θ_k' * Q_inv * f_θθ_k[p]
        end

        dGs[p] = SymPSDBorderedBlockTridiag{T,D,P}(
            diag_blocks, super_blocks, border_blocks, corner_block
        )
    end

    return dGs
end
