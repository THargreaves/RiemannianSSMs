"""
RHMC infrastructure adapted for Poisson observations with state-dependent weight matrix.

Key differences from Gaussian case:
- Likelihood uses Poisson distribution instead of MvNormal
- Weight matrix W_t = diag(λ_t) depends on both state and parameters
- Metric derivatives must account for ∂W_t/∂x and ∂W_t/∂θ

Supports both single-channel and multi-channel Poisson observations.
"""

using AdvancedHMC
using Distributions

import AdvancedHMC: DualValue

export PoissonRiemannianMetric
export calc_ll_param_poisson, calc_ll_grad_param_poisson
export calc_G_param_poisson, calc_dGs_state_poisson, calc_dGs_param_poisson

# ============================================================================
# Poisson Riemannian Metric (AdvancedHMC integration)
# ============================================================================

struct PoissonRiemannianMetric{T,D,P,SSM,YT,OBS} <: AdvancedHMC.AbstractRiemannianMetric
    ssm::SSM
    ys::YT
    obs_indices::OBS
    K::Int
    prior_mean::SVector{P,T}
    prior_prec::SVector{P,T}
end

function PoissonRiemannianMetric(
    ssm, ys, D::Int, P::Int, K::Int, prior_mean, prior_var, obs_indices::Vector{Int}
)
    prior_prec = SVector{P,Float64}(1 ./ prior_var)
    T = Float64
    return PoissonRiemannianMetric{T,D,P,typeof(ssm),typeof(ys),Vector{Int}}(
        ssm, ys, obs_indices, K, prior_mean, prior_prec
    )
end

Base.size(m::PoissonRiemannianMetric{T,D,P}) where {T,D,P} = (m.K * D + P,)
Base.size(m::PoissonRiemannianMetric{T,D,P}, ::Int) where {T,D,P} = m.K * D + P
Base.eltype(::PoissonRiemannianMetric{T}) where {T} = T
function Base.show(io::IO, m::PoissonRiemannianMetric)
    return print(io, "PoissonRiemannianMetric with K=$(m.K)")
end

function calc_G_poisson(
    metric::PoissonRiemannianMetric{T,D,P}, z::AbstractVector; λ=1e-7
) where {T,D,P}
    z_blocks = to_bordered_block_vector(z, Val(D), Val(P))
    return calc_G_param_poisson(
        z_blocks, metric.ssm, metric.prior_prec; λ=λ, obs_indices=metric.obs_indices
    )
end

function AdvancedHMC.rand_momentum(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    metric::PoissonRiemannianMetric{T,D,P},
    kinetic,
    z::AbstractVecOrMat,
) where {T,D,P}
    G = calc_G_poisson(metric, z)
    G_chol = cholesky(G)

    state_blocks = [(@SVector randn(rng, D)) for k in 1:(metric.K)]
    param_block = @SVector randn(rng, P)
    Z = BorderedBlockVector{Float64,D,P}(state_blocks, param_block)

    r_blocks = G_chol.factors' * Z
    r = from_bordered_block_vector(r_blocks)

    return r
end

function AdvancedHMC.neg_energy(
    h::Hamiltonian{<:PoissonRiemannianMetric{T,D,P}}, r::V, z::V
) where {V<:AbstractVecOrMat,T,D,P}
    metric = h.metric

    G = calc_G_poisson(metric, z)
    G_chol = cholesky(G)

    logZ = 1 / 2 * (length(z) * log(2π) + logdet(G_chol))
    w = G_chol.factors' \ to_bordered_block_vector(r, Val(D), Val(P))
    rTG_inv_r = sum(abs2, from_bordered_block_vector(w))

    return -logZ - rTG_inv_r / 2
end

function AdvancedHMC.∂H∂θ(
    h::Hamiltonian{<:PoissonRiemannianMetric{T,D,P}},
    z::AbstractVecOrMat{T},
    r::AbstractVecOrMat{T},
) where {T,D,P}
    return AdvancedHMC.∂H∂θ_cache(h, z, r)
end

function AdvancedHMC.∂H∂θ_cache(
    h::Hamiltonian{<:PoissonRiemannianMetric{T,D,P}},
    z::AbstractVecOrMat{T},
    r::AbstractVecOrMat{T};
    return_cache=false,
    cache=nothing,
) where {T,D,P}
    metric = h.metric

    if isnothing(cache)
        z_blocks = to_bordered_block_vector(z, Val(D), Val(P))
        ℓπ = calc_ll_param_poisson(
            z_blocks,
            metric.ys,
            metric.ssm,
            metric.prior_mean,
            metric.prior_prec;
            obs_indices=metric.obs_indices,
        )
        ∂ℓπ∂z = calc_ll_grad_param_poisson(
            z_blocks,
            metric.ys,
            metric.ssm,
            metric.prior_mean,
            metric.prior_prec;
            obs_indices=metric.obs_indices,
        )

        G = calc_G_param_poisson(
            z_blocks, metric.ssm, metric.prior_prec; obs_indices=metric.obs_indices
        )
        G_inv_parts = bordered_block_selected_inv(G)
        G_chol = cholesky(G)
        dGs_state = calc_dGs_state_poisson(
            z_blocks, metric.ssm; obs_indices=metric.obs_indices
        )
        dGs_param = calc_dGs_param_poisson(
            z_blocks, metric.ssm; obs_indices=metric.obs_indices
        )

        cached_grad = Vector{T}(undef, length(z))

        @inbounds for k in 1:(metric.K)
            for d in 1:D
                idx = (k - 1) * D + d
                v = -∂ℓπ∂z.state_blocks[k][d]
                dG = dGs_state[k, d]
                v -= -0.5 * tr_product(G_inv_parts, dG)
                cached_grad[idx] = v
            end
        end

        @inbounds for p in 1:P
            idx = metric.K * D + p
            v = -∂ℓπ∂z.param_block[p]
            dG = dGs_param[p]
            v -= -0.5 * tr_product(G_inv_parts, dG)
            cached_grad[idx] = v
        end
    else
        ℓπ, cached_grad, G_chol, dGs_state, dGs_param = cache
    end

    r_blocks = to_bordered_block_vector(r, Val(D), Val(P))
    G_inv_r = G_chol \ r_blocks

    grad = copy(cached_grad)

    @inbounds for k in 1:(metric.K)
        for d in 1:D
            idx = (k - 1) * D + d
            dG = dGs_state[k, d]
            v = 0.5 * quad_form(G_inv_r, dG)
            grad[idx] -= v
        end
    end

    @inbounds for p in 1:P
        idx = metric.K * D + p
        dG = dGs_param[p]
        v = 0.5 * quad_form(G_inv_r, dG)
        grad[idx] -= v
    end

    dv = DualValue(ℓπ, grad)
    return return_cache ? (dv, (; ℓπ, cached_grad, G_chol, dGs_state, dGs_param)) : dv
end

function AdvancedHMC.∂H∂r(
    h::Hamiltonian{<:PoissonRiemannianMetric{T,D,P}},
    z::AbstractVecOrMat,
    r::AbstractVecOrMat,
) where {T,D,P}
    metric = h.metric
    G = calc_G_poisson(metric, z)
    G_chol = cholesky(G)
    r_blocks = to_bordered_block_vector(r, Val(D), Val(P))
    result_blocks = G_chol \ r_blocks
    return from_bordered_block_vector(result_blocks)
end

# ============================================================================
# Likelihood and Gradient Functions
# ============================================================================

"""
Calculate log-likelihood for Poisson SSM.

Supports both single-channel and multi-channel Poisson observations.
Uses Poisson(λ_t) distribution where λ_t = exp(η_t) and η_t = h_param(sensor, x_t, θ).
"""
function calc_ll_param_poisson(
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

    # Poisson likelihood (only at observed time steps)
    if isnothing(obs_indices)
        @inbounds for k in 1:K
            η = h_param(ssm.sensor, z.state_blocks[k], θ)
            λ = exp.(η)
            y = ys[k]
            for m in eachindex(λ)
                ll += logpdf(Poisson(λ[m]), y[m])
            end
        end
    else
        @inbounds for (obs_idx, k) in enumerate(obs_indices)
            η = h_param(ssm.sensor, z.state_blocks[k], θ)
            λ = exp.(η)
            y = ys[obs_idx]
            for m in eachindex(λ)
                ll += logpdf(Poisson(λ[m]), y[m])
            end
        end
    end

    return ll
end

"""
Calculate gradient of log-likelihood for Poisson SSM.

Supports both single-channel and multi-channel Poisson observations.
Poisson gradient: ∂ℓ/∂η = y - λ (since ∂log P/∂λ = (y-λ)/λ and ∂λ/∂η = λ cancel)
"""
function calc_ll_grad_param_poisson(
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
    param_grad = -prior_prec .* (θ - prior_mean)

    Q_inv = calc_Qinv_param(ssm.dyn)

    # Incoming dynamics (state prior for k=1, dynamics for k>1)
    @inbounds state_grads[1] = -inv(ssm.prior.Σ) * (z.state_blocks[1] - ssm.prior.μ)
    @inbounds for k in 2:K
        μ = f_param(ssm.dyn, z.state_blocks[k - 1], θ)
        state_grads[k] = -Q_inv * (z.state_blocks[k] - μ)
    end

    # Poisson observation term (only at observed time steps)
    if isnothing(obs_indices)
        @inbounds for k in 1:K
            η = h_param(ssm.sensor, z.state_blocks[k], θ)
            λ = exp.(η)
            y = ys[k]
            gradient_term = y - λ  # M-vector

            Jh = calc_Jh_param(ssm.sensor, z.state_blocks[k], θ)  # M×D
            state_grads[k] += Jh' * gradient_term

            h_θ = calc_h_θ(ssm.sensor, z.state_blocks[k], θ)  # M×P
            param_grad += h_θ' * gradient_term
        end
    else
        @inbounds for (obs_idx, k) in enumerate(obs_indices)
            η = h_param(ssm.sensor, z.state_blocks[k], θ)
            λ = exp.(η)
            y = ys[obs_idx]
            gradient_term = y - λ

            Jh = calc_Jh_param(ssm.sensor, z.state_blocks[k], θ)
            state_grads[k] += Jh' * gradient_term

            h_θ = calc_h_θ(ssm.sensor, z.state_blocks[k], θ)
            param_grad += h_θ' * gradient_term
        end
    end

    # Transition term
    @inbounds for k in 2:K
        Jf = calc_Jf_param(ssm.dyn, z.state_blocks[k - 1], θ)
        μ = f_param(ssm.dyn, z.state_blocks[k - 1], θ)
        residual = z.state_blocks[k] - μ
        state_grads[k - 1] += Jf' * Q_inv * residual

        f_θ = calc_f_θ(ssm.dyn, z.state_blocks[k - 1], θ)
        param_grad += f_θ' * Q_inv * residual
    end

    return BorderedBlockVector{T,D,P}(state_grads, param_grad)
end

"""
Calculate Gauss-Newton metric for Poisson SSM.

Supports multi-channel observations with W_t = diag(λ_t).
"""
function calc_G_param_poisson(
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

    obs_set = isnothing(obs_indices) ? nothing : Set(obs_indices)

    @inbounds for k in 1:K
        Λ = k == 1 ? SMatrix{D,D,T}(inv(ssm.prior.Σ)) : SMatrix{D,D,T}(Q_inv)

        has_obs = isnothing(obs_set) || (k in obs_set)

        if has_obs
            W_t = calc_Rinv_param(ssm.sensor, z.state_blocks[k], θ)  # M×M diagonal
            Jh = calc_Jh_param(ssm.sensor, z.state_blocks[k], θ)  # M×D
            Λ += Jh' * W_t * Jh
        end

        if k < K
            Jf = calc_Jf_param(ssm.dyn, z.state_blocks[k], θ)
            Λ += Jf' * Q_inv * Jf
        end

        Λ = (Λ + Λ') / 2 + λ * I
        diag_blocks[k] = Λ

        border_k = zeros(SMatrix{D,P,T})

        if has_obs
            W_t = calc_Rinv_param(ssm.sensor, z.state_blocks[k], θ)
            Jh = calc_Jh_param(ssm.sensor, z.state_blocks[k], θ)
            h_θ = calc_h_θ(ssm.sensor, z.state_blocks[k], θ)
            border_k += Jh' * W_t * h_θ
        end

        if k > 1
            f_θ_k = calc_f_θ(ssm.dyn, z.state_blocks[k - 1], θ)
            border_k += -SMatrix{D,P,T}(Q_inv * f_θ_k)
        end

        if k < K
            Jf_k1 = calc_Jf_param(ssm.dyn, z.state_blocks[k], θ)
            f_θ_k1 = calc_f_θ(ssm.dyn, z.state_blocks[k], θ)
            border_k += Jf_k1' * Q_inv * f_θ_k1
        end

        border_blocks[k] = border_k

        if has_obs
            W_t = calc_Rinv_param(ssm.sensor, z.state_blocks[k], θ)
            h_θ_k = calc_h_θ(ssm.sensor, z.state_blocks[k], θ)
            corner_block += h_θ_k' * W_t * h_θ_k
        end
    end

    @inbounds for k in 2:K
        f_θ_k = calc_f_θ(ssm.dyn, z.state_blocks[k - 1], θ)
        corner_block += f_θ_k' * Q_inv * f_θ_k
    end

    corner_block += Diagonal(prior_prec)
    corner_block = (corner_block + corner_block') / 2 + λ * I

    @inbounds for k in 1:(K - 1)
        Jf = calc_Jf_param(ssm.dyn, z.state_blocks[k], θ)
        super_blocks[k] = -Jf' * Q_inv
    end

    return SymPSDBorderedBlockTridiag{T,D,P}(
        diag_blocks, super_blocks, border_blocks, corner_block
    )
end

"""
Compute derivatives of G w.r.t. each state coordinate for Poisson SSM.

Handles multi-channel observations. For weight matrix W = diag(λ):
∂W/∂x_d = diag(λ .* Jh[:, d]) since ∂λ_m/∂x_d = λ_m * Jh[m, d]
"""
function calc_dGs_state_poisson(
    z::BorderedBlockVector{T,D,P}, ssm; obs_indices=nothing
) where {T,D,P}
    K = length(z.state_blocks)
    θ = z.param_block

    dGs = Matrix{BlockSparseStateDerivative{T,D,P,D^2,P^2,D * P}}(undef, K, D)

    Q_inv = calc_Qinv_param(ssm.dyn)

    obs_set = isnothing(obs_indices) ? nothing : Set(obs_indices)

    @inbounds for k in 1:K
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

        if has_obs
            η_k = h_param(ssm.sensor, z.state_blocks[k], θ)
            λ_k = exp.(η_k)  # M-vector
            W_t = Diagonal(λ_k)  # M×M diagonal
            Jh_k = calc_Jh_param(ssm.sensor, z.state_blocks[k], θ)  # M×D
            Hhs_k = calc_Hhs_param(ssm.sensor, z.state_blocks[k], θ)
            h_θ_k = calc_h_θ(ssm.sensor, z.state_blocks[k], θ)  # M×P
            Hh_θs_k = calc_Hh_θs(ssm.sensor, z.state_blocks[k], θ)
        end

        Mf = Q_inv * Jf_k1

        for d in 1:D
            diag_block = zeros(SMatrix{D,D,T})
            if k < K
                diag_block += Hfs_k1[d]' * Mf + Mf' * Hfs_k1[d]
            end
            if has_obs
                # ∂(Jh^T W Jh)/∂x_d = Jh^T (∂W/∂x_d) Jh + Hhs terms (Hhs = 0 for linear η)
                # ∂W/∂x_d = diag(λ .* Jh[:, d])
                dW_dx_d = Diagonal(λ_k .* Jh_k[:, d])
                diag_block += Jh_k' * dW_dx_d * Jh_k
            end

            super_block = k < K ? -Hfs_k1[d]' * Q_inv : zeros(SMatrix{D,D,T})

            border_block_k = zeros(SMatrix{D,P,T})
            if k < K
                border_block_k += Hfs_k1[d]' * Q_inv * f_θ_k1 + Jf_k1' * Q_inv * Hf_θs_k1[d]
            end
            if has_obs
                # ∂(Jh^T W h_θ)/∂x_d = Jh^T (∂W/∂x_d) h_θ + Jh^T W (∂h_θ/∂x_d)
                dW_dx_d = Diagonal(λ_k .* Jh_k[:, d])
                border_block_k += Jh_k' * dW_dx_d * h_θ_k + Jh_k' * W_t * Hh_θs_k[d]
            end

            border_block_k1 = k < K ? -Q_inv * Hf_θs_k1[d] : zeros(SMatrix{D,P,T})

            corner_block = zeros(SMatrix{P,P,T})
            if k < K
                corner_block +=
                    Hf_θs_k1[d]' * Q_inv * f_θ_k1 + f_θ_k1' * Q_inv * Hf_θs_k1[d]
            end
            if has_obs
                # ∂(h_θ^T W h_θ)/∂x_d = h_θ^T (∂W/∂x_d) h_θ + Hh_θs terms
                dW_dx_d = Diagonal(λ_k .* Jh_k[:, d])
                corner_block +=
                    Hh_θs_k[d]' * W_t * h_θ_k +
                    h_θ_k' * dW_dx_d * h_θ_k +
                    h_θ_k' * W_t * Hh_θs_k[d]
            end

            dGs[k, d] = BlockSparseStateDerivative{T,D,P}(
                k, d, diag_block, super_block, border_block_k, border_block_k1, corner_block
            )
        end
    end

    return dGs
end

"""
Compute derivatives of G w.r.t. each parameter coordinate for Poisson SSM.

Handles multi-channel observations. For weight matrix W = diag(λ):
∂W/∂θ_p = diag(λ .* h_θ[:, p]) since ∂λ_m/∂θ_p = λ_m * h_θ[m, p]
"""
function calc_dGs_param_poisson(
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

    obs_set = isnothing(obs_indices) ? nothing : Set(obs_indices)

    for p in 1:P
        diag_blocks = Vector{SMatrix{D,D,T,D^2}}(undef, K)
        super_blocks = Vector{SMatrix{D,D,T,D^2}}(undef, K - 1)
        border_blocks = Vector{SMatrix{D,P,T,D * P}}(undef, K)
        corner_block = zeros(SMatrix{P,P,T,P^2})

        @inbounds for k in 1:K
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

            diag_block = zeros(SMatrix{D,D,T})
            if k < K
                diag_block += Jf_θ_k1[p]' * Q_inv * Jf_k1 + Jf_k1' * Q_inv * Jf_θ_k1[p]
            end
            if has_obs
                η_k = h_param(ssm.sensor, z.state_blocks[k], θ)
                λ_k = exp.(η_k)
                W_t = Diagonal(λ_k)
                Jh_k = calc_Jh_param(ssm.sensor, z.state_blocks[k], θ)
                Jh_θ_k = calc_Jh_θ(ssm.sensor, z.state_blocks[k], θ)
                h_θ_k = calc_h_θ(ssm.sensor, z.state_blocks[k], θ)

                # ∂W/∂θ_p = diag(λ .* h_θ[:, p])
                dW_dθ_p = Diagonal(λ_k .* h_θ_k[:, p])
                diag_block +=
                    Jh_θ_k[p]' * W_t * Jh_k +
                    Jh_k' * dW_dθ_p * Jh_k +
                    Jh_k' * W_t * Jh_θ_k[p]
            end
            diag_blocks[k] = diag_block

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
                η_k = h_param(ssm.sensor, z.state_blocks[k], θ)
                λ_k = exp.(η_k)
                W_t = Diagonal(λ_k)
                Jh_k = calc_Jh_param(ssm.sensor, z.state_blocks[k], θ)
                Jh_θ_k = calc_Jh_θ(ssm.sensor, z.state_blocks[k], θ)
                h_θ_k = calc_h_θ(ssm.sensor, z.state_blocks[k], θ)
                h_θθ_k = calc_h_θθ(ssm.sensor, z.state_blocks[k], θ)

                dW_dθ_p = Diagonal(λ_k .* h_θ_k[:, p])
                border_block +=
                    Jh_θ_k[p]' * W_t * h_θ_k +
                    Jh_k' * dW_dθ_p * h_θ_k +
                    Jh_k' * W_t * h_θθ_k[p]
            end
            border_blocks[k] = border_block

            if has_obs
                η_k = h_param(ssm.sensor, z.state_blocks[k], θ)
                λ_k = exp.(η_k)
                W_t = Diagonal(λ_k)
                h_θ_k = calc_h_θ(ssm.sensor, z.state_blocks[k], θ)
                h_θθ_k = calc_h_θθ(ssm.sensor, z.state_blocks[k], θ)

                dW_dθ_p = Diagonal(λ_k .* h_θ_k[:, p])
                corner_block +=
                    h_θθ_k[p]' * W_t * h_θ_k +
                    h_θ_k' * dW_dθ_p * h_θ_k +
                    h_θ_k' * W_t * h_θθ_k[p]
            end
        end

        @inbounds for k in 1:(K - 1)
            Jf_θ_k1 = calc_Jf_θ(ssm.dyn, z.state_blocks[k], θ)
            super_blocks[k] = -Jf_θ_k1[p]' * Q_inv
        end

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
