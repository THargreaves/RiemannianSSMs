"""
RHMC infrastructure adapted for Poisson observations with state-dependent weight matrix.

Key differences from Gaussian case:
- Likelihood uses Poisson distribution instead of MvNormal
- Weight matrix W_t = λ_t depends on both state and parameters
- Metric derivatives must account for ∂W_t/∂x and ∂W_t/∂θ
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
        # Backward compatible: observations at every time step
        @inbounds for k in 1:K
            η = h_param(ssm.sensor, z.state_blocks[k], θ)  # Linear predictor
            λ = exp(η[1])  # Poisson rate
            ll += logpdf(Poisson(λ), ys[k][1])  # y is 1-element SVector
        end
    else
        # Sparse observations
        @inbounds for (obs_idx, k) in enumerate(obs_indices)
            η = h_param(ssm.sensor, z.state_blocks[k], θ)
            λ = exp(η[1])
            ll += logpdf(Poisson(λ), ys[obs_idx][1])
        end
    end

    return ll
end

"""
Calculate gradient of log-likelihood for Poisson SSM.

Poisson gradient: ∂ℓ/∂η = y - λ (since ∂log P/∂λ = (y-λ)/λ and ∂λ/∂η = λ cancel)
Then ∂ℓ/∂z = (y - λ) · ∂η/∂z where η is the linear predictor.
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
    param_grad = -prior_prec .* (θ - prior_mean)  # Parameter prior gradient

    Q_inv = calc_Qinv_param(ssm.dyn)

    # Incoming dynamics (state prior for k=1, dynamics for k>1)
    @inbounds state_grads[1] = -inv(ssm.prior.Σ) * (z.state_blocks[1] - ssm.prior.μ)
    @inbounds for k in 2:K
        μ = f_param(ssm.dyn, z.state_blocks[k - 1], θ)
        state_grads[k] = -Q_inv * (z.state_blocks[k] - μ)
    end

    # Poisson observation term (only at observed time steps)
    if isnothing(obs_indices)
        # Backward compatible: observations at every time step
        @inbounds for k in 1:K
            η = h_param(ssm.sensor, z.state_blocks[k], θ)[1]
            λ = exp(η)
            y = ys[k][1]

            # Gradient term as 1-element SVector for proper matrix multiplication
            gradient_term = @SVector [y - λ]

            # Gradient w.r.t. state: (y - λ) · ∂η/∂x
            Jh = calc_Jh_param(ssm.sensor, z.state_blocks[k], θ)  # ∂η/∂x (1×2)
            state_grads[k] += Jh' * gradient_term  # (2×1) * (1-vec) = 2-vec

            # Gradient w.r.t. parameters: (y - λ) · ∂η/∂θ
            h_θ = calc_h_θ(ssm.sensor, z.state_blocks[k], θ)  # ∂η/∂θ (1×3)
            param_grad += h_θ' * gradient_term  # (3×1) * (1-vec) = 3-vec
        end
    else
        # Sparse observations
        @inbounds for (obs_idx, k) in enumerate(obs_indices)
            η = h_param(ssm.sensor, z.state_blocks[k], θ)[1]
            λ = exp(η)
            y = ys[obs_idx][1]

            gradient_term = @SVector [y - λ]

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

        # Parameter gradient from dynamics
        f_θ = calc_f_θ(ssm.dyn, z.state_blocks[k - 1], θ)
        param_grad += f_θ' * Q_inv * residual
    end

    return BorderedBlockVector{T,D,P}(state_grads, param_grad)
end

"""
Calculate Gauss-Newton metric for Poisson SSM.

Key difference: Weight matrix W_t = λ_t = exp(η_t) is state and parameter dependent!
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

    # Build set of observed indices for fast lookup
    obs_set = isnothing(obs_indices) ? nothing : Set(obs_indices)

    # Compute diagonal blocks and border blocks
    @inbounds for k in 1:K
        # Base term (prior for k=1, dynamics for k>1)
        Λ = k == 1 ? SMatrix{D,D,T}(inv(ssm.prior.Σ)) : SMatrix{D,D,T}(Q_inv)

        # Check if this time step has an observation
        has_obs = isnothing(obs_set) || (k in obs_set)

        # Poisson observation term (only if observed)
        if has_obs
            # W_t = λ_t (state-dependent weight!)
            W_t = calc_Rinv_param(ssm.sensor, z.state_blocks[k], θ)[1, 1]
            Jh = calc_Jh_param(ssm.sensor, z.state_blocks[k], θ)  # ∂η/∂x
            Λ += Jh' * W_t * Jh
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
            W_t = calc_Rinv_param(ssm.sensor, z.state_blocks[k], θ)[1, 1]
            Jh = calc_Jh_param(ssm.sensor, z.state_blocks[k], θ)
            h_θ = calc_h_θ(ssm.sensor, z.state_blocks[k], θ)
            border_k += Jh' * W_t * h_θ
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
            W_t = calc_Rinv_param(ssm.sensor, z.state_blocks[k], θ)[1, 1]
            h_θ_k = calc_h_θ(ssm.sensor, z.state_blocks[k], θ)
            corner_block += h_θ_k' * W_t * h_θ_k
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
Compute derivatives of G w.r.t. each state coordinate for Poisson SSM.

Critical: Must account for ∂W_t/∂x where W_t = λ_t = exp(η_t)!
"""
function calc_dGs_state_poisson(
    z::BorderedBlockVector{T,D,P}, ssm; obs_indices=nothing
) where {T,D,P}
    K = length(z.state_blocks)
    θ = z.param_block

    dGs = Matrix{BlockSparseStateDerivative{T,D,P,D^2,P^2,D * P}}(undef, K, D)

    Q_inv = calc_Qinv_param(ssm.dyn)

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

        # Compute observation-related derivatives and weight (only if observed)
        if has_obs
            η_k = h_param(ssm.sensor, z.state_blocks[k], θ)[1]
            W_t = exp(η_k)  # λ_t
            Jh_k = calc_Jh_param(ssm.sensor, z.state_blocks[k], θ)  # ∂η/∂x
            Hhs_k = calc_Hhs_param(ssm.sensor, z.state_blocks[k], θ)  # = 0 for linear η
            h_θ_k = calc_h_θ(ssm.sensor, z.state_blocks[k], θ)  # ∂η/∂θ
            Hh_θs_k = calc_Hh_θs(ssm.sensor, z.state_blocks[k], θ)  # ∂²η/∂x∂θ
        else
            W_t = zero(T)
            Jh_k = zeros(SMatrix{1,D,T})
            Hhs_k = ntuple(_ -> zeros(SMatrix{1,D,T}), D)
            h_θ_k = zeros(SMatrix{1,P,T})
            Hh_θs_k = ntuple(_ -> zeros(SMatrix{1,P,T}), D)
        end

        Mf = Q_inv * Jf_k1

        for d in 1:D
            # Diagonal block ∂G_{kk}/∂x_k^{(d)}
            diag_block = zeros(SMatrix{D,D,T})
            if k < K
                diag_block += Hfs_k1[d]' * Mf + Mf' * Hfs_k1[d]
            end
            if has_obs
                # For Poisson: ∂(Jh^T W Jh)/∂x_d = Jh^T (∂W/∂x_d) Jh
                # since Hhs = 0 (η is linear in x)
                # ∂W/∂x_d = λ · ∂η/∂x_d = λ · Jh[d]
                dW_dx_d = W_t * Jh_k[1, d]
                diag_block += Jh_k' * dW_dx_d * Jh_k
            end

            # Off-diagonal block ∂G_{k,k+1}/∂x_k^{(d)}
            super_block = k < K ? -Hfs_k1[d]' * Q_inv : zeros(SMatrix{D,D,T})

            # Border block ∂G_{k,θ}/∂x_k^{(d)}
            border_block_k = zeros(SMatrix{D,P,T})
            if k < K
                border_block_k += Hfs_k1[d]' * Q_inv * f_θ_k1 + Jf_k1' * Q_inv * Hf_θs_k1[d]
            end
            if has_obs
                # ∂(Jh^T W h_θ)/∂x_d = (∂Jh/∂x_d)^T W h_θ + Jh^T (∂W/∂x_d) h_θ + Jh^T W (∂h_θ/∂x_d)
                # = 0 + Jh^T (λ · Jh[d]) h_θ + Jh^T W Hh_θs[d]
                dW_dx_d = W_t * Jh_k[1, d]
                border_block_k += Jh_k' * dW_dx_d * h_θ_k + Jh_k' * W_t * Hh_θs_k[d]
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
                # ∂(h_θ^T W h_θ)/∂x_d = h_θ^T (∂W/∂x_d) h_θ  (since ∂h_θ/∂x = Hh_θs is simple)
                dW_dx_d = W_t * Jh_k[1, d]
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

Critical: Must account for ∂W_t/∂θ where W_t = λ_t = exp(η_t)!
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

            # Diagonal block ∂G_{kk}/∂θ^{(p)}
            diag_block = zeros(SMatrix{D,D,T})
            if k < K
                diag_block += Jf_θ_k1[p]' * Q_inv * Jf_k1 + Jf_k1' * Q_inv * Jf_θ_k1[p]
            end
            if has_obs
                # For Poisson: ∂(Jh^T W Jh)/∂θ_p = (∂Jh/∂θ_p)^T W Jh + Jh^T (∂W/∂θ_p) Jh + Jh^T W (∂Jh/∂θ_p)
                η_k = h_param(ssm.sensor, z.state_blocks[k], θ)[1]
                W_t = exp(η_k)
                Jh_k = calc_Jh_param(ssm.sensor, z.state_blocks[k], θ)
                Jh_θ_k = calc_Jh_θ(ssm.sensor, z.state_blocks[k], θ)
                h_θ_k = calc_h_θ(ssm.sensor, z.state_blocks[k], θ)

                # ∂W/∂θ_p = λ · ∂η/∂θ_p = λ · h_θ[p]
                dW_dθ_p = W_t * h_θ_k[1, p]
                diag_block +=
                    Jh_θ_k[p]' * W_t * Jh_k +
                    Jh_k' * dW_dθ_p * Jh_k +
                    Jh_k' * W_t * Jh_θ_k[p]
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
                η_k = h_param(ssm.sensor, z.state_blocks[k], θ)[1]
                W_t = exp(η_k)
                Jh_k = calc_Jh_param(ssm.sensor, z.state_blocks[k], θ)
                Jh_θ_k = calc_Jh_θ(ssm.sensor, z.state_blocks[k], θ)
                h_θ_k = calc_h_θ(ssm.sensor, z.state_blocks[k], θ)
                h_θθ_k = calc_h_θθ(ssm.sensor, z.state_blocks[k], θ)

                dW_dθ_p = W_t * h_θ_k[1, p]
                border_block +=
                    Jh_θ_k[p]' * W_t * h_θ_k +
                    Jh_k' * dW_dθ_p * h_θ_k +
                    Jh_k' * W_t * h_θθ_k[p]
            end
            border_blocks[k] = border_block

            # Corner block contributions from this k (only if observed)
            if has_obs
                η_k = h_param(ssm.sensor, z.state_blocks[k], θ)[1]
                W_t = exp(η_k)
                h_θ_k = calc_h_θ(ssm.sensor, z.state_blocks[k], θ)
                h_θθ_k = calc_h_θθ(ssm.sensor, z.state_blocks[k], θ)

                dW_dθ_p = W_t * h_θ_k[1, p]
                corner_block +=
                    h_θθ_k[p]' * W_t * h_θ_k +
                    h_θ_k' * dW_dθ_p * h_θ_k +
                    h_θ_k' * W_t * h_θθ_k[p]
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
