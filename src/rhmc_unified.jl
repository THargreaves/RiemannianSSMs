"""
Unified Riemannian HMC for state space models with exponential family observations.

This module provides RHMC inference for any StateSpaceModel that implements the
required interface. The key insight is that the Fisher information metric has a
bordered block-tridiagonal structure that can be exploited for O(T·D³) computation.

The metric construction uses the local expected Fisher information, which for
exponential family observations and Gaussian dynamics reduces to outer products
of Jacobians weighted by the observation information S_t = ∇²A(η_t).
"""

using AdvancedHMC
using Distributions
using LogDensityProblems

import AdvancedHMC: DualValue

export RiemannianMetric, RHMCLogDensity
export calc_loglikelihood, calc_loglikelihood_grad
export calc_G, calc_dGs_state, calc_dGs_param
export simulate

# =============================================================================
# Riemannian Metric Type (AdvancedHMC Integration)
# =============================================================================

"""
    RiemannianMetric{Dx, Dy, Dp, T, M, YT, OI}

Riemannian metric for RHMC with bordered block-tridiagonal Fisher information.

Type parameters:
- `Dx`: State dimension
- `Dy`: Observation dimension
- `Dp`: Parameter dimension
- `T`: Element type
- `M`: Model type (subtype of StateSpaceModel)
- `YT`: Observations type
- `OI`: Observation indices type (Vector{Int} or Nothing)
"""
struct RiemannianMetric{Dx,Dy,Dp,T,M<:StateSpaceModel{Dx,Dy,Dp},YT,OI} <:
       AdvancedHMC.AbstractRiemannianMetric
    model::M
    ys::YT
    obs_indices::OI
    K::Int                      # Total number of latent states
    prior_mean::SVector{Dp,T}   # Prior mean for parameters
    prior_prec::SVector{Dp,T}   # Prior precision (diagonal)
end

function RiemannianMetric(
    model::StateSpaceModel{Dx,Dy,Dp},
    ys,
    K::Int,
    prior_mean::AbstractVector,
    prior_var::AbstractVector;
    obs_indices::Union{Vector{Int},Nothing}=nothing,
) where {Dx,Dy,Dp}
    T = eltype(prior_mean)
    pm = SVector{Dp,T}(prior_mean)
    pp = SVector{Dp,T}(1 ./ prior_var)
    return RiemannianMetric{Dx,Dy,Dp,T,typeof(model),typeof(ys),typeof(obs_indices)}(
        model, ys, obs_indices, K, pm, pp
    )
end

Base.size(m::RiemannianMetric{Dx,Dy,Dp}) where {Dx,Dy,Dp} = (m.K * Dx + Dp,)
Base.size(m::RiemannianMetric{Dx,Dy,Dp}, ::Int) where {Dx,Dy,Dp} = m.K * Dx + Dp
Base.eltype(::RiemannianMetric{Dx,Dy,Dp,T}) where {Dx,Dy,Dp,T} = T

function Base.show(io::IO, m::RiemannianMetric{Dx,Dy,Dp}) where {Dx,Dy,Dp}
    return print(io, "RiemannianMetric(K=$(m.K), Dx=$Dx, Dy=$Dy, Dp=$Dp)")
end

# =============================================================================
# Log-Likelihood and Gradient
# =============================================================================

"""
    calc_loglikelihood(z, model, ys, prior_mean, prior_prec; obs_indices=nothing)

Compute log p(x_{1:K}, θ | y_{1:T}) up to a constant.

Components:
- Parameter prior: log p(θ)
- State prior: log p(x_1)
- Dynamics: Σ_{k=2}^K log p(x_k | x_{k-1}, θ)
- Observations: Σ_t log p(y_t | x_t, θ)
"""
function calc_loglikelihood(
    z::BorderedBlockVector{T,Dx,Dp},
    model::StateSpaceModel{Dx,Dy,Dp},
    ys,
    prior_mean::SVector{Dp,T},
    prior_prec::SVector{Dp,T};
    obs_indices=nothing,
) where {T,Dx,Dy,Dp}
    K = length(z.state_blocks)
    θ = z.param_block

    # Parameter prior (Gaussian with diagonal covariance)
    ll = -T(0.5) * sum(prior_prec .* (θ - prior_mean) .^ 2)

    # State prior
    ll += logpdf(initial_prior(model), z.state_blocks[1])

    # Dynamics
    Q = dynamics_covariance(model)
    @inbounds for k in 2:K
        μ = f(model, z.state_blocks[k - 1], θ)
        ll += logpdf(MvNormal(μ, Q), z.state_blocks[k])
    end

    # Observations (exponential family)
    ef = observation_family(model)
    if isnothing(obs_indices)
        @inbounds for k in 1:K
            η_k = η(model, z.state_blocks[k], θ)
            ll += log_likelihood(ef, η_k, ys[k])
        end
    else
        @inbounds for (obs_idx, k) in enumerate(obs_indices)
            η_k = η(model, z.state_blocks[k], θ)
            ll += log_likelihood(ef, η_k, ys[obs_idx])
        end
    end

    return ll
end

"""
    calc_loglikelihood_grad(z, model, ys, prior_mean, prior_prec; obs_indices=nothing)

Compute gradient of log-likelihood w.r.t. states and parameters.

For EF observations, the score w.r.t. natural parameter is:
    ∂ℓ/∂η = T(y) - E[T(y)|η] = T(y) - ∇A(η)
"""
function calc_loglikelihood_grad(
    z::BorderedBlockVector{T,Dx,Dp},
    model::StateSpaceModel{Dx,Dy,Dp},
    ys,
    prior_mean::SVector{Dp,T},
    prior_prec::SVector{Dp,T};
    obs_indices=nothing,
) where {T,Dx,Dy,Dp}
    K = length(z.state_blocks)
    θ = z.param_block

    state_grads = Vector{SVector{Dx,T}}(undef, K)
    param_grad = -prior_prec .* (θ - prior_mean)

    Q_inv = dynamics_covariance_inv(model)
    prior_dist = initial_prior(model)
    ef = observation_family(model)

    # State prior gradient
    @inbounds state_grads[1] = -inv(prior_dist.Σ) * (z.state_blocks[1] - prior_dist.μ)

    # Incoming dynamics gradient
    @inbounds for k in 2:K
        μ = f(model, z.state_blocks[k - 1], θ)
        state_grads[k] = -Q_inv * (z.state_blocks[k] - μ)
    end

    # Observation gradient (EF score)
    if isnothing(obs_indices)
        @inbounds for k in 1:K
            η_k = η(model, z.state_blocks[k], θ)
            y_k = ys[k]

            # Score: T(y) - ∇A(η)
            score = sufficient_stat(ef, y_k) - log_partition_d1(ef, η_k)

            # Chain rule to state and parameters
            Dη_x_k = Dη_x(model, z.state_blocks[k], θ)
            Dη_θ_k = Dη_θ(model, z.state_blocks[k], θ)

            state_grads[k] += Dη_x_k' * score
            param_grad += Dη_θ_k' * score
        end
    else
        @inbounds for (obs_idx, k) in enumerate(obs_indices)
            η_k = η(model, z.state_blocks[k], θ)
            y_k = ys[obs_idx]

            score = sufficient_stat(ef, y_k) - log_partition_d1(ef, η_k)

            Dη_x_k = Dη_x(model, z.state_blocks[k], θ)
            Dη_θ_k = Dη_θ(model, z.state_blocks[k], θ)

            state_grads[k] += Dη_x_k' * score
            param_grad += Dη_θ_k' * score
        end
    end

    # Outgoing dynamics gradient
    @inbounds for k in 2:K
        Df_x_k = Df_x(model, z.state_blocks[k - 1], θ)
        μ = f(model, z.state_blocks[k - 1], θ)
        residual = z.state_blocks[k] - μ

        state_grads[k - 1] += Df_x_k' * Q_inv * residual

        Df_θ_k = Df_θ(model, z.state_blocks[k - 1], θ)
        param_grad += Df_θ_k' * Q_inv * residual
    end

    return BorderedBlockVector{T,Dx,Dp}(state_grads, param_grad)
end

# =============================================================================
# Fisher Information Metric
# =============================================================================

"""
    calc_G(z, model, prior_prec; λ=1e-7, obs_indices=nothing)

Compute the bordered block-tridiagonal Fisher information metric.

Diagonal blocks G_{kk}:
    G_{kk} = Q⁻¹ + Dη_x' S_k Dη_x + Df_x' Q⁻¹ Df_x
    (with boundary adjustments)

Off-diagonal blocks G_{k,k+1}:
    G_{k,k+1} = -Df_x' Q⁻¹

Border blocks G_{k,θ}:
    G_{k,θ} = Dη_x' S_k Dη_θ - Q⁻¹ Df_θ + Df_x' Q⁻¹ Df_θ

Corner block G_{θθ}:
    G_{θθ} = Σ_k Dη_θ' S_k Dη_θ + Σ_k Df_θ' Q⁻¹ Df_θ + prior_prec
"""
function calc_G(
    z::BorderedBlockVector{T,Dx,Dp},
    model::StateSpaceModel{Dx,Dy,Dp},
    prior_prec::SVector{Dp,T};
    λ=T(1e-7),
    obs_indices=nothing,
) where {T,Dx,Dy,Dp}
    K = length(z.state_blocks)
    θ = z.param_block

    diag_blocks = Vector{SMatrix{Dx,Dx,T,Dx^2}}(undef, K)
    super_blocks = Vector{SMatrix{Dx,Dx,T,Dx^2}}(undef, K - 1)
    border_blocks = Vector{SMatrix{Dx,Dp,T,Dx * Dp}}(undef, K)
    corner_block = zeros(SMatrix{Dp,Dp,T,Dp^2})

    Q_inv = dynamics_covariance_inv(model)
    prior_dist = initial_prior(model)
    ef = observation_family(model)

    # Build observation index set
    obs_set = isnothing(obs_indices) ? nothing : Set(obs_indices)

    @inbounds for k in 1:K
        # Base term: prior for k=1, dynamics for k>1
        Λ = k == 1 ? SMatrix{Dx,Dx,T}(inv(prior_dist.Σ)) : SMatrix{Dx,Dx,T}(Q_inv)

        has_obs = isnothing(obs_set) || (k in obs_set)

        # Observation term
        if has_obs
            η_k = η(model, z.state_blocks[k], θ)
            S_k = log_partition_d2(ef, η_k)  # Fisher information (Dy vector for products)
            Dη_x_k = Dη_x(model, z.state_blocks[k], θ)

            # For product EF, S_k is a vector representing diagonal
            if S_k isa SVector
                Λ += Dη_x_k' * Diagonal(S_k) * Dη_x_k
            else
                # Scalar case (Dy=1)
                Λ += S_k * (Dη_x_k' * Dη_x_k)
            end
        end

        # Outgoing dynamics term (contribution to diagonal from p(x_{k+1}|x_k))
        if k < K
            Df_x_k = Df_x(model, z.state_blocks[k], θ)
            Λ += Df_x_k' * Q_inv * Df_x_k
        end

        # Symmetrize and regularize
        Λ = (Λ + Λ') / 2 + λ * I
        diag_blocks[k] = Λ

        # Border block G_{k,θ}
        border_k = zeros(SMatrix{Dx,Dp,T})

        # Observation contribution to border
        if has_obs
            η_k = η(model, z.state_blocks[k], θ)
            S_k = log_partition_d2(ef, η_k)
            Dη_x_k = Dη_x(model, z.state_blocks[k], θ)
            Dη_θ_k = Dη_θ(model, z.state_blocks[k], θ)

            if S_k isa SVector
                border_k += Dη_x_k' * Diagonal(S_k) * Dη_θ_k
            else
                border_k += S_k * (Dη_x_k' * Dη_θ_k)
            end
        end

        # Incoming dynamics contribution: -Q⁻¹ Df_θ_k (from p(x_k|x_{k-1}))
        if k > 1
            Df_θ_k = Df_θ(model, z.state_blocks[k - 1], θ)
            border_k += -SMatrix{Dx,Dp,T}(Q_inv * Df_θ_k)
        end

        # Outgoing dynamics contribution: Df_x' Q⁻¹ Df_θ (from p(x_{k+1}|x_k))
        if k < K
            Df_x_k = Df_x(model, z.state_blocks[k], θ)
            Df_θ_k = Df_θ(model, z.state_blocks[k], θ)
            border_k += Df_x_k' * Q_inv * Df_θ_k
        end

        border_blocks[k] = border_k

        # Corner block contributions from observations
        if has_obs
            η_k = η(model, z.state_blocks[k], θ)
            S_k = log_partition_d2(ef, η_k)
            Dη_θ_k = Dη_θ(model, z.state_blocks[k], θ)

            if S_k isa SVector
                corner_block += Dη_θ_k' * Diagonal(S_k) * Dη_θ_k
            else
                corner_block += S_k * (Dη_θ_k' * Dη_θ_k)
            end
        end
    end

    # Corner block contributions from dynamics
    @inbounds for k in 2:K
        Df_θ_k = Df_θ(model, z.state_blocks[k - 1], θ)
        corner_block += Df_θ_k' * Q_inv * Df_θ_k
    end

    # Add prior precision
    corner_block += Diagonal(prior_prec)
    corner_block = (corner_block + corner_block') / 2 + λ * I

    # Off-diagonal blocks
    @inbounds for k in 1:(K - 1)
        Df_x_k = Df_x(model, z.state_blocks[k], θ)
        super_blocks[k] = -Df_x_k' * Q_inv
    end

    return SymPSDBorderedBlockTridiag{T,Dx,Dp}(
        diag_blocks, super_blocks, border_blocks, corner_block
    )
end

# =============================================================================
# Metric Derivatives w.r.t. State
# =============================================================================

"""
    calc_dGs_state(z, model; obs_indices=nothing)

Compute ∂G/∂x_k^{(d)} for all k, d.

Returns a K×Dx matrix of BlockSparseStateDerivative.

For state-dependent S_t (non-Gaussian EF), we need:
    ∂S_t/∂x_d = Σ_m A'''(η_m) · (Dη_x)_{m,d}
"""
function calc_dGs_state(
    z::BorderedBlockVector{T,Dx,Dp}, model::StateSpaceModel{Dx,Dy,Dp}; obs_indices=nothing
) where {T,Dx,Dy,Dp}
    K = length(z.state_blocks)
    θ = z.param_block

    dGs = Matrix{BlockSparseStateDerivative{T,Dx,Dp,Dx^2,Dp^2,Dx * Dp}}(undef, K, Dx)

    Q_inv = dynamics_covariance_inv(model)
    ef = observation_family(model)

    obs_set = isnothing(obs_indices) ? nothing : Set(obs_indices)

    @inbounds for k in 1:K
        has_obs = isnothing(obs_set) || (k in obs_set)

        # Dynamics derivatives at k (for outgoing transition)
        if k < K
            Df_x_k = Df_x(model, z.state_blocks[k], θ)
            D2f_xx_k = D2f_xx(model, z.state_blocks[k], θ)
            Df_θ_k = Df_θ(model, z.state_blocks[k], θ)
            D2f_xθ_k = D2f_xθ(model, z.state_blocks[k], θ)
            Mf = Q_inv * Df_x_k
        else
            Df_x_k = zeros(SMatrix{Dx,Dx,T})
            D2f_xx_k = ntuple(_ -> zeros(SMatrix{Dx,Dx,T}), Dx)
            Df_θ_k = zeros(SMatrix{Dx,Dp,T})
            D2f_xθ_k = ntuple(_ -> zeros(SMatrix{Dx,Dp,T}), Dx)
            Mf = zeros(SMatrix{Dx,Dx,T})
        end

        # Observation derivatives at k
        if has_obs
            η_k = η(model, z.state_blocks[k], θ)
            S_k = log_partition_d2(ef, η_k)
            S_k_d3 = log_partition_d3(ef, η_k)  # ∂S/∂η
            Dη_x_k = Dη_x(model, z.state_blocks[k], θ)
            Dη_θ_k = Dη_θ(model, z.state_blocks[k], θ)
            D2η_xx_k = D2η_xx(model, z.state_blocks[k], θ)
            D2η_xθ_k = D2η_xθ(model, z.state_blocks[k], θ)
        end

        for d in 1:Dx
            # Diagonal block ∂G_{kk}/∂x_k^{(d)}
            diag_block = zeros(SMatrix{Dx,Dx,T})

            # Dynamics contribution
            if k < K
                H_fd = D2f_xx_k[d]  # ∂²f/∂x∂x_d
                diag_block += H_fd' * Mf + Mf' * H_fd
            end

            # Observation contribution
            if has_obs
                H_ηd = D2η_xx_k[d]  # ∂²η/∂x∂x_d

                if S_k isa SVector
                    # Product EF: S is diagonal
                    S_diag = Diagonal(S_k)
                    Mη = S_diag * Dη_x_k

                    # Term from ∂(Dη_x' S Dη_x)/∂x_d
                    diag_block += H_ηd' * Mη + Mη' * H_ηd

                    # Term from S depending on x: ∂S/∂x_d = diag(S_k_d3 .* Dη_x[:,d])
                    dS_dx_d = Diagonal(S_k_d3 .* Dη_x_k[:, d])
                    diag_block += Dη_x_k' * dS_dx_d * Dη_x_k
                else
                    # Scalar EF
                    Mη = S_k * Dη_x_k
                    diag_block += H_ηd' * Mη + Mη' * H_ηd

                    # Scalar S_k_d3
                    dS_dx_d = S_k_d3 * Dη_x_k[1, d]
                    diag_block += dS_dx_d * (Dη_x_k' * Dη_x_k)
                end
            end

            # Off-diagonal block ∂G_{k,k+1}/∂x_k^{(d)}
            if k < K
                H_fd = D2f_xx_k[d]
                super_block = -H_fd' * Q_inv
            else
                super_block = zeros(SMatrix{Dx,Dx,T})
            end

            # Border block ∂G_{k,θ}/∂x_k^{(d)}
            border_block_k = zeros(SMatrix{Dx,Dp,T})

            if k < K
                H_fd = D2f_xx_k[d]
                H_fθd = D2f_xθ_k[d]
                border_block_k += H_fd' * Q_inv * Df_θ_k + Df_x_k' * Q_inv * H_fθd
            end

            if has_obs
                H_ηd = D2η_xx_k[d]
                H_ηθd = D2η_xθ_k[d]

                if S_k isa SVector
                    S_diag = Diagonal(S_k)
                    border_block_k += H_ηd' * S_diag * Dη_θ_k + Dη_x_k' * S_diag * H_ηθd

                    # S depends on x
                    dS_dx_d = Diagonal(S_k_d3 .* Dη_x_k[:, d])
                    border_block_k += Dη_x_k' * dS_dx_d * Dη_θ_k
                else
                    border_block_k += S_k * (H_ηd' * Dη_θ_k + Dη_x_k' * H_ηθd)
                    dS_dx_d = S_k_d3 * Dη_x_k[1, d]
                    border_block_k += dS_dx_d * (Dη_x_k' * Dη_θ_k)
                end
            end

            # Border block ∂G_{k+1,θ}/∂x_k^{(d)}
            if k < K
                H_fθd = D2f_xθ_k[d]
                border_block_k1 = -Q_inv * H_fθd
            else
                border_block_k1 = zeros(SMatrix{Dx,Dp,T})
            end

            # Corner block ∂G_{θθ}/∂x_k^{(d)}
            corner_block = zeros(SMatrix{Dp,Dp,T})

            if k < K
                H_fθd = D2f_xθ_k[d]
                corner_block += H_fθd' * Q_inv * Df_θ_k + Df_θ_k' * Q_inv * H_fθd
            end

            if has_obs
                H_ηθd = D2η_xθ_k[d]

                if S_k isa SVector
                    S_diag = Diagonal(S_k)
                    corner_block += H_ηθd' * S_diag * Dη_θ_k + Dη_θ_k' * S_diag * H_ηθd

                    dS_dx_d = Diagonal(S_k_d3 .* Dη_x_k[:, d])
                    corner_block += Dη_θ_k' * dS_dx_d * Dη_θ_k
                else
                    corner_block += S_k * (H_ηθd' * Dη_θ_k + Dη_θ_k' * H_ηθd)
                    dS_dx_d = S_k_d3 * Dη_x_k[1, d]
                    corner_block += dS_dx_d * (Dη_θ_k' * Dη_θ_k)
                end
            end

            dGs[k, d] = BlockSparseStateDerivative{T,Dx,Dp}(
                k, d, diag_block, super_block, border_block_k, border_block_k1, corner_block
            )
        end
    end

    return dGs
end

# =============================================================================
# Metric Derivatives w.r.t. Parameters
# =============================================================================

"""
    calc_dGs_param(z, model; obs_indices=nothing)

Compute ∂G/∂θ^{(p)} for all p.

Returns a vector of P SymPSDBorderedBlockTridiag matrices.

For state-dependent S_t:
    ∂S_t/∂θ_p = Σ_m A'''(η_m) · (Dη_θ)_{m,p}
"""
function calc_dGs_param(
    z::BorderedBlockVector{T,Dx,Dp}, model::StateSpaceModel{Dx,Dy,Dp}; obs_indices=nothing
) where {T,Dx,Dy,Dp}
    K = length(z.state_blocks)
    θ = z.param_block

    dGs = Vector{
        SymPSDBorderedBlockTridiag{
            T,
            Dx,
            Dp,
            Dx^2,
            Dp^2,
            Dx * Dp,
            Vector{SMatrix{Dx,Dx,T,Dx^2}},
            Vector{SMatrix{Dx,Dp,T,Dx * Dp}},
        },
    }(
        undef, Dp
    )

    Q_inv = dynamics_covariance_inv(model)
    ef = observation_family(model)

    obs_set = isnothing(obs_indices) ? nothing : Set(obs_indices)

    for p in 1:Dp
        diag_blocks = Vector{SMatrix{Dx,Dx,T,Dx^2}}(undef, K)
        super_blocks = Vector{SMatrix{Dx,Dx,T,Dx^2}}(undef, K - 1)
        border_blocks = Vector{SMatrix{Dx,Dp,T,Dx * Dp}}(undef, K)
        corner_block = zeros(SMatrix{Dp,Dp,T,Dp^2})

        @inbounds for k in 1:K
            has_obs = isnothing(obs_set) || (k in obs_set)

            # Dynamics derivatives at k
            if k < K
                Df_x_k = Df_x(model, z.state_blocks[k], θ)
                DDf_x_θ_k = DDf_x_θ(model, z.state_blocks[k], θ)
                Df_θ_k = Df_θ(model, z.state_blocks[k], θ)
                D2f_θθ_k = D2f_θθ(model, z.state_blocks[k], θ)
            end

            # Diagonal block ∂G_{kk}/∂θ^{(p)}
            diag_block = zeros(SMatrix{Dx,Dx,T})

            if k < K
                dDf_x_dθp = DDf_x_θ_k[p]
                diag_block += dDf_x_dθp' * Q_inv * Df_x_k + Df_x_k' * Q_inv * dDf_x_dθp
            end

            if has_obs
                η_k = η(model, z.state_blocks[k], θ)
                S_k = log_partition_d2(ef, η_k)
                S_k_d3 = log_partition_d3(ef, η_k)
                Dη_x_k = Dη_x(model, z.state_blocks[k], θ)
                Dη_θ_k = Dη_θ(model, z.state_blocks[k], θ)
                DDη_x_θ_k = DDη_x_θ(model, z.state_blocks[k], θ)

                dDη_x_dθp = DDη_x_θ_k[p]

                if S_k isa SVector
                    S_diag = Diagonal(S_k)
                    diag_block +=
                        dDη_x_dθp' * S_diag * Dη_x_k + Dη_x_k' * S_diag * dDη_x_dθp

                    # S depends on θ
                    dS_dθ_p = Diagonal(S_k_d3 .* Dη_θ_k[:, p])
                    diag_block += Dη_x_k' * dS_dθ_p * Dη_x_k
                else
                    diag_block += S_k * (dDη_x_dθp' * Dη_x_k + Dη_x_k' * dDη_x_dθp)
                    dS_dθ_p = S_k_d3 * Dη_θ_k[1, p]
                    diag_block += dS_dθ_p * (Dη_x_k' * Dη_x_k)
                end
            end

            diag_blocks[k] = diag_block

            # Border block ∂G_{k,θ}/∂θ^{(p)}
            border_block = zeros(SMatrix{Dx,Dp,T})

            if k > 1
                D2f_θθ_km1 = D2f_θθ(model, z.state_blocks[k - 1], θ)
                border_block += -Q_inv * D2f_θθ_km1[p]
            end

            if k < K
                dDf_x_dθp = DDf_x_θ_k[p]
                df_θθ_p = D2f_θθ_k[p]
                border_block += dDf_x_dθp' * Q_inv * Df_θ_k + Df_x_k' * Q_inv * df_θθ_p
            end

            if has_obs
                dDη_x_dθp = DDη_x_θ_k[p]
                D2η_θθ_k = D2η_θθ(model, z.state_blocks[k], θ)
                dη_θθ_p = D2η_θθ_k[p]

                if S_k isa SVector
                    S_diag = Diagonal(S_k)
                    border_block +=
                        dDη_x_dθp' * S_diag * Dη_θ_k + Dη_x_k' * S_diag * dη_θθ_p

                    dS_dθ_p = Diagonal(S_k_d3 .* Dη_θ_k[:, p])
                    border_block += Dη_x_k' * dS_dθ_p * Dη_θ_k
                else
                    border_block += S_k * (dDη_x_dθp' * Dη_θ_k + Dη_x_k' * dη_θθ_p)
                    dS_dθ_p = S_k_d3 * Dη_θ_k[1, p]
                    border_block += dS_dθ_p * (Dη_x_k' * Dη_θ_k)
                end
            end

            border_blocks[k] = border_block

            # Corner block contributions
            if has_obs
                D2η_θθ_k = D2η_θθ(model, z.state_blocks[k], θ)
                dη_θθ_p = D2η_θθ_k[p]

                if S_k isa SVector
                    S_diag = Diagonal(S_k)
                    corner_block += dη_θθ_p' * S_diag * Dη_θ_k + Dη_θ_k' * S_diag * dη_θθ_p

                    dS_dθ_p = Diagonal(S_k_d3 .* Dη_θ_k[:, p])
                    corner_block += Dη_θ_k' * dS_dθ_p * Dη_θ_k
                else
                    corner_block += S_k * (dη_θθ_p' * Dη_θ_k + Dη_θ_k' * dη_θθ_p)
                    dS_dθ_p = S_k_d3 * Dη_θ_k[1, p]
                    corner_block += dS_dθ_p * (Dη_θ_k' * Dη_θ_k)
                end
            end
        end

        # Off-diagonal blocks
        @inbounds for k in 1:(K - 1)
            DDf_x_θ_k = DDf_x_θ(model, z.state_blocks[k], θ)
            super_blocks[k] = -DDf_x_θ_k[p]' * Q_inv
        end

        # Corner block dynamics contributions
        @inbounds for k in 2:K
            Df_θ_km1 = Df_θ(model, z.state_blocks[k - 1], θ)
            D2f_θθ_km1 = D2f_θθ(model, z.state_blocks[k - 1], θ)
            df_θθ_p = D2f_θθ_km1[p]
            corner_block += df_θθ_p' * Q_inv * Df_θ_km1 + Df_θ_km1' * Q_inv * df_θθ_p
        end

        dGs[p] = SymPSDBorderedBlockTridiag{T,Dx,Dp}(
            diag_blocks, super_blocks, border_blocks, corner_block
        )
    end

    return dGs
end

# =============================================================================
# AdvancedHMC Integration
# =============================================================================

function calc_G_from_metric(
    metric::RiemannianMetric{Dx,Dy,Dp,T}, z::AbstractVector; λ=T(1e-7)
) where {Dx,Dy,Dp,T}
    z_blocks = to_bordered_block_vector(z, Val(Dx), Val(Dp))
    return calc_G(
        z_blocks, metric.model, metric.prior_prec; λ=λ, obs_indices=metric.obs_indices
    )
end

function AdvancedHMC.rand_momentum(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    metric::RiemannianMetric{Dx,Dy,Dp,T},
    kinetic,
    z::AbstractVecOrMat,
) where {Dx,Dy,Dp,T}
    G = calc_G_from_metric(metric, z)
    G_chol = cholesky(G)

    # Sample Z ~ N(0, I) as bordered block vector
    state_blocks = [(@SVector randn(rng, Dx)) for _ in 1:(metric.K)]
    param_block = @SVector randn(rng, Dp)
    Z = BorderedBlockVector{T,Dx,Dp}(state_blocks, param_block)

    # Transform to r ~ N(0, G)
    r_blocks = G_chol.factors' * Z
    return from_bordered_block_vector(r_blocks)
end

function AdvancedHMC.neg_energy(
    h::Hamiltonian{<:RiemannianMetric{Dx,Dy,Dp,T}}, r::V, z::V
) where {V<:AbstractVecOrMat,Dx,Dy,Dp,T}
    metric = h.metric

    G = calc_G_from_metric(metric, z)
    G_chol = cholesky(G)

    logZ = T(0.5) * (length(z) * log(2π) + logdet(G_chol))
    w = G_chol.factors' \ to_bordered_block_vector(r, Val(Dx), Val(Dp))
    rTG_inv_r = sum(abs2, from_bordered_block_vector(w))

    return -logZ - rTG_inv_r / 2
end

function AdvancedHMC.∂H∂θ(
    h::Hamiltonian{<:RiemannianMetric{Dx,Dy,Dp,T}},
    z::AbstractVecOrMat{T},
    r::AbstractVecOrMat{T},
) where {Dx,Dy,Dp,T}
    return AdvancedHMC.∂H∂θ_cache(h, z, r)
end

function AdvancedHMC.∂H∂θ_cache(
    h::Hamiltonian{<:RiemannianMetric{Dx,Dy,Dp,T}},
    z::AbstractVecOrMat{T},
    r::AbstractVecOrMat{T};
    return_cache=false,
    cache=nothing,
) where {Dx,Dy,Dp,T}
    metric = h.metric

    if isnothing(cache)
        z_blocks = to_bordered_block_vector(z, Val(Dx), Val(Dp))

        ℓπ = calc_loglikelihood(
            z_blocks,
            metric.model,
            metric.ys,
            metric.prior_mean,
            metric.prior_prec;
            obs_indices=metric.obs_indices,
        )

        ∂ℓπ∂z = calc_loglikelihood_grad(
            z_blocks,
            metric.model,
            metric.ys,
            metric.prior_mean,
            metric.prior_prec;
            obs_indices=metric.obs_indices,
        )

        G = calc_G(
            z_blocks, metric.model, metric.prior_prec; obs_indices=metric.obs_indices
        )
        G_inv_parts = bordered_block_selected_inv(G)
        G_chol = cholesky(G)

        dGs_state = calc_dGs_state(z_blocks, metric.model; obs_indices=metric.obs_indices)
        dGs_param = calc_dGs_param(z_blocks, metric.model; obs_indices=metric.obs_indices)

        # Pre-compute gradient contributions independent of r
        cached_grad = Vector{T}(undef, length(z))

        # State gradient components
        @inbounds for k in 1:(metric.K)
            for d in 1:Dx
                idx = (k - 1) * Dx + d
                v = -∂ℓπ∂z.state_blocks[k][d]

                # Trace term: (1/2) tr(G⁻¹ ∂G/∂z_k^{(d)})
                dG = dGs_state[k, d]
                v -= -T(0.5) * tr_product(G_inv_parts, dG)

                cached_grad[idx] = v
            end
        end

        # Parameter gradient components
        @inbounds for p in 1:Dp
            idx = metric.K * Dx + p
            v = -∂ℓπ∂z.param_block[p]

            dG = dGs_param[p]
            v -= -T(0.5) * tr_product(G_inv_parts, dG)

            cached_grad[idx] = v
        end
    else
        ℓπ, cached_grad, G_chol, dGs_state, dGs_param = cache
    end

    r_blocks = to_bordered_block_vector(r, Val(Dx), Val(Dp))
    G_inv_r = G_chol \ r_blocks

    grad = copy(cached_grad)

    # Quadratic form terms
    @inbounds for k in 1:(metric.K)
        for d in 1:Dx
            idx = (k - 1) * Dx + d
            dG = dGs_state[k, d]
            v = T(0.5) * quad_form(G_inv_r, dG)
            grad[idx] -= v
        end
    end

    @inbounds for p in 1:Dp
        idx = metric.K * Dx + p
        dG = dGs_param[p]
        v = T(0.5) * quad_form(G_inv_r, dG)
        grad[idx] -= v
    end

    dv = DualValue(ℓπ, grad)
    return return_cache ? (dv, (; ℓπ, cached_grad, G_chol, dGs_state, dGs_param)) : dv
end

function AdvancedHMC.∂H∂r(
    h::Hamiltonian{<:RiemannianMetric{Dx,Dy,Dp,T}}, z::AbstractVecOrMat, r::AbstractVecOrMat
) where {Dx,Dy,Dp,T}
    metric = h.metric

    G = calc_G_from_metric(metric, z)
    G_chol = cholesky(G)

    r_blocks = to_bordered_block_vector(r, Val(Dx), Val(Dp))
    result_blocks = G_chol \ r_blocks

    return from_bordered_block_vector(result_blocks)
end

# =============================================================================
# Simulation
# =============================================================================

"""
    simulate(rng, model, θ, K; obs_indices=nothing)

Simulate a trajectory from the state space model.

Returns `(states, observations)` where:
- `states` is a `Vector{SVector{Dx,T}}` of length K
- `observations` is a `Vector{SVector{Dy,T}}` of length N_obs

If `obs_indices` is `nothing`, observations are generated at every time step.
Otherwise, observations are only generated at the specified indices.
"""
function simulate(
    rng::AbstractRNG,
    model::StateSpaceModel{Dx,Dy,Dp},
    θ::SVector{Dp,T},
    K::Int;
    obs_indices::Union{Vector{Int},Nothing}=nothing,
) where {Dx,Dy,Dp,T}
    states = Vector{SVector{Dx,T}}(undef, K)

    Q = dynamics_covariance(model)
    prior = initial_prior(model)
    ef = observation_family(model)

    # Determine observation structure
    if isnothing(obs_indices)
        n_obs = K
        obs_set = nothing
    else
        n_obs = length(obs_indices)
        obs_set = Set(obs_indices)
    end

    observations = Vector{SVector{Dy,T}}(undef, n_obs)

    obs_idx = 1
    @inbounds for k in 1:K
        # State dynamics
        if k == 1
            states[k] = SVector{Dx,T}(rand(rng, prior))
        else
            μ_k = f(model, states[k - 1], θ)
            states[k] = μ_k + SVector{Dx,T}(rand(rng, MvNormal(zeros(T, Dx), Q)))
        end

        # Observation
        has_obs = isnothing(obs_set) || (k in obs_set)
        if has_obs
            η_k = η(model, states[k], θ)
            observations[obs_idx] = rand_obs(rng, ef, η_k)
            obs_idx += 1
        end
    end

    return states, observations
end

# =============================================================================
# LogDensityProblems Interface
# =============================================================================

"""
    RHMCLogDensity{Dx, Dy, Dp, T, M, YT, OI}

Log density wrapper for joint state-parameter inference with RHMC.

Implements the LogDensityProblems interface for use with AdvancedHMC. This is
the recommended way to define the target density for sampling.

# Example
```julia
model = VanDerPolModel(δt=0.1, σ_obs=0.5)
ℓπ = RHMCLogDensity(model, ys, K;
    prior_mean=@SVector([0.0]),
    prior_var=@SVector([4.0]),
    obs_indices=obs_indices
)
adv_model = AdvancedHMC.LogDensityModel(ℓπ)
```
"""
struct RHMCLogDensity{Dx,Dy,Dp,T,M<:StateSpaceModel{Dx,Dy,Dp},YT,OI}
    model::M
    ys::YT
    K::Int
    prior_mean::SVector{Dp,T}
    prior_prec::SVector{Dp,T}
    obs_indices::OI
end

function RHMCLogDensity(
    model::StateSpaceModel{Dx,Dy,Dp},
    ys,
    K::Int;
    prior_mean::AbstractVector{T},
    prior_var::AbstractVector{T},
    obs_indices::Union{Vector{Int},Nothing}=nothing,
) where {Dx,Dy,Dp,T}
    pm = SVector{Dp,T}(prior_mean)
    pp = SVector{Dp,T}(1 ./ prior_var)
    return RHMCLogDensity{Dx,Dy,Dp,T,typeof(model),typeof(ys),typeof(obs_indices)}(
        model, ys, K, pm, pp, obs_indices
    )
end

LogDensityProblems.dimension(p::RHMCLogDensity{Dx,Dy,Dp}) where {Dx,Dy,Dp} = p.K * Dx + Dp

function LogDensityProblems.capabilities(::Type{<:RHMCLogDensity})
    return LogDensityProblems.LogDensityOrder{1}()
end

function LogDensityProblems.logdensity(p::RHMCLogDensity{Dx,Dy,Dp}, θ) where {Dx,Dy,Dp}
    z = to_bordered_block_vector(θ, Val(Dx), Val(Dp))
    return calc_loglikelihood(
        z, p.model, p.ys, p.prior_mean, p.prior_prec; obs_indices=p.obs_indices
    )
end

function LogDensityProblems.logdensity_and_gradient(
    p::RHMCLogDensity{Dx,Dy,Dp}, θ
) where {Dx,Dy,Dp}
    z = to_bordered_block_vector(θ, Val(Dx), Val(Dp))
    ll = calc_loglikelihood(
        z, p.model, p.ys, p.prior_mean, p.prior_prec; obs_indices=p.obs_indices
    )
    grad = calc_loglikelihood_grad(
        z, p.model, p.ys, p.prior_mean, p.prior_prec; obs_indices=p.obs_indices
    )
    return ll, from_bordered_block_vector(grad)
end
