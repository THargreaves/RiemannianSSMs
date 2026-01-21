"""
Kleppe's observed Hessian metric for RMHMC (arXiv:1612.04093v2).

This implements the modified Cholesky approach using the observed (negative) Hessian
of the log-posterior, rather than the Fisher/GGN approximation.

Hardcoded for: VanDerPolDynamicsParam + PartialLinearObservationParam

Uses the MCRHMC submodule for modified Cholesky factorization and its derivatives.
"""

using AdvancedHMC
using Distributions

import AdvancedHMC: DualValue, get_regularization_sensitivities

export ObservedHessianMetric
export calc_observed_hessian, calc_observed_hessian_derivs
export calc_ll_observed, calc_ll_grad_observed

# ============================================================================
# Metric Type
# ============================================================================

mutable struct ObservedHessianMetric{T,D,P,SSM,YT,OI} <:
               AdvancedHMC.AbstractRiemannianMetric
    ssm::SSM
    ys::YT
    obs_indices::OI
    K::Int                     # Total number of latent states
    prior_mean::Vector{T}      # Prior mean for parameters
    prior_prec::Vector{T}      # Prior precision for parameters
    u::Vector{T}               # Regularization parameters for modified Cholesky
    symbolic::Any              # Cached symbolic factorization (or nothing)
    last_factorization::Union{Nothing,MCRHMC.SparseLDLFactor{T,Int}}  # Cached factorization for adaptor
end

function ObservedHessianMetric(
    ssm,
    ys,
    ::Val{D},
    ::Val{P},
    K::Int,
    prior_mean,
    prior_var,
    obs_indices::Vector{Int},
    u::Vector{T},
) where {T,D,P}
    prior_prec = 1 ./ prior_var
    n = K * D + P
    @assert length(u) == n "Regularization vector u must have length $n, got $(length(u))"
    return ObservedHessianMetric{T,D,P,typeof(ssm),typeof(ys),Vector{Int}}(
        ssm,
        ys,
        obs_indices,
        K,
        collect(prior_mean),
        collect(prior_prec),
        u,
        nothing,
        nothing,
    )
end

Base.size(m::ObservedHessianMetric{T,D,P}) where {T,D,P} = (m.K * D + P,)
Base.size(m::ObservedHessianMetric{T,D,P}, ::Int) where {T,D,P} = m.K * D + P
Base.eltype(::ObservedHessianMetric{T}) where {T} = T

function Base.show(io::IO, ::Type{<:ObservedHessianMetric{T,D,P}}) where {T,D,P}
    return print(io, "ObservedHessianMetric{$T,$D,$P}")
end

function Base.show(io::IO, m::ObservedHessianMetric{T,D,P}) where {T,D,P}
    return print(io, "ObservedHessianMetric(K=$(m.K), D=$D, P=$P)")
end

# ============================================================================
# Log-likelihood and gradient (same as before, but using plain vectors)
# ============================================================================

function calc_ll_observed(
    z::AbstractVector{T},
    ys,
    ssm,
    K::Int,
    ::Val{D},
    ::Val{P},
    prior_mean::Vector{T},
    prior_prec::Vector{T},
    obs_indices,
) where {T,D,P}
    # Extract parameter
    θ = @SVector [z[K * D + 1]]

    # Parameter prior
    ll = -0.5 * prior_prec[1] * (θ[1] - prior_mean[1])^2

    # State prior (k=1)
    z1 = @SVector [z[1], z[2]]
    ll += logpdf(ssm.prior, z1)

    # Dynamics (k=2:K)
    Q = calc_Q_param(ssm.dyn)
    @inbounds for k in 2:K
        idx = (k - 1) * D
        z_k = @SVector [z[idx + 1], z[idx + 2]]
        z_km1 = @SVector [z[idx - D + 1], z[idx - D + 2]]
        μ = f_param(ssm.dyn, z_km1, θ)
        ll += logpdf(MvNormal(μ, Q), z_k)
    end

    # Observations
    R = calc_R_param(ssm.sensor)
    @inbounds for (obs_idx, k) in enumerate(obs_indices)
        idx = (k - 1) * D
        z_k = @SVector [z[idx + 1], z[idx + 2]]
        μ = h_param(ssm.sensor, z_k, θ)
        ll += logpdf(MvNormal(μ, R), ys[obs_idx])
    end

    return ll
end

function calc_ll_grad_observed(
    z::AbstractVector{T},
    ys,
    ssm,
    K::Int,
    ::Val{D},
    ::Val{P},
    prior_mean::Vector{T},
    prior_prec::Vector{T},
    obs_indices,
) where {T,D,P}
    n = K * D + P
    grad = zeros(T, n)

    θ = @SVector [z[K * D + 1]]

    # Parameter prior gradient
    grad[K * D + 1] = -prior_prec[1] * (θ[1] - prior_mean[1])

    Q_inv = calc_Qinv_param(ssm.dyn)
    R_inv = calc_Rinv_param(ssm.sensor)

    # State prior gradient (k=1)
    z1 = @SVector [z[1], z[2]]
    Σ0_inv = inv(ssm.prior.Σ)
    g1 = -Σ0_inv * (z1 - ssm.prior.μ)
    grad[1] = g1[1]
    grad[2] = g1[2]

    # Incoming dynamics gradients (k=2:K)
    @inbounds for k in 2:K
        idx = (k - 1) * D
        z_k = @SVector [z[idx + 1], z[idx + 2]]
        z_km1 = @SVector [z[idx - D + 1], z[idx - D + 2]]
        μ = f_param(ssm.dyn, z_km1, θ)
        g_k = -Q_inv * (z_k - μ)
        grad[idx + 1] += g_k[1]
        grad[idx + 2] += g_k[2]
    end

    # Outgoing dynamics gradients (k=1:K-1)
    @inbounds for k in 1:(K - 1)
        idx = (k - 1) * D
        z_k = @SVector [z[idx + 1], z[idx + 2]]
        z_kp1 = @SVector [z[idx + D + 1], z[idx + D + 2]]
        μ = f_param(ssm.dyn, z_k, θ)
        Jf = calc_Jf_param(ssm.dyn, z_k, θ)
        residual = z_kp1 - μ
        g_k = Jf' * Q_inv * residual
        grad[idx + 1] += g_k[1]
        grad[idx + 2] += g_k[2]

        # Parameter gradient from dynamics
        f_θ = calc_f_θ(ssm.dyn, z_k, θ)
        grad[K * D + 1] += (f_θ' * Q_inv * residual)[1]
    end

    # Observation gradients
    @inbounds for (obs_idx, k) in enumerate(obs_indices)
        idx = (k - 1) * D
        z_k = @SVector [z[idx + 1], z[idx + 2]]
        μ = h_param(ssm.sensor, z_k, θ)
        Jh = calc_Jh_param(ssm.sensor, z_k, θ)
        residual = ys[obs_idx] - μ
        g_k = Jh' * R_inv * residual
        grad[idx + 1] += g_k[1]
        grad[idx + 2] += g_k[2]
    end

    return grad
end

# ============================================================================
# Observed Hessian Computation (Sparse)
# ============================================================================

"""
    calc_observed_hessian(z, ssm, K, D, P, prior_prec, obs_indices)

Compute the negative Hessian of the log-posterior as a sparse matrix.

This is the OBSERVED Hessian, which includes second-order terms from nonlinear dynamics:
    -∂²log p(z_{k}|z_{k-1}, θ)/∂z_{k-1}² = Jf' Q⁻¹ Jf + Σ_d [Q⁻¹ r_k]_d H_f^d

where r_k = z_k - f(z_{k-1}, θ) is the residual.
"""
function calc_observed_hessian(
    z::AbstractVector{T},
    ssm,
    K::Int,
    ::Val{D},
    ::Val{P},
    prior_prec::Vector{T},
    obs_indices,
) where {T,D,P}
    n = K * D + P

    # Build sparse matrix using COO format then convert
    I_idx = Int[]
    J_idx = Int[]
    V_val = T[]

    θ = @SVector [z[K * D + 1]]

    Q_inv = calc_Qinv_param(ssm.dyn)
    R_inv = calc_Rinv_param(ssm.sensor)

    obs_set = Set(obs_indices)

    # ----- STATE BLOCKS -----

    @inbounds for k in 1:K
        idx_base = (k - 1) * D
        z_k = @SVector [z[idx_base + 1], z[idx_base + 2]]

        # Initialize diagonal block for state k
        H_kk = zeros(SMatrix{D,D,T})

        # Prior term (k=1) or incoming dynamics term (k>1)
        if k == 1
            H_kk += SMatrix{D,D,T}(inv(ssm.prior.Σ))
        else
            # Incoming dynamics: -∂²/∂z_k² = Q⁻¹
            H_kk += SMatrix{D,D,T}(Q_inv)
        end

        # Observation term (only if observed)
        if k in obs_set
            Jh = calc_Jh_param(ssm.sensor, z_k, θ)
            # For partial linear observation, Hh = 0, so just Jh' R⁻¹ Jh
            H_kk += Jh' * R_inv * Jh

            # Should normally have the Hessian correction here, but it's zero for linear h
        end

        # Outgoing dynamics term (k < K)
        if k < K
            z_kp1 = @SVector [z[idx_base + D + 1], z[idx_base + D + 2]]
            μ = f_param(ssm.dyn, z_k, θ)
            residual = z_kp1 - μ
            Jf = calc_Jf_param(ssm.dyn, z_k, θ)
            Hfs = calc_dyn_hessians(ssm.dyn, z_k, θ)

            # GGN term: Jf' Q⁻¹ Jf
            H_kk += Jf' * Q_inv * Jf

            # Observed Hessian correction: -Σ_d [Q⁻¹ r]_d ∇²f_d
            Q_inv_r = Q_inv * residual
            for d in 1:D
                H_kk -= Q_inv_r[d] * Hfs[d]
            end
        end

        # Add diagonal block entries
        for i in 1:D
            for j in 1:D
                push!(I_idx, idx_base + i)
                push!(J_idx, idx_base + j)
                push!(V_val, H_kk[i, j])
            end
        end

        # Off-diagonal block G_{k, k+1} (super-diagonal)
        if k < K
            z_kp1 = @SVector [z[idx_base + D + 1], z[idx_base + D + 2]]
            Jf = calc_Jf_param(ssm.dyn, z_k, θ)
            H_k_kp1 = -Jf' * Q_inv

            for i in 1:D
                for j in 1:D
                    # Upper triangle entry
                    push!(I_idx, idx_base + i)
                    push!(J_idx, idx_base + D + j)
                    push!(V_val, H_k_kp1[i, j])
                    # Lower triangle entry (symmetric)
                    push!(I_idx, idx_base + D + j)
                    push!(J_idx, idx_base + i)
                    push!(V_val, H_k_kp1[i, j])
                end
            end
        end
    end

    # ----- BORDER BLOCKS (state-parameter coupling) -----

    @inbounds for k in 1:K
        idx_base = (k - 1) * D
        z_k = @SVector [z[idx_base + 1], z[idx_base + 2]]

        H_k_θ = zeros(SMatrix{D,P,T})

        # Incoming dynamics term (k > 1): -Q⁻¹ f_θ
        # (negative because we compute -∇²ℓ, and ∂²ℓ/(∂x_k ∂θ) = Q⁻¹ f_θ)
        if k > 1
            z_km1 = @SVector [z[idx_base - D + 1], z[idx_base - D + 2]]
            f_θ = calc_f_θ(ssm.dyn, z_km1, θ)
            H_k_θ -= SMatrix{D,P,T}(Q_inv * f_θ)
        end

        # Outgoing dynamics term (k < K): ∂²/∂z_k∂θ from Jf'Q⁻¹f_θ + Hf_θ terms
        if k < K
            z_kp1 = @SVector [z[idx_base + D + 1], z[idx_base + D + 2]]
            μ = f_param(ssm.dyn, z_k, θ)
            residual = z_kp1 - μ
            Jf = calc_Jf_param(ssm.dyn, z_k, θ)
            f_θ = calc_f_θ(ssm.dyn, z_k, θ)
            mixed_hess = calc_mixed_hessians(ssm.dyn, z_k, θ)

            # GGN term
            H_k_θ += Jf' * Q_inv * f_θ

            # Observed Hessian correction: -Σ_i [Q⁻¹ r]_i ∇²_{z,θ} f_i
            Q_inv_r = Q_inv * residual
            for i in 1:D
                H_k_θ -= Q_inv_r[i] * mixed_hess[i]
            end
        end

        # Add border block entries (both directions for symmetry)
        for i in 1:D
            for p in 1:P
                push!(I_idx, idx_base + i)
                push!(J_idx, K * D + p)
                push!(V_val, H_k_θ[i, p])
                push!(I_idx, K * D + p)
                push!(J_idx, idx_base + i)
                push!(V_val, H_k_θ[i, p])
            end
        end
    end

    # ----- CORNER BLOCK (parameter-parameter) -----

    H_θθ = zeros(SMatrix{P,P,T})

    # Parameter prior
    H_θθ += Diagonal(SVector{P,T}(prior_prec))

    # Dynamics contributions
    @inbounds for k in 1:(K - 1)
        idx_base = (k - 1) * D
        z_k = @SVector [z[idx_base + 1], z[idx_base + 2]]
        z_kp1 = @SVector [z[idx_base + D + 1], z[idx_base + D + 2]]
        μ = f_param(ssm.dyn, z_k, θ)
        residual = z_kp1 - μ

        f_θ = calc_f_θ(ssm.dyn, z_k, θ)
        f_θθ = calc_f_θθ(ssm.dyn, z_k, θ)

        # GGN term
        H_θθ += f_θ' * Q_inv * f_θ

        # Observed Hessian correction: -Σ_d (Q⁻¹ r)_d × ∂²f^{(d)}/∂θ∂θ'
        Q_inv_r = Q_inv * residual
        for p in 1:P
            # f_θθ[p] is D×P matrix: [d,q] = ∂²f^{(d)}/∂θ_p∂θ_q
            # Q_inv_r' * f_θθ[p] gives 1×P contribution to row p
            H_θθ -= Q_inv_r' * f_θθ[p]
        end
    end

    # Add corner block entries
    for i in 1:P
        for j in 1:P
            push!(I_idx, K * D + i)
            push!(J_idx, K * D + j)
            push!(V_val, H_θθ[i, j])
        end
    end

    # Build sparse matrix
    G = sparse(I_idx, J_idx, V_val, n, n)

    return G
end

# ============================================================================
# Hessian Derivatives (for dH/dθ computation)
# ============================================================================

"""
    calc_observed_hessian_derivs(z, ssm, K, D, P, obs_indices)

Compute derivatives of the observed Hessian w.r.t. each component of z.

Returns a vector of sparse matrices ∂G/∂z_i for i = 1:n.
"""
function calc_observed_hessian_derivs(
    z::AbstractVector{T}, ssm, K::Int, vD::Val{D}, vP::Val{P}, obs_indices
) where {T,D,P}
    n = K * D + P
    dGs = Vector{SparseMatrixCSC{T,Int}}(undef, n)

    θ = @SVector [z[K * D + 1]]

    Q_inv = calc_Qinv_param(ssm.dyn)
    R_inv = calc_Rinv_param(ssm.sensor)

    obs_set = Set(obs_indices)

    # Derivatives w.r.t. state components
    @inbounds for k in 1:K
        for d in 1:D
            idx = (k - 1) * D + d
            dGs[idx] = calc_dG_dz_kd(z, ssm, K, vD, vP, k, d, θ, Q_inv, R_inv, obs_set)
        end
    end

    # Derivatives w.r.t. parameter components
    @inbounds for p in 1:P
        idx = K * D + p
        dGs[idx] = calc_dG_dθ_p(z, ssm, K, vD, vP, p, θ, Q_inv, R_inv, obs_set)
    end

    return dGs
end

"""
Compute ∂G/∂z_k^{(d)} - derivative of Hessian w.r.t. state component d at time k.

Following the notes in "Joint State-Parameter Hessian (3rd Deriv).md":
- G is the NEGATIVE Hessian of log p(x,θ|y)
- G_{k,k} from outgoing dynamics contains: Jf' Q⁻¹ Jf - Σ_i (Q⁻¹r)_i × ∇²f_i
- G_{k,k+1} = -Jf' Q⁻¹
"""
function calc_dG_dz_kd(
    z::AbstractVector{T},
    ssm,
    K::Int,
    ::Val{D},
    ::Val{P},
    k::Int,
    d::Int,
    θ,
    Q_inv,
    R_inv,
    obs_set,
) where {T,D,P}
    n = K * D + P
    I_idx = Int[]
    J_idx = Int[]
    V_val = T[]

    idx_base = (k - 1) * D
    z_k = @SVector [z[idx_base + 1], z[idx_base + 2]]

    # ===== STATE-STATE BLOCKS =====

    # ----- Effect on G_{k,k} from outgoing dynamics (transition k → k+1) -----
    if k < K
        z_kp1 = @SVector [z[idx_base + D + 1], z[idx_base + D + 2]]
        μ = f_param(ssm.dyn, z_k, θ)
        residual = z_kp1 - μ
        Jf = calc_Jf_param(ssm.dyn, z_k, θ)
        Hfs = calc_Hfs_param(ssm.dyn, z_k, θ)  # ∂Jf/∂z_d for GGN term
        dyn_hess = calc_dyn_hessians(ssm.dyn, z_k, θ)  # ∇²f_i for correction term
        dyn_hess_derivs = calc_dyn_hessian_derivs(ssm.dyn, z_k, θ)  # ∂(∇²f_i)/∂z_c

        # 1. GGN term: ∂(Jf' Q⁻¹ Jf)/∂z_k^{(d)} = Hfs[d]' Q⁻¹ Jf + Jf' Q⁻¹ Hfs[d]
        Mf = Q_inv * Jf
        dH_kk = Hfs[d]' * Mf + Mf' * Hfs[d]

        # 2. Observed correction term derivative:
        #    G has: -Σ_i (Q⁻¹r)_i × ∇²f_i
        #    ∂/∂z_k^{(d)}[-Σ_i (Q⁻¹r)_i × ∇²f_i]
        #      = -Σ_i (Q⁻¹ × ∂r/∂z_k^{(d)})_i × ∇²f_i - Σ_i (Q⁻¹r)_i × ∂(∇²f_i)/∂z_k^{(d)}
        #    where ∂r/∂z_k^{(d)} = -Jf[:,d]
        #      = +Σ_i (Q⁻¹ Jf[:,d])_i × ∇²f_i - Σ_i (Q⁻¹r)_i × ∂(∇²f_i)/∂z_k^{(d)}

        Q_inv_Jf_col = Q_inv * Jf[:, d]
        for i in 1:D
            dH_kk += Q_inv_Jf_col[i] * dyn_hess[i]  # First term (weight change)
        end

        Q_inv_r = Q_inv * residual
        for i in 1:D
            dH_kk -= Q_inv_r[i] * dyn_hess_derivs[i][d]  # Second term (third derivatives)
        end

        for i in 1:D
            for j in 1:D
                if abs(dH_kk[i, j]) > 1e-15
                    push!(I_idx, idx_base + i)
                    push!(J_idx, idx_base + j)
                    push!(V_val, dH_kk[i, j])
                end
            end
        end

        # ----- Effect on G_{k,k+1} -----
        # G_{k,k+1} = -Jf' Q⁻¹
        # ∂G_{k,k+1}/∂z_k^{(d)} = -(∂Jf/∂z_k^{(d)})' Q⁻¹ = -Hfs[d]' Q⁻¹
        dH_k_kp1 = -Hfs[d]' * Q_inv

        for i in 1:D
            for j in 1:D
                if abs(dH_k_kp1[i, j]) > 1e-15
                    push!(I_idx, idx_base + i)
                    push!(J_idx, idx_base + D + j)
                    push!(V_val, dH_k_kp1[i, j])
                    push!(I_idx, idx_base + D + j)
                    push!(J_idx, idx_base + i)
                    push!(V_val, dH_k_kp1[i, j])
                end
            end
        end
    end

    # ----- Effect on G_{k-1,k-1} from incoming dynamics (transition k-1 → k) -----
    if k > 1
        # z_k appears in residual r_k = z_k - f(z_{k-1}, θ)
        # G_{k-1,k-1} contains: -Σ_i (Q⁻¹r_k)_i × ∇²f_i|_{k-1}
        # ∂/∂z_k^{(d)}[-Σ_i (Q⁻¹r_k)_i × ∇²f_i] = -Σ_i (Q⁻¹ e_d)_i × ∇²f_i
        # (since ∂r_k/∂z_k^{(d)} = e_d and ∇²f_i doesn't depend on z_k)

        idx_base_km1 = (k - 2) * D
        z_km1 = @SVector [z[idx_base_km1 + 1], z[idx_base_km1 + 2]]
        dyn_hess_km1 = calc_dyn_hessians(ssm.dyn, z_km1, θ)

        e_d = zeros(SVector{D,T})
        e_d = setindex(e_d, one(T), d)
        Q_inv_ed = Q_inv * e_d

        dH_km1_km1 = zeros(SMatrix{D,D,T})
        for i in 1:D
            dH_km1_km1 -= Q_inv_ed[i] * dyn_hess_km1[i]
        end

        for i in 1:D
            for j in 1:D
                if abs(dH_km1_km1[i, j]) > 1e-15
                    push!(I_idx, idx_base_km1 + i)
                    push!(J_idx, idx_base_km1 + j)
                    push!(V_val, dH_km1_km1[i, j])
                end
            end
        end
    end

    # ===== BORDER BLOCKS (state-parameter) =====

    # ----- Effect on G_{k-1,θ} from outgoing dynamics at k-1 (via residual r_k) -----
    if k > 1
        idx_base_km1 = (k - 2) * D
        z_km1 = @SVector [z[idx_base_km1 + 1], z[idx_base_km1 + 2]]
        mixed_hess_km1 = calc_mixed_hessians(ssm.dyn, z_km1, θ)

        e_d = zeros(SVector{D,T})
        e_d = setindex(e_d, one(T), d)
        Q_inv_ed = Q_inv * e_d

        # G_{k-1,θ} from outgoing dynamics at k-1 contains: -Σ_i (Q⁻¹r_k)_i × ∇²_{z,θ}f_i|_{k-1}
        # where r_k = z_k - f(z_{k-1},θ). Taking ∂/∂z_k^{(d)} with ∂r_k/∂z_k^{(d)} = e_d:
        # ∂/∂z_k^{(d)} = -Σ_i (Q⁻¹ e_d)_i × ∇²_{z,θ}f_i|_{k-1}
        dH_km1_θ = zeros(SMatrix{D,P,T})
        for i in 1:D
            dH_km1_θ -= Q_inv_ed[i] * mixed_hess_km1[i]
        end

        for i in 1:D
            for p in 1:P
                if abs(dH_km1_θ[i, p]) > 1e-15
                    push!(I_idx, idx_base_km1 + i)
                    push!(J_idx, K * D + p)
                    push!(V_val, dH_km1_θ[i, p])
                    push!(I_idx, K * D + p)
                    push!(J_idx, idx_base_km1 + i)
                    push!(V_val, dH_km1_θ[i, p])
                end
            end
        end
    end

    # ----- Effect on G_{k+1,θ} from incoming dynamics at k+1 -----
    if k < K
        # G_{k+1,θ} from incoming dynamics contains: -Q⁻¹ f_θ|_k
        # where f_θ is evaluated at z_k.
        # ∂/∂z_k^{(d)} = -Q⁻¹ × Hf_θs[d]|_k
        Hf_θs = calc_Hf_θs(ssm.dyn, z_k, θ)
        idx_base_kp1 = k * D

        dH_kp1_θ = -Q_inv * Hf_θs[d]

        for i in 1:D
            for p in 1:P
                if abs(dH_kp1_θ[i, p]) > 1e-15
                    push!(I_idx, idx_base_kp1 + i)
                    push!(J_idx, K * D + p)
                    push!(V_val, dH_kp1_θ[i, p])
                    push!(I_idx, K * D + p)
                    push!(J_idx, idx_base_kp1 + i)
                    push!(V_val, dH_kp1_θ[i, p])
                end
            end
        end
    end

    # ----- Effect on G_{k,θ} from outgoing dynamics -----
    if k < K
        z_kp1 = @SVector [z[idx_base + D + 1], z[idx_base + D + 2]]
        μ = f_param(ssm.dyn, z_k, θ)
        residual = z_kp1 - μ
        Jf = calc_Jf_param(ssm.dyn, z_k, θ)
        f_θ = calc_f_θ(ssm.dyn, z_k, θ)
        Hfs = calc_Hfs_param(ssm.dyn, z_k, θ)
        Hf_θs = calc_Hf_θs(ssm.dyn, z_k, θ)
        mixed_hess = calc_mixed_hessians(ssm.dyn, z_k, θ)

        # G_{k,θ} from outgoing dynamics contains:
        #   Jf' Q⁻¹ f_θ - Σ_i (Q⁻¹r)_i × ∇²_{z,θ}f_i

        # 1. GGN term: ∂(Jf' Q⁻¹ f_θ)/∂z_k^{(d)} = Hfs[d]' Q⁻¹ f_θ + Jf' Q⁻¹ Hf_θs[d]
        dH_k_θ = Hfs[d]' * Q_inv * f_θ + Jf' * Q_inv * Hf_θs[d]

        # 2. Correction term: ∂[-Σ_i (Q⁻¹r)_i × ∇²_{z,θ}f_i]/∂z_k^{(d)}
        #    = +Σ_i (Q⁻¹ Jf[:,d])_i × ∇²_{z,θ}f_i - Σ_i (Q⁻¹r)_i × ∂(∇²_{z,θ}f_i)/∂z_k^{(d)}
        Q_inv_Jf_col = Q_inv * Jf[:, d]
        Q_inv_r = Q_inv * residual
        mixed_hess_derivs = calc_mixed_hessian_derivs(ssm.dyn, z_k, θ)
        for i in 1:D
            dH_k_θ += Q_inv_Jf_col[i] * mixed_hess[i]
            dH_k_θ -= Q_inv_r[i] * mixed_hess_derivs[i][d]
        end

        for i in 1:D
            for p in 1:P
                if abs(dH_k_θ[i, p]) > 1e-15
                    push!(I_idx, idx_base + i)
                    push!(J_idx, K * D + p)
                    push!(V_val, dH_k_θ[i, p])
                    push!(I_idx, K * D + p)
                    push!(J_idx, idx_base + i)
                    push!(V_val, dH_k_θ[i, p])
                end
            end
        end
    end

    # ===== CORNER BLOCK (parameter-parameter) =====
    dH_θθ = zeros(SMatrix{P,P,T})

    # ----- Effect from outgoing dynamics at k -----
    if k < K
        z_kp1 = @SVector [z[idx_base + D + 1], z[idx_base + D + 2]]
        μ = f_param(ssm.dyn, z_k, θ)
        residual = z_kp1 - μ
        Jf = calc_Jf_param(ssm.dyn, z_k, θ)
        f_θ = calc_f_θ(ssm.dyn, z_k, θ)
        f_θθ = calc_f_θθ(ssm.dyn, z_k, θ)
        Hf_θs = calc_Hf_θs(ssm.dyn, z_k, θ)

        # G_{θ,θ} contains: f_θ' Q⁻¹ f_θ - Σ_i (Q⁻¹r)_i × f_θθ[i]

        # 1. GGN term: ∂(f_θ' Q⁻¹ f_θ)/∂z_k^{(d)} = Hf_θs[d]' Q⁻¹ f_θ + f_θ' Q⁻¹ Hf_θs[d]
        dH_θθ += Hf_θs[d]' * Q_inv * f_θ + f_θ' * Q_inv * Hf_θs[d]

        # 2. Correction term: ∂[-Σ_i (Q⁻¹r)_i × f_θθ[i]]/∂z_k^{(d)}
        #    = +Σ_i (Q⁻¹ Jf[:,d])_i × f_θθ[i] - Σ_i (Q⁻¹r)_i × ∂f_θθ[i]/∂z_k^{(d)}
        #    For VDP: ∂f_θθ/∂z = Hf_θs (since f_θθ = f_θ)
        Q_inv_Jf_col = Q_inv * Jf[:, d]
        Q_inv_r = Q_inv * residual
        for i in 1:D
            f_θθ_i = @SMatrix [f_θθ[1][i, 1]]  # Extract scalar for P=1
            dH_θθ += Q_inv_Jf_col[i] * f_θθ_i
            # Third derivative: ∂f_θθ[i]/∂z_k^{(d)} = Hf_θs[d][i,:]
            df_θθ_i = @SMatrix [Hf_θs[d][i, 1]]
            dH_θθ -= Q_inv_r[i] * df_θθ_i
        end
    end

    # ----- Effect from incoming dynamics at k (transition k-1 → k) -----
    if k > 1
        idx_base_km1 = (k - 2) * D
        z_km1 = @SVector [z[idx_base_km1 + 1], z[idx_base_km1 + 2]]
        f_θθ_km1 = calc_f_θθ(ssm.dyn, z_km1, θ)

        e_d = zeros(SVector{D,T})
        e_d = setindex(e_d, one(T), d)
        Q_inv_ed = Q_inv * e_d

        # G_{θ,θ} from incoming dynamics contains: -Σ_i (Q⁻¹r_k)_i × f_θθ[i]
        # ∂/∂z_k^{(d)} = -Σ_i (Q⁻¹ e_d)_i × f_θθ[i]
        for i in 1:D
            f_θθ_i = @SMatrix [f_θθ_km1[1][i, 1]]
            dH_θθ -= Q_inv_ed[i] * f_θθ_i
        end
    end

    for i in 1:P
        for j in 1:P
            if abs(dH_θθ[i, j]) > 1e-15
                push!(I_idx, K * D + i)
                push!(J_idx, K * D + j)
                push!(V_val, dH_θθ[i, j])
            end
        end
    end

    return sparse(I_idx, J_idx, V_val, n, n)
end

"""
Compute ∂G/∂θ^{(p)} - derivative of Hessian w.r.t. parameter component p.

Following the same pattern as calc_dG_dz_kd:
- G is the NEGATIVE Hessian of log p(x,θ|y)
- G_{k,k} from outgoing dynamics contains: Jf' Q⁻¹ Jf - Σ_i (Q⁻¹r)_i × ∇²f_i
- For VDP with log-parameterization: ∂(∇²f_i)/∂θ = ∇²f_i (third derivatives)
"""
function calc_dG_dθ_p(
    z::AbstractVector{T}, ssm, K::Int, ::Val{D}, ::Val{P}, p::Int, θ, Q_inv, R_inv, obs_set
) where {T,D,P}
    n = K * D + P
    I_idx = Int[]
    J_idx = Int[]
    V_val = T[]

    @inbounds for k in 1:K
        idx_base = (k - 1) * D
        z_k = @SVector [z[idx_base + 1], z[idx_base + 2]]

        # ===== STATE-STATE BLOCKS =====

        # ----- Effect on G_{k,k} from outgoing dynamics -----
        dH_kk = zeros(SMatrix{D,D,T})

        if k < K
            z_kp1 = @SVector [z[idx_base + D + 1], z[idx_base + D + 2]]
            μ = f_param(ssm.dyn, z_k, θ)
            residual = z_kp1 - μ
            Jf = calc_Jf_param(ssm.dyn, z_k, θ)
            Jf_θ = calc_Jf_θ(ssm.dyn, z_k, θ)
            f_θ = calc_f_θ(ssm.dyn, z_k, θ)
            dyn_hess = calc_dyn_hessians(ssm.dyn, z_k, θ)

            # 1. GGN term: ∂(Jf' Q⁻¹ Jf)/∂θ_p = Jf_θ[p]' Q⁻¹ Jf + Jf' Q⁻¹ Jf_θ[p]
            Mf = Q_inv * Jf
            dH_kk += Jf_θ[p]' * Mf + Mf' * Jf_θ[p]

            # 2. Correction term: ∂[-Σ_i (Q⁻¹r)_i × ∇²f_i]/∂θ_p
            #    = +Σ_i (Q⁻¹ f_θ[:,p])_i × ∇²f_i - Σ_i (Q⁻¹r)_i × (∂∇²f_i/∂θ_p)
            #    For VDP: ∂(∇²f_i)/∂θ = ∇²f_i (log-parameterization)
            Q_inv_f_θ = Q_inv * f_θ[:, p]
            Q_inv_r = Q_inv * residual
            for i in 1:D
                dH_kk += Q_inv_f_θ[i] * dyn_hess[i]
                dH_kk -= Q_inv_r[i] * dyn_hess[i]  # Third derivative: ∂(∇²f_i)/∂θ = ∇²f_i
            end
        end

        for i in 1:D
            for j in 1:D
                if abs(dH_kk[i, j]) > 1e-15
                    push!(I_idx, idx_base + i)
                    push!(J_idx, idx_base + j)
                    push!(V_val, dH_kk[i, j])
                end
            end
        end

        # ----- Effect on G_{k,k+1} -----
        if k < K
            Jf_θ = calc_Jf_θ(ssm.dyn, z_k, θ)
            # G_{k,k+1} = -Jf' Q⁻¹
            # ∂G_{k,k+1}/∂θ_p = -Jf_θ[p]' Q⁻¹
            dH_k_kp1 = -Jf_θ[p]' * Q_inv

            for i in 1:D
                for j in 1:D
                    if abs(dH_k_kp1[i, j]) > 1e-15
                        push!(I_idx, idx_base + i)
                        push!(J_idx, idx_base + D + j)
                        push!(V_val, dH_k_kp1[i, j])
                        push!(I_idx, idx_base + D + j)
                        push!(J_idx, idx_base + i)
                        push!(V_val, dH_k_kp1[i, j])
                    end
                end
            end
        end

        # ===== BORDER BLOCKS (state-parameter) =====

        dH_k_θ = zeros(SMatrix{D,P,T})

        # ----- Effect on G_{k,θ} from incoming dynamics -----
        if k > 1
            z_km1 = @SVector [z[idx_base - D + 1], z[idx_base - D + 2]]
            # G_{k,θ}^{inc} = -Q⁻¹ f_θ|_{k-1}
            # ∂G_{k,θ}^{inc}/∂θ_p = -Q⁻¹ f_θθ|_{k-1}[p]
            f_θθ_km1 = calc_f_θθ(ssm.dyn, z_km1, θ)
            dH_k_θ += -Q_inv * f_θθ_km1[p]
        end

        # ----- Effect on G_{k,θ} from outgoing dynamics -----
        if k < K
            z_kp1 = @SVector [z[idx_base + D + 1], z[idx_base + D + 2]]
            μ = f_param(ssm.dyn, z_k, θ)
            residual = z_kp1 - μ
            Jf = calc_Jf_param(ssm.dyn, z_k, θ)
            Jf_θ = calc_Jf_θ(ssm.dyn, z_k, θ)
            f_θ = calc_f_θ(ssm.dyn, z_k, θ)
            f_θθ = calc_f_θθ(ssm.dyn, z_k, θ)
            mixed_hess = calc_mixed_hessians(ssm.dyn, z_k, θ)

            # G_{k,θ} = Jf' Q⁻¹ f_θ - Σ_i (Q⁻¹r)_i × ∇²_{z,θ}f_i

            # 1. GGN term: ∂(Jf' Q⁻¹ f_θ)/∂θ_p = Jf_θ[p]' Q⁻¹ f_θ + Jf' Q⁻¹ f_θθ[p]
            dH_k_θ += Jf_θ[p]' * Q_inv * f_θ + Jf' * Q_inv * f_θθ[p]

            # 2. Correction term: ∂[-Σ_i (Q⁻¹r)_i × ∇²_{z,θ}f_i]/∂θ_p
            #    = +Σ_i (Q⁻¹ f_θ[:,p])_i × ∇²_{z,θ}f_i - Σ_i (Q⁻¹r)_i × (∂∇²_{z,θ}f_i/∂θ_p)
            #    For VDP: ∂(∇²_{z,θ}f_i)/∂θ = ∇²_{z,θ}f_i (log-parameterization)
            Q_inv_f_θ = Q_inv * f_θ[:, p]
            Q_inv_r = Q_inv * residual
            for i in 1:D
                dH_k_θ += Q_inv_f_θ[i] * mixed_hess[i]
                dH_k_θ -= Q_inv_r[i] * mixed_hess[i]  # Third derivative
            end
        end

        for i in 1:D
            for pp in 1:P
                if abs(dH_k_θ[i, pp]) > 1e-15
                    push!(I_idx, idx_base + i)
                    push!(J_idx, K * D + pp)
                    push!(V_val, dH_k_θ[i, pp])
                    push!(I_idx, K * D + pp)
                    push!(J_idx, idx_base + i)
                    push!(V_val, dH_k_θ[i, pp])
                end
            end
        end
    end

    # ===== CORNER BLOCK (parameter-parameter) =====
    dH_θθ = zeros(SMatrix{P,P,T})

    @inbounds for k in 1:(K - 1)
        idx_base = (k - 1) * D
        z_k = @SVector [z[idx_base + 1], z[idx_base + 2]]
        z_kp1 = @SVector [z[idx_base + D + 1], z[idx_base + D + 2]]
        μ = f_param(ssm.dyn, z_k, θ)
        residual = z_kp1 - μ

        f_θ = calc_f_θ(ssm.dyn, z_k, θ)
        f_θθ = calc_f_θθ(ssm.dyn, z_k, θ)

        # G_{θ,θ} = f_θ' Q⁻¹ f_θ - Σ_i (Q⁻¹r)_i × f_θθ[i]

        # 1. GGN term: ∂(f_θ' Q⁻¹ f_θ)/∂θ_p = f_θθ[p]' Q⁻¹ f_θ + f_θ' Q⁻¹ f_θθ[p]
        dH_θθ += f_θθ[p]' * Q_inv * f_θ + f_θ' * Q_inv * f_θθ[p]

        # 2. Correction term: ∂[-Σ_i (Q⁻¹r)_i × f_θθ[i]]/∂θ_p
        #    = +Σ_i (Q⁻¹ f_θ[:,p])_i × f_θθ[i] - Σ_i (Q⁻¹r)_i × (∂f_θθ[i]/∂θ_p)
        #    For VDP: ∂f_θθ/∂θ = f_θθ (log-parameterization)
        Q_inv_f_θ = Q_inv * f_θ[:, p]
        Q_inv_r = Q_inv * residual
        for i in 1:D
            f_θθ_i = @SMatrix [f_θθ[1][i, 1]]  # Extract scalar for P=1
            dH_θθ += Q_inv_f_θ[i] * f_θθ_i
            dH_θθ -= Q_inv_r[i] * f_θθ_i  # Third derivative: ∂f_θθ/∂θ = f_θθ
        end
    end

    for i in 1:P
        for j in 1:P
            if abs(dH_θθ[i, j]) > 1e-15
                push!(I_idx, K * D + i)
                push!(J_idx, K * D + j)
                push!(V_val, dH_θθ[i, j])
            end
        end
    end

    return sparse(I_idx, J_idx, V_val, n, n)
end

# ============================================================================
# AdvancedHMC Interface
# ============================================================================

function AdvancedHMC.rand_momentum(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    metric::ObservedHessianMetric{T,D,P},
    kinetic,
    z::AbstractVecOrMat,
) where {T,D,P}
    G = calc_observed_hessian(
        z, metric.ssm, metric.K, Val{D}(), Val{P}(), metric.prior_prec, metric.obs_indices
    )

    # Modified Cholesky factorization
    F = MCRHMC.modified_cholesky(G, metric.u, 0)

    # Cache factorization for adaptation
    metric.last_factorization = F

    # Sample Z ~ N(0, I)
    n = metric.K * D + P
    Z = randn(rng, n)

    # Transform to r ~ N(0, G) via r = L * sqrt(D_diag) * Z
    # where G ≈ L D_diag L'
    L = MCRHMC.get_L(F)
    D_diag = MCRHMC.get_D(F)

    # r = L * (sqrt.(D_diag) .* Z)
    r = L * (sqrt.(D_diag.diag) .* Z)

    return r
end

"""
Compute negative kinetic energy: -log p(r|z) - normalizing_constant.
"""
function AdvancedHMC.neg_energy(
    h::Hamiltonian{<:ObservedHessianMetric{T,D,P}}, r::V, z::V
) where {V<:AbstractVecOrMat,T,D,P}
    metric = h.metric
    n = metric.K * D + P

    G = calc_observed_hessian(
        z, metric.ssm, metric.K, Val{D}(), Val{P}(), metric.prior_prec, metric.obs_indices
    )

    # Modified Cholesky factorization
    F = MCRHMC.modified_cholesky(G, metric.u, 0)

    # Cache factorization for adaptation
    metric.last_factorization = F

    # log|G| = sum(log(D_diag))
    logdetG = MCRHMC.logdet(F)
    logZ = 0.5 * (n * log(2π) + logdetG)

    # r' G^{-1} r via solve
    G_inv_r = F \ collect(r)
    rTG_inv_r = dot(r, G_inv_r)

    return -logZ - rTG_inv_r / 2
end

function AdvancedHMC.∂H∂θ(
    h::Hamiltonian{<:ObservedHessianMetric{T,D,P}},
    z::AbstractVecOrMat{T},
    r::AbstractVecOrMat{T},
) where {T,D,P}
    metric = h.metric
    n = metric.K * D + P

    # Compute log-likelihood and gradient
    ℓπ = calc_ll_observed(
        z,
        metric.ys,
        metric.ssm,
        metric.K,
        Val{D}(),
        Val{P}(),
        metric.prior_mean,
        metric.prior_prec,
        metric.obs_indices,
    )
    ∂ℓπ∂z = calc_ll_grad_observed(
        z,
        metric.ys,
        metric.ssm,
        metric.K,
        Val{D}(),
        Val{P}(),
        metric.prior_mean,
        metric.prior_prec,
        metric.obs_indices,
    )

    # Compute observed Hessian
    G = calc_observed_hessian(
        z, metric.ssm, metric.K, Val{D}(), Val{P}(), metric.prior_prec, metric.obs_indices
    )

    # Modified Cholesky factorization
    F = MCRHMC.modified_cholesky(G, metric.u, 0)

    # Cache factorization for adaptation
    metric.last_factorization = F

    # Compute v = G^{-1} r (needed for quadform pullback)
    v = F \ collect(r)

    # Compute ∂H/∂G using MCRHMC's hamiltonian_grad
    # This gives ∂(½ log|G| + ½ r' G^{-1} r)/∂G
    H_grad = MCRHMC.hamiltonian_grad(F, G, v)

    # Compute derivatives of G w.r.t. each component
    dGs = calc_observed_hessian_derivs(
        z, metric.ssm, metric.K, Val{D}(), Val{P}(), metric.obs_indices
    )

    # Compute full gradient: ∂H/∂z_i = -∂ℓπ/∂z_i + trace(H_grad' × ∂G/∂z_i)
    grad = Vector{T}(undef, n)
    @inbounds for i in 1:n
        # Potential energy gradient
        grad[i] = -∂ℓπ∂z[i]

        # Metric tensor gradient via chain rule
        # trace(H_grad' × dG) for symmetric matrices
        grad[i] += MCRHMC.sparse_trace_symmetric(H_grad, dGs[i])
    end

    return DualValue(ℓπ, grad)
end

function AdvancedHMC.∂H∂r(
    h::Hamiltonian{<:ObservedHessianMetric{T,D,P,SSM,YT,OI}},
    z::AbstractVecOrMat,
    r::AbstractVecOrMat,
) where {T,D,P,SSM,YT,OI}
    metric = h.metric

    G = calc_observed_hessian(
        z, metric.ssm, metric.K, Val{D}(), Val{P}(), metric.prior_prec, metric.obs_indices
    )

    # Modified Cholesky factorization
    F = MCRHMC.modified_cholesky(G, metric.u, 0)

    # Cache factorization for adaptation
    metric.last_factorization = F

    # ∂H/∂r = G^{-1} r
    return F \ collect(r)
end

# ============================================================================
# Adaptation Interface
# ============================================================================

"""
    get_regularization_sensitivities(metric::ObservedHessianMetric)

Get regularization sensitivities from the metric's cached factorization.
Called by the integrator when a divergence is detected during fixed-point iteration.

Returns a vector of sensitivities for each regularized dimension, or nothing
if no factorization is cached.
"""
function get_regularization_sensitivities(metric::ObservedHessianMetric)
    if metric.last_factorization !== nothing
        return MCRHMC.compute_regularization_sensitivities(metric.last_factorization)
    end
    return nothing
end

"""
    update_regularization!(metric::ObservedHessianMetric, u_new::Vector)

Update the regularization parameters in the metric.
Called by the adaptor after adjusting u values based on divergence rates.
"""
function update_regularization!(metric::ObservedHessianMetric, u_new::Vector)
    metric.u .= u_new
    return nothing
end

export get_regularization_sensitivities, update_regularization!
