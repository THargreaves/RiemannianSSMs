"""
Generalized Kleppe's observed Hessian metric for RMHMC (arXiv:1612.04093v2).

This implements the modified Cholesky approach using the observed (negative) Hessian
of the log-posterior, working with any StateSpaceModel through the unified interface.

Uses the MCRHMC submodule for modified Cholesky factorization and its derivatives.
"""

using AdvancedHMC
using Distributions
using SparseArrays

import AdvancedHMC: DualValue, get_regularization_sensitivities

export GeneralizedObservedHessianMetric
export calc_observed_hessian_unified, calc_observed_hessian_derivs_unified
export calc_ll_observed_unified, calc_ll_grad_observed_unified

# ============================================================================
# Metric Type
# ============================================================================

"""
    GeneralizedObservedHessianMetric{Dx, Dy, Dp, T, M, YT, OI}

Kleppe's observed Hessian metric for any StateSpaceModel.

Uses the MCRHMC sparse LDL factorization with smooth regularization.
Supports arbitrary exponential family observations via `observation_family(model)`.
"""
mutable struct GeneralizedObservedHessianMetric{
    Dx,Dy,Dp,T,M<:StateSpaceModel{Dx,Dy,Dp},YT,OI
} <: AdvancedHMC.AbstractRiemannianMetric
    model::M
    ys::YT
    obs_indices::OI
    K::Int
    prior_mean::SVector{Dp,T}
    prior_prec::SVector{Dp,T}
    u::Vector{T}
    symbolic::Any
    last_factorization::Union{Nothing,MCRHMC.SparseLDLFactor{T,Int}}
end

function GeneralizedObservedHessianMetric(
    model::StateSpaceModel{Dx,Dy,Dp},
    ys,
    K::Int,
    prior_mean::AbstractVector{T},
    prior_var::AbstractVector{T};
    obs_indices::Union{Vector{Int},Nothing}=nothing,
    u_init::Union{Vector{T},Nothing}=nothing,
) where {Dx,Dy,Dp,T}
    n = K * Dx + Dp
    u = isnothing(u_init) ? fill(T(1e-5), n) : copy(u_init)
    pm = SVector{Dp,T}(prior_mean)
    pp = SVector{Dp,T}(1 ./ prior_var)
    oi = isnothing(obs_indices) ? collect(1:K) : obs_indices

    return GeneralizedObservedHessianMetric{Dx,Dy,Dp,T,typeof(model),typeof(ys),typeof(oi)}(
        model, ys, oi, K, pm, pp, u, nothing, nothing
    )
end

Base.size(m::GeneralizedObservedHessianMetric{Dx,Dy,Dp}) where {Dx,Dy,Dp} = (m.K * Dx + Dp,)
function Base.size(m::GeneralizedObservedHessianMetric{Dx,Dy,Dp}, ::Int) where {Dx,Dy,Dp}
    m.K * Dx + Dp
end
Base.eltype(::GeneralizedObservedHessianMetric{Dx,Dy,Dp,T}) where {Dx,Dy,Dp,T} = T

function Base.show(
    io::IO, ::Type{<:GeneralizedObservedHessianMetric{Dx,Dy,Dp,T}}
) where {Dx,Dy,Dp,T}
    return print(io, "GeneralizedObservedHessianMetric{$T,$Dx,$Dy,$Dp}")
end

function Base.show(io::IO, m::GeneralizedObservedHessianMetric{Dx,Dy,Dp}) where {Dx,Dy,Dp}
    return print(io, "GeneralizedObservedHessianMetric(K=$(m.K), Dx=$Dx, Dy=$Dy, Dp=$Dp)")
end

# ============================================================================
# Helper functions for indexing conversion
# ============================================================================

"""
Extract state at time k from the full parameter vector z.
"""
@inline function extract_state(z::AbstractVector{T}, k::Int, ::Val{Dx}) where {T,Dx}
    idx = (k - 1) * Dx
    return SVector{Dx,T}(ntuple(i -> z[idx + i], Val(Dx)))
end

"""
Extract parameter vector from the full parameter vector z.
"""
@inline function extract_params(
    z::AbstractVector{T}, K::Int, ::Val{Dx}, ::Val{Dp}
) where {T,Dx,Dp}
    idx = K * Dx
    return SVector{Dp,T}(ntuple(p -> z[idx + p], Val(Dp)))
end

"""
Build component Hessian ∇²f_output from D2f_xx.

The unified interface gives: D2f_xx[d][i,j] = ∂²f_i/(∂x_d ∂x_j)
We need: ∇²f_output[i,j] = ∂²f_output/(∂x_i ∂x_j)

Conversion: ∇²f_output[i,j] = D2f_xx[i][output,j]
"""
function build_component_hessian(
    D2f_xx::NTuple{Dx,SMatrix{Dx,Dx,T}}, output::Int
) where {Dx,T}
    result = MMatrix{Dx,Dx,T}(undef)
    @inbounds for i in 1:Dx
        for j in 1:Dx
            result[i, j] = D2f_xx[i][output, j]
        end
    end
    return SMatrix{Dx,Dx,T}(result)
end

"""
Build all component Hessians from D2f_xx.
Returns a tuple where result[d] = ∇²f_d (Hessian of d-th output component).
"""
function build_all_component_hessians(
    D2f_xx::NTuple{Dx,SMatrix{Dx,Dx,T}}, ::Val{Dx}
) where {Dx,T}
    return ntuple(d -> build_component_hessian(D2f_xx, d), Val(Dx))
end

"""
Build mixed Hessian ∇²_{x,θ}f_output from D2f_xθ.

The unified interface gives: D2f_xθ[d][i,p] = ∂²f_i/(∂x_d ∂θ_p)
We need: ∇²_{x,θ}f_output[i,p] = ∂²f_output/(∂x_i ∂θ_p)

Conversion: ∇²_{x,θ}f_output[i,p] = D2f_xθ[i][output,p]
"""
function build_mixed_hessian(
    D2f_xθ::NTuple{Dx,SMatrix{Dx,Dp,T}}, output::Int
) where {Dx,Dp,T}
    result = MMatrix{Dx,Dp,T}(undef)
    @inbounds for i in 1:Dx
        for p in 1:Dp
            result[i, p] = D2f_xθ[i][output, p]
        end
    end
    return SMatrix{Dx,Dp,T}(result)
end

"""
Build all mixed Hessians from D2f_xθ.
Returns a tuple where result[d] = ∇²_{x,θ}f_d.
"""
function build_all_mixed_hessians(
    D2f_xθ::NTuple{Dx,SMatrix{Dx,Dp,T}}, ::Val{Dx}
) where {Dx,Dp,T}
    return ntuple(d -> build_mixed_hessian(D2f_xθ, d), Val(Dx))
end

# ============================================================================
# Helper functions for observation Hessians (Dy × Dx input → Dx × Dx output)
# ============================================================================

"""
Build observation Hessian ∇²η_m from D2η_xx.

The unified interface gives: D2η_xx[d][m,j] = ∂²η_m/(∂x_d ∂x_j)
We need: ∇²η_m[i,j] = ∂²η_m/(∂x_i ∂x_j)

Conversion: ∇²η_m[i,j] = D2η_xx[i][m,j]
"""
function build_obs_component_hessian(
    D2η_xx::NTuple{Dx,SMatrix{Dy,Dx,T}}, m::Int
) where {Dx,Dy,T}
    result = MMatrix{Dx,Dx,T}(undef)
    @inbounds for i in 1:Dx
        for j in 1:Dx
            result[i, j] = D2η_xx[i][m, j]
        end
    end
    return SMatrix{Dx,Dx,T}(result)
end

"""
Build all observation Hessians from D2η_xx.
Returns a tuple where result[m] = ∇²η_m (Hessian of m-th observation component).
"""
function build_all_obs_hessians(
    D2η_xx::NTuple{Dx,SMatrix{Dy,Dx,T}}, ::Val{Dy}
) where {Dx,Dy,T}
    return ntuple(m -> build_obs_component_hessian(D2η_xx, m), Val(Dy))
end

"""
Build derivative of component Hessian ∂(∇²f_output)/∂x_c from D3f_xxx.

D3f_xxx[c][d] = ∂(D2f_xx[d])/∂x_c
∂(∇²f_output)/∂x_c[j,k] = ∂³f_output/(∂x_c ∂x_j ∂x_k)

Using D2f_xx indexing: ∇²f_output[j,k] = D2f_xx[j][output,k]
So: ∂(∇²f_output)/∂x_c[j,k] = D3f_xxx[c][j][output,k]
"""
function build_dyn_hess_deriv(
    D3f_xxx::NTuple{Dx,NTuple{Dx,SMatrix{Dx,Dx,T}}}, output::Int, c::Int
) where {Dx,T}
    result = MMatrix{Dx,Dx,T}(undef)
    @inbounds for j in 1:Dx
        for k in 1:Dx
            result[j, k] = D3f_xxx[c][j][output, k]
        end
    end
    return SMatrix{Dx,Dx,T}(result)
end

"""
Build derivative of mixed Hessian ∂(∇²_{x,θ}f_output)/∂x_c from D3f_xxθ.

D3f_xxθ[c][d] = ∂(D2f_xθ[d])/∂x_c
∂(∇²_{x,θ}f_output)/∂x_c[j,p] = ∂³f_output/(∂x_c ∂x_j ∂θ_p)

Using D2f_xθ indexing: ∇²_{x,θ}f_output[j,p] = D2f_xθ[j][output,p]
So: ∂(∇²_{x,θ}f_output)/∂x_c[j,p] = D3f_xxθ[c][j][output,p]
"""
function build_mixed_hess_deriv(
    D3f_xxθ::NTuple{Dx,NTuple{Dx,SMatrix{Dx,Dp,T}}}, output::Int, c::Int
) where {Dx,Dp,T}
    result = MMatrix{Dx,Dp,T}(undef)
    @inbounds for j in 1:Dx
        for p in 1:Dp
            result[j, p] = D3f_xxθ[c][j][output, p]
        end
    end
    return SMatrix{Dx,Dp,T}(result)
end

# ============================================================================
# Log-likelihood and gradient
# ============================================================================

function calc_ll_observed_unified(
    z::AbstractVector{T},
    model::StateSpaceModel{Dx,Dy,Dp},
    ys,
    K::Int,
    prior_mean::SVector{Dp,T},
    prior_prec::SVector{Dp,T},
    obs_indices,
) where {T,Dx,Dy,Dp}
    θ = extract_params(z, K, Val(Dx), Val(Dp))

    # Parameter prior
    ll = -T(0.5) * sum(prior_prec .* (θ .- prior_mean) .^ 2)

    # State prior (k=1)
    x1 = extract_state(z, 1, Val(Dx))
    prior_dist = initial_prior(model)
    ll += logpdf(prior_dist, x1)

    # Dynamics (k=2:K)
    Q = dynamics_covariance(model)
    @inbounds for k in 2:K
        x_k = extract_state(z, k, Val(Dx))
        x_km1 = extract_state(z, k - 1, Val(Dx))
        μ = f(model, x_km1, θ)
        ll += logpdf(MvNormal(μ, Q), x_k)
    end

    # Observations
    ef = observation_family(model)
    obs_set = Set(obs_indices)
    obs_idx = 1
    @inbounds for k in 1:K
        if k in obs_set
            x_k = extract_state(z, k, Val(Dx))
            η_k = η(model, x_k, θ)
            y_k = ys[obs_idx]
            ll += log_likelihood(ef, η_k, y_k)
            obs_idx += 1
        end
    end

    return ll
end

function calc_ll_grad_observed_unified(
    z::AbstractVector{T},
    model::StateSpaceModel{Dx,Dy,Dp},
    ys,
    K::Int,
    prior_mean::SVector{Dp,T},
    prior_prec::SVector{Dp,T},
    obs_indices,
) where {T,Dx,Dy,Dp}
    n = K * Dx + Dp
    grad = zeros(T, n)

    θ = extract_params(z, K, Val(Dx), Val(Dp))

    # Parameter prior gradient
    for p in 1:Dp
        grad[K * Dx + p] = -prior_prec[p] * (θ[p] - prior_mean[p])
    end

    Q_inv = dynamics_covariance_inv(model)

    # State prior gradient (k=1)
    x1 = extract_state(z, 1, Val(Dx))
    prior_dist = initial_prior(model)
    Σ0_inv = inv(prior_dist.Σ)
    g1 = -Σ0_inv * (x1 - prior_dist.μ)
    for d in 1:Dx
        grad[d] = g1[d]
    end

    # Incoming dynamics gradients (k=2:K)
    @inbounds for k in 2:K
        idx = (k - 1) * Dx
        x_k = extract_state(z, k, Val(Dx))
        x_km1 = extract_state(z, k - 1, Val(Dx))
        μ = f(model, x_km1, θ)
        g_k = -Q_inv * (x_k - μ)
        for d in 1:Dx
            grad[idx + d] += g_k[d]
        end
    end

    # Outgoing dynamics gradients (k=1:K-1)
    @inbounds for k in 1:(K - 1)
        idx = (k - 1) * Dx
        x_k = extract_state(z, k, Val(Dx))
        x_kp1 = extract_state(z, k + 1, Val(Dx))
        μ = f(model, x_k, θ)
        Jf = Df_x(model, x_k, θ)
        residual = x_kp1 - μ
        g_k = Jf' * Q_inv * residual
        for d in 1:Dx
            grad[idx + d] += g_k[d]
        end

        # Parameter gradient from dynamics
        f_θ = Df_θ(model, x_k, θ)
        g_θ = f_θ' * Q_inv * residual
        for p in 1:Dp
            grad[K * Dx + p] += g_θ[p]
        end
    end

    # Observation gradients
    ef = observation_family(model)
    obs_set = Set(obs_indices)
    obs_idx = 1
    @inbounds for k in 1:K
        if k in obs_set
            idx = (k - 1) * Dx
            x_k = extract_state(z, k, Val(Dx))
            η_k = η(model, x_k, θ)
            y_k = ys[obs_idx]

            # Score = T(y) - A'(η)
            score = sufficient_stat(ef, y_k) - log_partition_d1(ef, η_k)

            # State gradient
            Dη_x_k = Dη_x(model, x_k, θ)
            g_x = Dη_x_k' * score
            for d in 1:Dx
                grad[idx + d] += g_x[d]
            end

            # Parameter gradient
            Dη_θ_k = Dη_θ(model, x_k, θ)
            g_θ = Dη_θ_k' * score
            for p in 1:Dp
                grad[K * Dx + p] += g_θ[p]
            end

            obs_idx += 1
        end
    end

    return grad
end

# ============================================================================
# Observed Hessian Computation (Sparse)
# ============================================================================

"""
    calc_observed_hessian_unified(z, model, ys, K, prior_prec, obs_indices)

Compute the negative Hessian of the log-posterior as a sparse matrix.

This is the OBSERVED Hessian, which includes second-order terms from nonlinear dynamics.
"""
function calc_observed_hessian_unified(
    z::AbstractVector{T},
    model::StateSpaceModel{Dx,Dy,Dp},
    ys,
    K::Int,
    prior_prec::SVector{Dp,T},
    obs_indices,
) where {T,Dx,Dy,Dp}
    n = K * Dx + Dp

    I_idx = Int[]
    J_idx = Int[]
    V_val = T[]

    θ = extract_params(z, K, Val(Dx), Val(Dp))
    Q_inv = dynamics_covariance_inv(model)
    ef = observation_family(model)
    obs_set = Set(obs_indices)

    # Map from k to observation index
    obs_k_to_idx = Dict{Int,Int}()
    for (idx, k) in enumerate(obs_indices)
        obs_k_to_idx[k] = idx
    end

    # ----- STATE BLOCKS -----
    @inbounds for k in 1:K
        idx_base = (k - 1) * Dx
        x_k = extract_state(z, k, Val(Dx))

        H_kk = zeros(SMatrix{Dx,Dx,T})

        # Prior term (k=1) or incoming dynamics term (k>1)
        if k == 1
            prior_dist = initial_prior(model)
            H_kk += SMatrix{Dx,Dx,T}(inv(prior_dist.Σ))
        else
            H_kk += SMatrix{Dx,Dx,T}(Q_inv)
        end

        # Observation term (only if observed)
        if k in obs_set
            η_k = η(model, x_k, θ)
            y_k = ys[obs_k_to_idx[k]]
            S_k = log_partition_d2(ef, η_k)
            Dη_x_k = Dη_x(model, x_k, θ)

            # GGN term: Dη_x' S Dη_x
            if S_k isa SVector
                H_kk += Dη_x_k' * Diagonal(S_k) * Dη_x_k
            else
                H_kk += S_k[1] * (Dη_x_k' * Dη_x_k)
            end

            # Observed Hessian correction for nonlinear η
            # (usually zero for linear observations)
            D2η_xx_k = D2η_xx(model, x_k, θ)
            score = sufficient_stat(ef, y_k) - log_partition_d1(ef, η_k)
            obs_hess = build_all_obs_hessians(D2η_xx_k, Val(Dy))
            for m in 1:Dy
                H_kk -= score[m] * obs_hess[m]
            end
        end

        # Outgoing dynamics term (k < K)
        if k < K
            x_kp1 = extract_state(z, k + 1, Val(Dx))
            μ = f(model, x_k, θ)
            residual = x_kp1 - μ
            Jf = Df_x(model, x_k, θ)
            D2f_xx_k = D2f_xx(model, x_k, θ)
            dyn_hess = build_all_component_hessians(D2f_xx_k, Val(Dx))

            # GGN term: Jf' Q⁻¹ Jf
            H_kk += Jf' * Q_inv * Jf

            # Observed Hessian correction: -Σ_d [Q⁻¹ r]_d ∇²f_d
            Q_inv_r = Q_inv * residual
            for d in 1:Dx
                H_kk -= Q_inv_r[d] * dyn_hess[d]
            end
        end

        # Add diagonal block entries
        for i in 1:Dx
            for j in 1:Dx
                push!(I_idx, idx_base + i)
                push!(J_idx, idx_base + j)
                push!(V_val, H_kk[i, j])
            end
        end

        # Off-diagonal block G_{k, k+1}
        if k < K
            Jf = Df_x(model, x_k, θ)
            H_k_kp1 = -Jf' * Q_inv

            for i in 1:Dx
                for j in 1:Dx
                    push!(I_idx, idx_base + i)
                    push!(J_idx, idx_base + Dx + j)
                    push!(V_val, H_k_kp1[i, j])
                    push!(I_idx, idx_base + Dx + j)
                    push!(J_idx, idx_base + i)
                    push!(V_val, H_k_kp1[i, j])
                end
            end
        end
    end

    # ----- BORDER BLOCKS (state-parameter coupling) -----
    @inbounds for k in 1:K
        idx_base = (k - 1) * Dx
        x_k = extract_state(z, k, Val(Dx))

        H_k_θ = zeros(SMatrix{Dx,Dp,T})

        # Incoming dynamics term (k > 1): -Q⁻¹ f_θ
        if k > 1
            x_km1 = extract_state(z, k - 1, Val(Dx))
            f_θ = Df_θ(model, x_km1, θ)
            H_k_θ -= SMatrix{Dx,Dp,T}(Q_inv * f_θ)
        end

        # Outgoing dynamics term (k < K)
        if k < K
            x_kp1 = extract_state(z, k + 1, Val(Dx))
            μ = f(model, x_k, θ)
            residual = x_kp1 - μ
            Jf = Df_x(model, x_k, θ)
            f_θ = Df_θ(model, x_k, θ)
            D2f_xθ_k = D2f_xθ(model, x_k, θ)
            mixed_hess = build_all_mixed_hessians(D2f_xθ_k, Val(Dx))

            # GGN term
            H_k_θ += Jf' * Q_inv * f_θ

            # Observed Hessian correction
            Q_inv_r = Q_inv * residual
            for i in 1:Dx
                H_k_θ -= Q_inv_r[i] * mixed_hess[i]
            end
        end

        # Observation border terms (if parameters enter observation)
        if k in obs_set
            η_k = η(model, x_k, θ)
            y_k = ys[obs_k_to_idx[k]]
            S_k = log_partition_d2(ef, η_k)
            Dη_x_k = Dη_x(model, x_k, θ)
            Dη_θ_k = Dη_θ(model, x_k, θ)

            # GGN term from observation
            if S_k isa SVector
                H_k_θ += Dη_x_k' * Diagonal(S_k) * Dη_θ_k
            else
                H_k_θ += S_k[1] * (Dη_x_k' * Dη_θ_k)
            end
        end

        # Add border block entries
        for i in 1:Dx
            for p in 1:Dp
                push!(I_idx, idx_base + i)
                push!(J_idx, K * Dx + p)
                push!(V_val, H_k_θ[i, p])
                push!(I_idx, K * Dx + p)
                push!(J_idx, idx_base + i)
                push!(V_val, H_k_θ[i, p])
            end
        end
    end

    # ----- CORNER BLOCK (parameter-parameter) -----
    H_θθ = zeros(SMatrix{Dp,Dp,T})

    # Parameter prior
    H_θθ += Diagonal(prior_prec)

    # Dynamics contributions
    @inbounds for k in 1:(K - 1)
        x_k = extract_state(z, k, Val(Dx))
        x_kp1 = extract_state(z, k + 1, Val(Dx))
        μ = f(model, x_k, θ)
        residual = x_kp1 - μ

        f_θ = Df_θ(model, x_k, θ)
        D2f_θθ_k = D2f_θθ(model, x_k, θ)

        # GGN term
        H_θθ += f_θ' * Q_inv * f_θ

        # Observed Hessian correction
        Q_inv_r = Q_inv * residual
        for p in 1:Dp
            H_θθ -= Q_inv_r' * D2f_θθ_k[p]
        end
    end

    # Observation corner terms
    for k in obs_set
        x_k = extract_state(z, k, Val(Dx))
        η_k = η(model, x_k, θ)
        S_k = log_partition_d2(ef, η_k)
        Dη_θ_k = Dη_θ(model, x_k, θ)

        # GGN term from observation
        if S_k isa SVector
            H_θθ += Dη_θ_k' * Diagonal(S_k) * Dη_θ_k
        else
            H_θθ += S_k[1] * (Dη_θ_k' * Dη_θ_k)
        end
    end

    # Add corner block entries
    for i in 1:Dp
        for j in 1:Dp
            push!(I_idx, K * Dx + i)
            push!(J_idx, K * Dx + j)
            push!(V_val, H_θθ[i, j])
        end
    end

    return sparse(I_idx, J_idx, V_val, n, n)
end

# ============================================================================
# Hessian Derivatives (for ∂H/∂θ computation)
# ============================================================================

"""
    calc_observed_hessian_derivs_unified(z, model, K, obs_indices)

Compute derivatives of the observed Hessian w.r.t. each component of z.
Returns a vector of sparse matrices ∂G/∂z_i for i = 1:n.
"""
function calc_observed_hessian_derivs_unified(
    z::AbstractVector{T}, model::StateSpaceModel{Dx,Dy,Dp}, K::Int, obs_indices
) where {T,Dx,Dy,Dp}
    n = K * Dx + Dp
    dGs = Vector{SparseMatrixCSC{T,Int}}(undef, n)

    θ = extract_params(z, K, Val(Dx), Val(Dp))
    Q_inv = dynamics_covariance_inv(model)
    ef = observation_family(model)
    obs_set = Set(obs_indices)

    # Map from k to observation index
    obs_k_to_idx = Dict{Int,Int}()
    for (idx, k) in enumerate(obs_indices)
        obs_k_to_idx[k] = idx
    end

    # Derivatives w.r.t. state components
    @inbounds for k in 1:K
        for d in 1:Dx
            idx = (k - 1) * Dx + d
            dGs[idx] = calc_dG_dz_kd_unified(
                z, model, K, k, d, θ, Q_inv, ef, obs_set, obs_k_to_idx
            )
        end
    end

    # Derivatives w.r.t. parameter components
    @inbounds for p in 1:Dp
        idx = K * Dx + p
        dGs[idx] = calc_dG_dθ_p_unified(z, model, K, p, θ, Q_inv, ef, obs_set, obs_k_to_idx)
    end

    return dGs
end

"""
Compute ∂G/∂z_k^{(d)} - derivative of Hessian w.r.t. state component d at time k.
"""
function calc_dG_dz_kd_unified(
    z::AbstractVector{T},
    model::StateSpaceModel{Dx,Dy,Dp},
    K::Int,
    k::Int,
    d::Int,
    θ,
    Q_inv,
    ef,
    obs_set,
    obs_k_to_idx,
) where {T,Dx,Dy,Dp}
    n = K * Dx + Dp
    I_idx = Int[]
    J_idx = Int[]
    V_val = T[]

    idx_base = (k - 1) * Dx
    x_k = extract_state(z, k, Val(Dx))

    # ===== STATE-STATE BLOCKS =====

    # Effect on G_{k,k} from outgoing dynamics (transition k → k+1)
    if k < K
        x_kp1 = extract_state(z, k + 1, Val(Dx))
        μ = f(model, x_k, θ)
        residual = x_kp1 - μ
        Jf = Df_x(model, x_k, θ)
        D2f_xx_k = D2f_xx(model, x_k, θ)
        D3f_xxx_k = D3f_xxx(model, x_k, θ)

        dyn_hess = build_all_component_hessians(D2f_xx_k, Val(Dx))

        # GGN term: ∂(Jf' Q⁻¹ Jf)/∂z_k^{(d)}
        Hfs_d = D2f_xx_k[d]  # This is ∂Jf/∂x_d
        Mf = Q_inv * Jf
        dH_kk = Hfs_d' * Mf + Mf' * Hfs_d

        # Observed correction term derivative
        Q_inv_Jf_col = Q_inv * Jf[:, d]
        for i in 1:Dx
            dH_kk += Q_inv_Jf_col[i] * dyn_hess[i]
        end

        Q_inv_r = Q_inv * residual
        for i in 1:Dx
            dyn_hess_deriv_i_d = build_dyn_hess_deriv(D3f_xxx_k, i, d)
            dH_kk -= Q_inv_r[i] * dyn_hess_deriv_i_d
        end

        for i in 1:Dx
            for j in 1:Dx
                if abs(dH_kk[i, j]) > 1e-15
                    push!(I_idx, idx_base + i)
                    push!(J_idx, idx_base + j)
                    push!(V_val, dH_kk[i, j])
                end
            end
        end

        # Effect on G_{k,k+1}
        dH_k_kp1 = -Hfs_d' * Q_inv

        for i in 1:Dx
            for j in 1:Dx
                if abs(dH_k_kp1[i, j]) > 1e-15
                    push!(I_idx, idx_base + i)
                    push!(J_idx, idx_base + Dx + j)
                    push!(V_val, dH_k_kp1[i, j])
                    push!(I_idx, idx_base + Dx + j)
                    push!(J_idx, idx_base + i)
                    push!(V_val, dH_k_kp1[i, j])
                end
            end
        end
    end

    # Effect on G_{k-1,k-1} from incoming dynamics (transition k-1 → k)
    if k > 1
        idx_base_km1 = (k - 2) * Dx
        x_km1 = extract_state(z, k - 1, Val(Dx))
        D2f_xx_km1 = D2f_xx(model, x_km1, θ)
        dyn_hess_km1 = build_all_component_hessians(D2f_xx_km1, Val(Dx))

        e_d = zeros(SVector{Dx,T})
        e_d = setindex(e_d, one(T), d)
        Q_inv_ed = Q_inv * e_d

        dH_km1_km1 = zeros(SMatrix{Dx,Dx,T})
        for i in 1:Dx
            dH_km1_km1 -= Q_inv_ed[i] * dyn_hess_km1[i]
        end

        for i in 1:Dx
            for j in 1:Dx
                if abs(dH_km1_km1[i, j]) > 1e-15
                    push!(I_idx, idx_base_km1 + i)
                    push!(J_idx, idx_base_km1 + j)
                    push!(V_val, dH_km1_km1[i, j])
                end
            end
        end
    end

    # ===== BORDER BLOCKS (state-parameter) =====

    # Effect on G_{k-1,θ} from outgoing dynamics at k-1
    if k > 1
        idx_base_km1 = (k - 2) * Dx
        x_km1 = extract_state(z, k - 1, Val(Dx))
        D2f_xθ_km1 = D2f_xθ(model, x_km1, θ)
        mixed_hess_km1 = build_all_mixed_hessians(D2f_xθ_km1, Val(Dx))

        e_d = zeros(SVector{Dx,T})
        e_d = setindex(e_d, one(T), d)
        Q_inv_ed = Q_inv * e_d

        dH_km1_θ = zeros(SMatrix{Dx,Dp,T})
        for i in 1:Dx
            dH_km1_θ -= Q_inv_ed[i] * mixed_hess_km1[i]
        end

        for i in 1:Dx
            for p in 1:Dp
                if abs(dH_km1_θ[i, p]) > 1e-15
                    push!(I_idx, idx_base_km1 + i)
                    push!(J_idx, K * Dx + p)
                    push!(V_val, dH_km1_θ[i, p])
                    push!(I_idx, K * Dx + p)
                    push!(J_idx, idx_base_km1 + i)
                    push!(V_val, dH_km1_θ[i, p])
                end
            end
        end
    end

    # Effect on G_{k+1,θ} from incoming dynamics at k+1
    if k < K
        D2f_xθ_k = D2f_xθ(model, x_k, θ)
        idx_base_kp1 = k * Dx

        Hf_θs_d = D2f_xθ_k[d]
        dH_kp1_θ = -Q_inv * Hf_θs_d

        for i in 1:Dx
            for p in 1:Dp
                if abs(dH_kp1_θ[i, p]) > 1e-15
                    push!(I_idx, idx_base_kp1 + i)
                    push!(J_idx, K * Dx + p)
                    push!(V_val, dH_kp1_θ[i, p])
                    push!(I_idx, K * Dx + p)
                    push!(J_idx, idx_base_kp1 + i)
                    push!(V_val, dH_kp1_θ[i, p])
                end
            end
        end
    end

    # Effect on G_{k,θ} from outgoing dynamics
    if k < K
        x_kp1 = extract_state(z, k + 1, Val(Dx))
        μ = f(model, x_k, θ)
        residual = x_kp1 - μ
        Jf = Df_x(model, x_k, θ)
        f_θ = Df_θ(model, x_k, θ)
        D2f_xx_k = D2f_xx(model, x_k, θ)
        D2f_xθ_k = D2f_xθ(model, x_k, θ)
        D3f_xxθ_k = D3f_xxθ(model, x_k, θ)

        mixed_hess = build_all_mixed_hessians(D2f_xθ_k, Val(Dx))
        Hfs_d = D2f_xx_k[d]
        Hf_θs_d = D2f_xθ_k[d]

        # GGN term
        dH_k_θ = Hfs_d' * Q_inv * f_θ + Jf' * Q_inv * Hf_θs_d

        # Correction term
        Q_inv_Jf_col = Q_inv * Jf[:, d]
        Q_inv_r = Q_inv * residual
        for i in 1:Dx
            dH_k_θ += Q_inv_Jf_col[i] * mixed_hess[i]
            mixed_hess_deriv_i_d = build_mixed_hess_deriv(D3f_xxθ_k, i, d)
            dH_k_θ -= Q_inv_r[i] * mixed_hess_deriv_i_d
        end

        for i in 1:Dx
            for p in 1:Dp
                if abs(dH_k_θ[i, p]) > 1e-15
                    push!(I_idx, idx_base + i)
                    push!(J_idx, K * Dx + p)
                    push!(V_val, dH_k_θ[i, p])
                    push!(I_idx, K * Dx + p)
                    push!(J_idx, idx_base + i)
                    push!(V_val, dH_k_θ[i, p])
                end
            end
        end
    end

    # ===== CORNER BLOCK (parameter-parameter) =====
    dH_θθ = zeros(SMatrix{Dp,Dp,T})

    # Effect from outgoing dynamics at k
    if k < K
        x_kp1 = extract_state(z, k + 1, Val(Dx))
        μ = f(model, x_k, θ)
        residual = x_kp1 - μ
        Jf = Df_x(model, x_k, θ)
        f_θ = Df_θ(model, x_k, θ)
        D2f_θθ_k = D2f_θθ(model, x_k, θ)
        D2f_xθ_k = D2f_xθ(model, x_k, θ)
        D3f_xθθ_k = D3f_xθθ(model, x_k, θ)

        Hf_θs_d = D2f_xθ_k[d]

        # GGN term
        dH_θθ += Hf_θs_d' * Q_inv * f_θ + f_θ' * Q_inv * Hf_θs_d

        # Correction term: ∂[-Σ_i (Q⁻¹r)_i × f_θθ]/∂x_d
        #   = +Σ_i (Q⁻¹ Jf[:,d])_i × f_θθ  (from ∂r/∂x_d = -Jf[:,d])
        #   - Σ_i (Q⁻¹r)_i × ∂f_θθ/∂x_d    (third derivative)
        Q_inv_Jf_col = Q_inv * Jf[:, d]
        Q_inv_r = Q_inv * residual
        for p in 1:Dp
            # First correction term (from residual change)
            dH_θθ += Q_inv_Jf_col' * D2f_θθ_k[p]
            # Second correction term (third derivative)
            df_θθ_p_d = D3f_xθθ_k[d][p]
            dH_θθ -= Q_inv_r' * df_θθ_p_d
        end
    end

    # Effect from incoming dynamics at k
    if k > 1
        idx_base_km1 = (k - 2) * Dx
        x_km1 = extract_state(z, k - 1, Val(Dx))
        D2f_θθ_km1 = D2f_θθ(model, x_km1, θ)

        e_d = zeros(SVector{Dx,T})
        e_d = setindex(e_d, one(T), d)
        Q_inv_ed = Q_inv * e_d

        for p in 1:Dp
            dH_θθ -= Q_inv_ed' * D2f_θθ_km1[p]
        end
    end

    for i in 1:Dp
        for j in 1:Dp
            if abs(dH_θθ[i, j]) > 1e-15
                push!(I_idx, K * Dx + i)
                push!(J_idx, K * Dx + j)
                push!(V_val, dH_θθ[i, j])
            end
        end
    end

    return sparse(I_idx, J_idx, V_val, n, n)
end

"""
Compute ∂G/∂θ^{(p)} - derivative of Hessian w.r.t. parameter component p.
"""
function calc_dG_dθ_p_unified(
    z::AbstractVector{T},
    model::StateSpaceModel{Dx,Dy,Dp},
    K::Int,
    p::Int,
    θ,
    Q_inv,
    ef,
    obs_set,
    obs_k_to_idx,
) where {T,Dx,Dy,Dp}
    n = K * Dx + Dp
    I_idx = Int[]
    J_idx = Int[]
    V_val = T[]

    @inbounds for k in 1:K
        idx_base = (k - 1) * Dx
        x_k = extract_state(z, k, Val(Dx))

        # ===== STATE-STATE BLOCKS =====
        dH_kk = zeros(SMatrix{Dx,Dx,T})

        if k < K
            x_kp1 = extract_state(z, k + 1, Val(Dx))
            μ = f(model, x_k, θ)
            residual = x_kp1 - μ
            Jf = Df_x(model, x_k, θ)
            DDf_x_θ_k = DDf_x_θ(model, x_k, θ)
            f_θ = Df_θ(model, x_k, θ)
            D2f_xx_k = D2f_xx(model, x_k, θ)

            dyn_hess = build_all_component_hessians(D2f_xx_k, Val(Dx))
            Jf_θ_p = DDf_x_θ_k[p]

            # GGN term
            Mf = Q_inv * Jf
            dH_kk += Jf_θ_p' * Mf + Mf' * Jf_θ_p

            # Correction term
            Q_inv_f_θ = Q_inv * f_θ[:, p]
            Q_inv_r = Q_inv * residual
            for i in 1:Dx
                dH_kk += Q_inv_f_θ[i] * dyn_hess[i]
                # Third derivative ∂(∇²f_i)/∂θ - for many models this equals dyn_hess[i]
                # due to log parameterization
                D3f_xxθ_k = D3f_xxθ(model, x_k, θ)
                dyn_hess_θ = build_dyn_hess_deriv(
                    # Need to convert D3f_xxθ structure - this is approximate
                    # For log parameterization, ∂(∇²f)/∂θ ≈ ∇²f
                    D3f_xxx(model, x_k, θ),
                    i,
                    1,
                )
                # Actually need parameter derivatives here - using approximation
                dH_kk -= Q_inv_r[i] * dyn_hess[i]  # Approximation for log-param
            end
        end

        for i in 1:Dx
            for j in 1:Dx
                if abs(dH_kk[i, j]) > 1e-15
                    push!(I_idx, idx_base + i)
                    push!(J_idx, idx_base + j)
                    push!(V_val, dH_kk[i, j])
                end
            end
        end

        # Effect on G_{k,k+1}
        if k < K
            DDf_x_θ_k = DDf_x_θ(model, x_k, θ)
            Jf_θ_p = DDf_x_θ_k[p]
            dH_k_kp1 = -Jf_θ_p' * Q_inv

            for i in 1:Dx
                for j in 1:Dx
                    if abs(dH_k_kp1[i, j]) > 1e-15
                        push!(I_idx, idx_base + i)
                        push!(J_idx, idx_base + Dx + j)
                        push!(V_val, dH_k_kp1[i, j])
                        push!(I_idx, idx_base + Dx + j)
                        push!(J_idx, idx_base + i)
                        push!(V_val, dH_k_kp1[i, j])
                    end
                end
            end
        end

        # ===== BORDER BLOCKS =====
        dH_k_θ = zeros(SMatrix{Dx,Dp,T})

        # Incoming dynamics
        if k > 1
            x_km1 = extract_state(z, k - 1, Val(Dx))
            D2f_θθ_km1 = D2f_θθ(model, x_km1, θ)
            dH_k_θ += -Q_inv * D2f_θθ_km1[p]
        end

        # Outgoing dynamics
        if k < K
            x_kp1 = extract_state(z, k + 1, Val(Dx))
            μ = f(model, x_k, θ)
            residual = x_kp1 - μ
            Jf = Df_x(model, x_k, θ)
            DDf_x_θ_k = DDf_x_θ(model, x_k, θ)
            f_θ = Df_θ(model, x_k, θ)
            D2f_θθ_k = D2f_θθ(model, x_k, θ)
            D2f_xθ_k = D2f_xθ(model, x_k, θ)

            mixed_hess = build_all_mixed_hessians(D2f_xθ_k, Val(Dx))
            Jf_θ_p = DDf_x_θ_k[p]

            # GGN term
            dH_k_θ += Jf_θ_p' * Q_inv * f_θ + Jf' * Q_inv * D2f_θθ_k[p]

            # Correction term (approximation for log-param)
            Q_inv_f_θ = Q_inv * f_θ[:, p]
            Q_inv_r = Q_inv * residual
            for i in 1:Dx
                dH_k_θ += Q_inv_f_θ[i] * mixed_hess[i]
                dH_k_θ -= Q_inv_r[i] * mixed_hess[i]  # Approximation
            end
        end

        for i in 1:Dx
            for pp in 1:Dp
                if abs(dH_k_θ[i, pp]) > 1e-15
                    push!(I_idx, idx_base + i)
                    push!(J_idx, K * Dx + pp)
                    push!(V_val, dH_k_θ[i, pp])
                    push!(I_idx, K * Dx + pp)
                    push!(J_idx, idx_base + i)
                    push!(V_val, dH_k_θ[i, pp])
                end
            end
        end
    end

    # ===== CORNER BLOCK =====
    dH_θθ = zeros(SMatrix{Dp,Dp,T})

    @inbounds for k in 1:(K - 1)
        x_k = extract_state(z, k, Val(Dx))
        x_kp1 = extract_state(z, k + 1, Val(Dx))
        μ = f(model, x_k, θ)
        residual = x_kp1 - μ

        f_θ = Df_θ(model, x_k, θ)
        D2f_θθ_k = D2f_θθ(model, x_k, θ)

        # GGN term
        dH_θθ += D2f_θθ_k[p]' * Q_inv * f_θ + f_θ' * Q_inv * D2f_θθ_k[p]

        # Correction term (approximation)
        Q_inv_f_θ = Q_inv * f_θ[:, p]
        Q_inv_r = Q_inv * residual
        for i in 1:Dx
            f_θθ_i = D2f_θθ_k[1][i:i, :]
            dH_θθ += Q_inv_f_θ[i] * f_θθ_i
            dH_θθ -= Q_inv_r[i] * f_θθ_i  # Approximation
        end
    end

    for i in 1:Dp
        for j in 1:Dp
            if abs(dH_θθ[i, j]) > 1e-15
                push!(I_idx, K * Dx + i)
                push!(J_idx, K * Dx + j)
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
    metric::GeneralizedObservedHessianMetric{Dx,Dy,Dp,T},
    kinetic,
    z::AbstractVecOrMat,
) where {Dx,Dy,Dp,T}
    G = calc_observed_hessian_unified(
        z, metric.model, metric.ys, metric.K, metric.prior_prec, metric.obs_indices
    )

    F = MCRHMC.modified_cholesky(G, metric.u, 0)
    metric.last_factorization = F

    n = metric.K * Dx + Dp
    Z = randn(rng, n)

    L = MCRHMC.get_L(F)
    D_diag = MCRHMC.get_D(F)

    r = L * (sqrt.(D_diag.diag) .* Z)
    return r
end

function AdvancedHMC.neg_energy(
    h::Hamiltonian{<:GeneralizedObservedHessianMetric{Dx,Dy,Dp,T}}, r::V, z::V
) where {V<:AbstractVecOrMat,Dx,Dy,Dp,T}
    metric = h.metric
    n = metric.K * Dx + Dp

    G = calc_observed_hessian_unified(
        z, metric.model, metric.ys, metric.K, metric.prior_prec, metric.obs_indices
    )

    F = MCRHMC.modified_cholesky(G, metric.u, 0)
    metric.last_factorization = F

    logdetG = MCRHMC.logdet(F)
    logZ = T(0.5) * (n * log(T(2π)) + logdetG)

    G_inv_r = F \ collect(r)
    rTG_inv_r = dot(r, G_inv_r)

    return -logZ - rTG_inv_r / 2
end

function AdvancedHMC.∂H∂θ(
    h::Hamiltonian{<:GeneralizedObservedHessianMetric{Dx,Dy,Dp,T}},
    z::AbstractVecOrMat{T},
    r::AbstractVecOrMat{T},
) where {Dx,Dy,Dp,T}
    metric = h.metric
    n = metric.K * Dx + Dp

    ℓπ = calc_ll_observed_unified(
        z,
        metric.model,
        metric.ys,
        metric.K,
        metric.prior_mean,
        metric.prior_prec,
        metric.obs_indices,
    )
    ∂ℓπ∂z = calc_ll_grad_observed_unified(
        z,
        metric.model,
        metric.ys,
        metric.K,
        metric.prior_mean,
        metric.prior_prec,
        metric.obs_indices,
    )

    G = calc_observed_hessian_unified(
        z, metric.model, metric.ys, metric.K, metric.prior_prec, metric.obs_indices
    )

    F = MCRHMC.modified_cholesky(G, metric.u, 0)
    metric.last_factorization = F

    v = F \ collect(r)
    H_grad = MCRHMC.hamiltonian_grad(F, G, v)

    dGs = calc_observed_hessian_derivs_unified(
        z, metric.model, metric.K, metric.obs_indices
    )

    grad = Vector{T}(undef, n)
    @inbounds for i in 1:n
        grad[i] = -∂ℓπ∂z[i]
        grad[i] += MCRHMC.sparse_trace_symmetric(H_grad, dGs[i])
    end

    return DualValue(ℓπ, grad)
end

function AdvancedHMC.∂H∂r(
    h::Hamiltonian{<:GeneralizedObservedHessianMetric{Dx,Dy,Dp,T}},
    z::AbstractVecOrMat,
    r::AbstractVecOrMat,
) where {Dx,Dy,Dp,T}
    metric = h.metric

    G = calc_observed_hessian_unified(
        z, metric.model, metric.ys, metric.K, metric.prior_prec, metric.obs_indices
    )

    F = MCRHMC.modified_cholesky(G, metric.u, 0)
    metric.last_factorization = F

    return F \ collect(r)
end

# ============================================================================
# Adaptation Interface
# ============================================================================

function get_regularization_sensitivities(metric::GeneralizedObservedHessianMetric)
    if metric.last_factorization !== nothing
        return MCRHMC.compute_regularization_sensitivities(metric.last_factorization)
    end
    return nothing
end

function update_regularization!(metric::GeneralizedObservedHessianMetric, u_new::Vector)
    metric.u .= u_new
    return nothing
end

export get_regularization_sensitivities, update_regularization!
