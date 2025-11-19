export LeapfrogParams, glf_step, calc_hamiltonian, ∇θ_H, ∇p_H
export calc_dGs, calc_G, calc_ll, calc_ll_grad

using Distributions

function calc_hamiltonian(θs::BlockVector{T,D}, ps::BlockVector{T,D}, ys, ssm) where {T,D}
    ll = calc_ll(θs, ys, ssm)
    G = calc_G(θs, ssm)
    return _calc_hamiltonian(θs, ps, ll, G)
end

function _calc_hamiltonian(θs::BlockVector{T,D}, ps::BlockVector{T,D}, ll, G) where {T,D}
    G_chol = cholesky(G)
    K = length(θs.blocks)

    log_det = logdet(G_chol)

    # Whiten ps to compute quadratic form stably
    w = G_chol.factors' \ ps
    pTG_inv_p = sum(abs2, w)

    return -ll + T(0.5) * (K * D * log(T(2π)) + log_det + pTG_inv_p)
end

function ∇θ_H(θs::BlockVector{T,D}, ps::BlockVector{T,D}, ys, ssm) where {T,D}
    grad = calc_ll_grad(θs, ys, ssm)
    G = calc_G(θs, ssm)
    # Stored in packed format since each derivative only has values in blocks (k, k), (k,
    # k+1), (k+1, k). Each element of dGs is block-tridiagonal and corresponds to the
    # derivatives of each block with respect to x_k^(d)
    dGs = calc_dGs(θs, ssm)

    return _∇θ_H(ps, grad, G, dGs)
end

function _∇θ_H(ps::BlockVector{T,D}, grad, G, dGs) where {T,D}
    G_chol = cholesky(G)
    G_inv = block_tridiag_selected_inv(G)

    return _∇θ_H_inner(ps, grad, G_chol, G_inv, dGs)
end

function _∇θ_H_inner(ps::BlockVector{T,D}, grad, G_chol, G_inv, dGs) where {T,D}
    G_inv_p = G_chol \ ps
    K = length(ps.blocks)

    ∇θ_blocks = Vector{SVector{D,T}}(undef, K)
    @inbounds for k in 1:K
        ∇θ_blocks[k] = SVector{D,T}(
            ntuple(
                d -> begin
                    v = -grad.blocks[k][d]
                    # Trace term
                    v -= -0.5 * sum(G_inv.diag_blocks[k] .* dGs[d].diag_blocks[k])
                    if k < K
                        v -= -sum(G_inv.super_blocks[k] .* dGs[d].super_blocks[k])
                    end
                    # Quadratic form term
                    v -=
                        0.5 *
                        # TODO: Faster to do this as a dot product of whitened vectors
                        G_inv_p.blocks[k]' *
                        dGs[d].diag_blocks[k] *
                        G_inv_p.blocks[k]
                    if k < K
                        v -=
                            G_inv_p.blocks[k]' *
                            dGs[d].super_blocks[k] *
                            G_inv_p.blocks[k + 1]
                    end
                    v
                end,
                Val(D),
            ),
        )
    end

    return BlockVector{T,D}(∇θ_blocks)
end

function ∇p_H(θs::BlockVector{T,D}, ps::BlockVector{T,D}, ssm) where {T,D}
    G = calc_G(θs, ssm)
    return _∇p_H(ps, G)
end

function _∇p_H(ps::BlockVector{T,D}, G) where {T,D}
    G_chol = cholesky(G)
    return G_chol \ ps
end

struct LeapfrogParams{T}
    ϵ::T
    tol::T
    max_reps::Int
end

function p_half_step(
    θs::BlockVector{T,D}, ps::BlockVector{T,D}, ys, ssm, lf_params::LeapfrogParams{T}
) where {T,D}
    ps_new_1 = copy(ps)
    ps_new_2 = copy(ps)

    # grad and G can be cached since only ps is changing
    grad = calc_ll_grad(θs, ys, ssm)
    G = calc_G(θs, ssm)
    dGs = calc_dGs(θs, ssm)
    G_chol = cholesky(G)
    G_inv = block_tridiag_selected_inv(G)

    reps = 0
    while reps < lf_params.max_reps
        reps += 1
        # ps_new_2 = ps - lf_params.ϵ / 2 * ∇θ_H(θs, ps_new_1, ys, ssm)
        ps_new_2 = ps - lf_params.ϵ / 2 * _∇θ_H_inner(ps_new_1, grad, G_chol, G_inv, dGs)
        if norm(ps_new_1 - ps_new_2) < lf_params.tol
            break
        end
        ps_new_1 = ps_new_2
    end

    if reps == lf_params.max_reps
        # @warn "Failed to converge in $(lf_params.max_reps) repetitions."
        return ps, true
    end

    return ps_new_2, false
end

function θ_step(
    θs::BlockVector{T,D}, ps::BlockVector{T,D}, ssm, lf_params::LeapfrogParams{T}
) where {T,D}
    θs_new_1 = copy(θs)
    θs_new_2 = copy(θs)

    reps = 0
    while reps < lf_params.max_reps
        reps += 1
        θs_new_2 = θs + lf_params.ϵ / 2 * (∇p_H(θs_new_1, ps, ssm) + ∇p_H(θs, ps, ssm))
        if norm(θs_new_1 - θs_new_2) < lf_params.tol
            break
        end
        θs_new_1 = θs_new_2
    end

    if reps == lf_params.max_reps
        # @warn "Failed to converge in $(lf_params.max_reps) repetitions."
        return θs, true
    end

    return θs_new_2, false
end

function glf_step(
    θs::BlockVector{T,D}, ps::BlockVector{T,D}, ys, ssm, lf_params::LeapfrogParams{T}
) where {T,D}
    ps, diverged = p_half_step(θs, ps, ys, ssm, lf_params)
    diverged && return θs, ps, true
    θs, diverged = θ_step(θs, ps, ssm, lf_params)
    diverged && return θs, ps, true
    ps, diverged = p_half_step(θs, ps, ys, ssm, lf_params)
    diverged && return θs, ps, true
    return θs, ps, false
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
        Vector{SMatrix{D,D,T}}(undef, K), Vector{SMatrix{D,D,T}}(undef, K - 1)
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
    dGs = Vector{SymPSDBlockTridiag{T,D,D^2}}(undef, D)
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
