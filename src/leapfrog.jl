function calc_hamiltonian(θs::BlockVector{T,D}, ps::BlockVector{T,D}, ys, ssm) where {T,D}
    ll = calc_ll(θs, ys, ssm)
    G = calc_G(zs, inv(ssm.prior.Σ), ssm.dyn, ssm.sensor, K)
    return _calc_hamiltonian(θs, ps, ll, G)
end

function _calc_hamiltonian(θs::BlockVector{T,D}, ps::BlockVector{T,D}, ll, G) where {T,D}
    G_chol = cholesky(G)
    K = length(θs.blocks)

    log_det = logdet(G_chol)

    # Whiten ps to compute quadratic form stably
    w = G_chol.U.data' \ ps
    pTG_inv_p = sum(abs2, w)

    return -ll + T(0.5) * (K * D * log(T(2π)) + log_det + pTG_inv_p)
end

function ∇θ_H(θs::BlockVector{T,D}, ps::BlockVector{T,D}, ys::Vector{T}, ssm) where {T,D}
    grad = calc_ll_grad(θs, ys, ssm)
    G = calc_G(θs, inv(ssm.prior.Σ), ssm.dyn, ssm.sensor, K)
    # Stored in packed format since each derivative only has values in blocks (k, k), (k,
    # k+1), (k+1, k). Each element of dGs is block-tridiagonal and corresponds to the
    # derivatives of each block with respect to x_k^(d)
    dGs = calc_dGs(θs, ssm.dyn, ssm.sensor, K)

    return _∇θ_H(ps, grad, G, dGs)
end

function _∇θ_H(ps::BlockVector{T,D}, grad, G, dGs) where {T,D}
    G_chol = cholesky(G)
    G_inv_p = G_chol \ ps  # TODO: We have UpperTriangular{BlockUpperBidiag}
    G_inv = block_tridiag_selected_inv(G)
    K = length(ps.blocks)

    ∇θ_blocks = Vector{SVector{D,T}}(undef, K)
    @inbounds for k in 1:K
        ∇θ_blocks[k] = @SVector [
            begin
                v = -grad.blocks[k][d]
                # Trace term
                v -= -0.5 * sum(G_inv.diag_blocks[k] .* dGs[d].diag_blocks[k])
                if k < K
                    v -= -sum(G_inv.super_blocks[k] .* dGs[d].super_blocks[k])
                end
                # Quadratic form term
                v -= 0.5 * G_inv_p.blocks[k]' * dGs[d].diag_blocks[k] * G_inv_p.blocks[k]
                if k < K
                    v -=
                        G_inv_p.blocks[k]' * dGs[d].super_blocks[k] * G_inv_p.blocks[k + 1]
                end
                v
            end for d in 1:D
        ]
    end

    return BlockVector{T,D}(∇θ_blocks)
end

function ∇p_H(θs::BlockVector{T,D}, ps::BlockVector{T,D}, ssm) where {T,D}
    G = calc_G(θs, inv(ssm.prior.Σ), ssm.dyn, ssm.sensor, K)
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
    # Using StaticArrays so safe to avoid copying
    ps_new_1, ps_new_2 = ps

    reps = 0
    while reps < lf_params.max_reps
        reps += 1
        # TODO: grad and G can be cached since only ps is changing
        ps_new_2 = ps - lf_params.ϵ / 2 * ∇θ_H(θs, ps_new_1, ys, ssm)
        if norm(ps_new_1 - ps_new_2) < lf_params.fp_tol
            break
        end
        ps_new_1 = ps_new_2
    end

    if reps == lf_params.max_reps
        @warn "Failed to converge in $(lf_params.max_reps) repetitions."
    end

    return ps_new_2
end

function θ_step(
    θs::BlockVector{T,D}, ps::BlockVector{T,D}, ssm, lf_params::LeapfrogParams{T}
) where {T,D}
    # Using StaticArrays so safe to avoid copying
    θs_new_1, θs_new_2 = θs

    reps = 0
    while reps < lf_params.max_reps
        reps += 1
        θs_new_2 = θs + lf_params.ϵ / 2 * (∇p_H(θs_new_1, ps, ssm) + ∇p_H(θs, ps, ssm))
        if norm(θs_new_1 - θs_new_2) < lf_params.fp_tol
            break
        end
        θs_new_1 = θs_new_2
    end

    if reps == lf_params.max_reps
        @warn "Failed to converge in $(lf_params.max_reps) repetitions."
    end

    return θs_new_2
end

function glf_step(
    θs::BlockVector{T,D}, ps::BlockVector{T,D}, ys, ssm, lf_params::LeapfrogParams{T}
) where {T,D}
    ps = p_half_step(θs, ps, ys, ssm, lf_params)
    θs = θ_step(θs, ps, ssm, lf_params)
    ps = p_half_step(θs, ps, ys, ssm, lf_params)
    return θs, ps
end
