function ∇θ_H(θs::BlockVector{T,D}, ps::BlockVector{T,D}, ys::Vector{T}, ssm) where {T,D}
    grad = calc_ll_grad(θs, ys, ssm)
    G = calc_G(θs, inv(ssm.prior.Σ), ssm.dyn, ssm.sensor, K)
    # Stored in packed format since each derivative only has values in blocks (k, k), (k,
    # k+1), (k+1, k). Each element of dGs is block-tridiagonal and corresponds to the
    # derivatives of each block with respect to x_k^(d)
    dGs = calc_dGs(θs, ssm.dyn, ssm.sensor, K)

    return _∇θ_H(ps, grad, G, dGs)
end

# TODO: should this return a block vector too?
function _∇θ_H(ps::BlockVector{T,D}, grad, G, dGs) where {T,D}
    G_chol = cholesky(G)
    G_inv_p = G_chol \ ps  # TODO: We have UpperTriangular{BlockUpperBidiag}
    G_inv = block_tridiag_selected_inv(G)
    K = length(ps.blocks)

    ∇θ = Vector{Float64}(undef, D * K)
    for k in 1:K
        for d in 1:D
            v = -grad.blocks[k][d]
            # Trace term
            v -= -0.5 * sum(G_inv.diag_blocks[k] .* dGs[d].diag_blocks[k])
            if k < K
                v -= -sum(G_inv.super_blocks[k] .* dGs[d].super_blocks[k])
            end
            # Quadratic form term
            v -= 0.5 * G_inv_p.blocks[k]' * dGs[d].diag_blocks[k] * G_inv_p.blocks[k]
            if k < K
                v -= G_inv_p.blocks[k]' * dGs[d].super_blocks[k] * G_inv_p.blocks[k + 1]
            end

            n = (k - 1) * D + d
            ∇θ[n] = v
        end
    end

    return ∇θ
end

function ∇p_H(θs::BlockVector{T,D}, ps::BlockVector{T,D}, ssm) where {T,D}
    G = calc_G(θs, inv(ssm.prior.Σ), ssm.dyn, ssm.sensor, K)
    return _∇p_H(ps, G)
end

function _∇p_H(ps::BlockVector{T,D}, G) where {T,D}
    G_chol = cholesky(G)
    return G_chol \ ps
end
