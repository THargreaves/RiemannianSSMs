"""
Adjoint (backward pass) for the modified Cholesky factorization.

Given cotangents ∂Loss/∂L̃ and ∂Loss/∂D, computes ∂Loss/∂A.
"""

"""
    SparseLDLAdjoint{Tv,Ti}

Storage for adjoint computation of modified Cholesky.

Fields:
- `L_bar`: Cotangent with respect to L̃ non-zeros
- `D_bar`: Cotangent with respect to D
- `A_bar`: Result - cotangent with respect to A (same sparsity pattern)
- `work`: Workspace vector
"""
mutable struct SparseLDLAdjoint{Tv<:AbstractFloat,Ti<:Integer}
    L_bar::Vector{Tv}
    D_bar::Vector{Tv}
    A_bar_colptr::Vector{Ti}
    A_bar_rowval::Vector{Ti}
    A_bar_nzval::Vector{Tv}
    work::Vector{Tv}
end

"""
    SparseLDLAdjoint(F::SparseLDLFactor, A::SparseMatrixCSC)

Create adjoint storage matching the factor and input matrix structure.
"""
function SparseLDLAdjoint(
    F::SparseLDLFactor{Tv,Ti}, A::SparseMatrixCSC{Tv,Ti}
) where {Tv,Ti}
    n = F.symbolic.n
    nz_L = nnz_L(F.symbolic)
    nz_A = nnz(A)

    L_bar = zeros(Tv, nz_L)
    D_bar = zeros(Tv, n)
    A_bar_colptr = copy(A.colptr)
    A_bar_rowval = copy(A.rowval)
    A_bar_nzval = zeros(Tv, nz_A)
    work = zeros(Tv, n)

    return SparseLDLAdjoint(L_bar, D_bar, A_bar_colptr, A_bar_rowval, A_bar_nzval, work)
end

"""
    get_A_bar(adj::SparseLDLAdjoint, n)

Construct the adjoint ∂Loss/∂A as a SparseMatrixCSC.
"""
function get_A_bar(adj::SparseLDLAdjoint{Tv,Ti}, n::Integer) where {Tv,Ti}
    return SparseMatrixCSC(
        Ti(n), Ti(n), adj.A_bar_colptr, adj.A_bar_rowval, adj.A_bar_nzval
    )
end

"""
    find_L_index(sym::SparseLDLSymbolic, i, j)

Find the index in Lnzval for entry L̃[i,j] (i > j), or 0 if not present.
"""
@inline function find_L_index(sym::SparseLDLSymbolic{Ti}, i::Integer, j::Integer) where {Ti}
    for p in sym.Lcolptr[j]:(sym.Lcolptr[j + 1] - 1)
        if sym.Lrowval[p] == i
            return p
        end
    end
    return zero(Ti)
end

"""
    find_A_index(A::SparseMatrixCSC, i, j)

Find the index in A.nzval for entry A[i,j], or 0 if not present.
"""
@inline function find_A_index(
    A::SparseMatrixCSC{Tv,Ti}, i::Integer, j::Integer
) where {Tv,Ti}
    for p in A.colptr[j]:(A.colptr[j + 1] - 1)
        if A.rowval[p] == i
            return p
        end
    end
    return zero(Ti)
end

"""
    modified_cholesky_adjoint!(adj::SparseLDLAdjoint, F::SparseLDLFactor,
                               A::SparseMatrixCSC, L_bar_input, D_bar_input)

Compute the adjoint (backward pass) of the modified Cholesky factorization.

Given:
- L_bar_input: ∂Loss/∂L̃ (vector of cotangents for each L̃ entry)
- D_bar_input: ∂Loss/∂D (vector of cotangents for diagonal)

Computes:
- adj.A_bar_nzval: ∂Loss/∂A in the same sparsity pattern as A

The adjoint is computed by reversing Algorithm 1 and applying the chain rule.
"""
function modified_cholesky_adjoint!(
    adj::SparseLDLAdjoint{Tv,Ti},
    F::SparseLDLFactor{Tv,Ti},
    A::SparseMatrixCSC{Tv,Ti},
    L_bar_input::Vector{Tv},
    D_bar_input::Vector{Tv},
) where {Tv,Ti}
    sym = F.symbolic
    n = sym.n
    perm = sym.perm
    invperm = sym.invperm
    Lcolptr = sym.Lcolptr
    Lrowval = sym.Lrowval
    Lnzval = F.Lnzval
    D = F.D
    D_raw = F.D_raw
    u = F.u
    K = F.K
    work = adj.work

    L_bar = copy(L_bar_input)
    D_bar = copy(D_bar_input)

    fill!(adj.A_bar_nzval, zero(Tv))
    fill!(work, zero(Tv))

    for j in n:-1:1
        for p in Lcolptr[j]:(Lcolptr[j + 1] - 1)
            i = Lrowval[p]
            L_ij = Lnzval[p]

            work_bar_i = L_bar[p] / D[j]
            D_bar[j] -= L_bar[p] * L_ij / D[j]

            work[i] += work_bar_i
        end

        if j > K
            u_idx = j - K
            deriv = sabs_deriv(D_raw[j], u[u_idx])
            D_bar[j] *= deriv
        end

        for k in (j - 1):-1:1
            L_jk_idx = find_L_index(sym, j, k)
            if L_jk_idx == 0
                continue
            end
            L_jk = Lnzval[L_jk_idx]

            if L_jk == zero(Tv)
                continue
            end

            for p in Lcolptr[k]:(Lcolptr[k + 1] - 1)
                i = Lrowval[p]
                if i > j
                    L_ik = Lnzval[p]

                    L_bar[p] -= work[i] * L_jk * D[k]
                    L_bar[L_jk_idx] -= work[i] * L_ik * D[k]
                    D_bar[k] -= work[i] * L_ik * L_jk
                end
            end

            L_bar[L_jk_idx] -= D_bar[j] * 2 * L_jk * D[k]
            D_bar[k] -= D_bar[j] * L_jk * L_jk
        end

        pj = perm[j]

        pj_diag_idx = find_A_index(A, pj, pj)
        if pj_diag_idx > 0
            adj.A_bar_nzval[pj_diag_idx] += D_bar[j]
        end

        for p in Lcolptr[j]:(Lcolptr[j + 1] - 1)
            i = Lrowval[p]
            pi = perm[i]

            # For symmetric matrices, add half to each triangle so that:
            # - grad[i,j] = grad[j,i] = true partial derivative
            # - Matches numerical gradient and standard autodiff conventions
            A_idx = find_A_index(A, pi, pj)
            A_idx_t = find_A_index(A, pj, pi)

            if A_idx > 0 && A_idx_t > 0 && A_idx != A_idx_t
                # Both entries exist and are distinct - split the gradient
                half_work = work[i] / 2
                adj.A_bar_nzval[A_idx] += half_work
                adj.A_bar_nzval[A_idx_t] += half_work
            elseif A_idx > 0
                adj.A_bar_nzval[A_idx] += work[i]
            elseif A_idx_t > 0
                adj.A_bar_nzval[A_idx_t] += work[i]
            end
        end

        fill!(work, zero(Tv))
        D_bar[j] = zero(Tv)
    end

    return adj
end

"""
    modified_cholesky_pullback(F::SparseLDLFactor, A::SparseMatrixCSC, L_bar, D_bar)

Compute the pullback (VJP) of the modified Cholesky factorization.

Returns ∂Loss/∂A as a SparseMatrixCSC with the same sparsity pattern as A.
"""
function modified_cholesky_pullback(
    F::SparseLDLFactor{Tv,Ti},
    A::SparseMatrixCSC{Tv,Ti},
    L_bar::Vector{Tv},
    D_bar::Vector{Tv},
) where {Tv,Ti}
    adj = SparseLDLAdjoint(F, A)
    modified_cholesky_adjoint!(adj, F, A, L_bar, D_bar)
    return get_A_bar(adj, F.symbolic.n)
end

"""
    logdet_pullback(F::SparseLDLFactor, A::SparseMatrixCSC)

Compute ∂(log|L̃DL̃ᵀ|)/∂A.

Since logdet = sum(log.(D)), we have ∂logdet/∂D = 1 ./ D.
"""
function logdet_pullback(F::SparseLDLFactor{Tv,Ti}, A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    n = F.symbolic.n
    nz_L = nnz_L(F.symbolic)

    D_bar = one(Tv) ./ F.D
    L_bar = zeros(Tv, nz_L)

    return modified_cholesky_pullback(F, A, L_bar, D_bar)
end

"""
    quadform_pullback(F::SparseLDLFactor, A::SparseMatrixCSC, v::Vector)

Compute ∂(½ pᵀ G⁻¹ p)/∂A where v = G⁻¹ p.

The quadratic form can be written as ½ y' D⁻¹ y where y = L̃⁻¹ p_perm.
Defining z = L̃' v_perm = D⁻¹ y, the derivatives are:
- ∂/∂D[j] = -½ z[j]²
- ∂/∂L̃[i,j] = -D[j] z[j] v_perm[i]  for i > j

These are then propagated through the modified Cholesky adjoint to get ∂/∂A.
"""
function quadform_pullback(
    F::SparseLDLFactor{Tv,Ti}, A::SparseMatrixCSC{Tv,Ti}, v::Vector{Tv}
) where {Tv,Ti}
    sym = F.symbolic
    n = sym.n
    perm = sym.perm
    Lcolptr = sym.Lcolptr
    Lrowval = sym.Lrowval
    Lnzval = F.Lnzval
    D = F.D
    nz_L = nnz_L(sym)

    # Permute v to work in the factorization's space
    v_perm = Vector{Tv}(undef, n)
    @inbounds for i in 1:n
        v_perm[i] = v[perm[i]]
    end

    # Compute z = L̃' v_perm (multiply by L̃ transpose)
    # z[j] = v_perm[j] + sum_{i>j} L̃[i,j] * v_perm[i]
    z = copy(v_perm)
    @inbounds for j in 1:n
        for p in Lcolptr[j]:(Lcolptr[j + 1] - 1)
            i = Lrowval[p]
            z[j] += Lnzval[p] * v_perm[i]
        end
    end

    # D_bar[j] = -½ z[j]²
    D_bar = Vector{Tv}(undef, n)
    @inbounds for j in 1:n
        D_bar[j] = -Tv(0.5) * z[j] * z[j]
    end

    # L_bar[i,j] = -D[j] * z[j] * v_perm[i] for i > j
    L_bar = Vector{Tv}(undef, nz_L)
    @inbounds for j in 1:n
        Dj_zj = D[j] * z[j]
        for p in Lcolptr[j]:(Lcolptr[j + 1] - 1)
            i = Lrowval[p]
            L_bar[p] = -Dj_zj * v_perm[i]
        end
    end

    return modified_cholesky_pullback(F, A, L_bar, D_bar)
end

"""
    hamiltonian_grad(F::SparseLDLFactor, A::SparseMatrixCSC, v::Vector)

Compute the combined gradient of the RMHMC Hamiltonian terms involving G:
    ½ log|G| + ½ pᵀ G⁻¹ p

where v = G⁻¹ p (precomputed).

Returns ∂H/∂A. For the force computation, you then apply:
    force_θ = -∂H/∂θ = -trace((∂H/∂A)' × ∂A/∂θ)
"""
function hamiltonian_grad(
    F::SparseLDLFactor{Tv,Ti}, A::SparseMatrixCSC{Tv,Ti}, v::Vector{Tv}
) where {Tv,Ti}
    # ∂(½ log|G|)/∂A = ½ * ∂(log|G|)/∂A
    # Note: logdet_pullback returns ∂(log|G|)/∂A, so we scale by 0.5
    logdet_grad = logdet_pullback(F, A)

    # ∂(½ pᵀ G⁻¹ p)/∂A (quadform_pullback already includes the ½ factor)
    quadform_grad = quadform_pullback(F, A, v)

    # Combined gradient: ∂(½ log|G| + ½ pᵀ G⁻¹ p)/∂A
    # Scale logdet_grad.nzval by 0.5
    result = SparseMatrixCSC(
        size(logdet_grad, 1),
        size(logdet_grad, 2),
        logdet_grad.colptr,
        logdet_grad.rowval,
        Tv(0.5) .* logdet_grad.nzval .+ quadform_grad.nzval,
    )
    return result
end

"""
    sparse_trace_symmetric(grad::SparseMatrixCSC, dA::SparseMatrixCSC)

Compute the trace inner product tr(grad' × dA) for symmetric matrices.

Both `grad` and `dA` should be symmetric matrices with equal values in both triangles.
The gradient entries represent true partial derivatives (not doubled).

The trace is computed by summing over all stored entries:

    tr = Σ_{i,j} grad[i,j] × dA[i,j]

For symmetric matrices, this correctly accounts for off-diagonal contributions
(each off-diagonal pair contributes twice, once from each triangle).

# Arguments
- `grad`: Gradient matrix from a pullback function (e.g., logdet_pullback, hamiltonian_grad)
- `dA`: Derivative of the input matrix w.r.t. some parameter (∂A/∂θ)

# Returns
- Scalar representing ∂f/∂θ where f is the function whose pullback produced `grad`

# Example
```julia
F = modified_cholesky(G, u, K)
H_grad = hamiltonian_grad(F, G, v)
dG_dtheta = ...  # your ∂G/∂θ matrix with same sparsity as G
dH_dtheta = sparse_trace_symmetric(H_grad, dG_dtheta)
```
"""
function sparse_trace_symmetric(
    grad::SparseMatrixCSC{Tv,Ti}, dA::SparseMatrixCSC{Tv,Ti}
) where {Tv,Ti}
    result = zero(Tv)
    n = size(grad, 1)
    @inbounds for j in 1:n
        for p in dA.colptr[j]:(dA.colptr[j + 1] - 1)
            i = dA.rowval[p]
            # Sum over all entries (both triangles)
            for q in grad.colptr[j]:(grad.colptr[j + 1] - 1)
                if grad.rowval[q] == i
                    result += grad.nzval[q] * dA.nzval[p]
                    break
                end
            end
        end
    end
    return result
end
