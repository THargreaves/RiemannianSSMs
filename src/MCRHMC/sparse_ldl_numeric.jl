"""
Numeric modified Cholesky factorization (Algorithm 1 from arXiv:1612.04093v2).

Computes L̃ D L̃ᵀ = P A Pᵀ + J where J is a diagonal modification
ensuring positive definiteness.
"""

"""
    get_sparse_entry(A::SparseMatrixCSC, i, j)

Get A[i,j] from a sparse matrix, returning zero if not present.
"""
@inline function get_sparse_entry(
    A::SparseMatrixCSC{Tv,Ti}, i::Integer, j::Integer
) where {Tv,Ti}
    for p in A.colptr[j]:(A.colptr[j + 1] - 1)
        if A.rowval[p] == i
            return A.nzval[p]
        end
    end
    return zero(Tv)
end

"""
    modified_cholesky!(F::SparseLDLFactor, A::SparseMatrixCSC)

Perform in-place modified Cholesky factorization.

Computes L̃ D L̃ᵀ = P A Pᵀ + J where P is the permutation from symbolic analysis
and J is a diagonal modification to ensure positive definiteness.

The modification uses the smooth absolute value function `sabs` for differentiability.
Pre-sabs diagonal values are stored in F.D_raw for use in the adjoint.
"""
function modified_cholesky!(
    F::SparseLDLFactor{Tv,Ti}, A::SparseMatrixCSC{Tv,Ti}
) where {Tv,Ti}
    sym = F.symbolic
    n = sym.n
    perm = sym.perm
    Lcolptr = sym.Lcolptr
    Lrowval = sym.Lrowval
    Lnzval = F.Lnzval
    D = F.D
    D_raw = F.D_raw
    u = F.u
    K = F.K
    work = F.work

    fill!(work, zero(Tv))
    fill!(Lnzval, zero(Tv))

    for j in 1:n
        pj = perm[j]
        D[j] = get_sparse_entry(A, pj, pj)
    end

    for j in 1:n
        pj = perm[j]

        for p in Lcolptr[j]:(Lcolptr[j + 1] - 1)
            i = Lrowval[p]
            pi = perm[i]

            val = get_sparse_entry(A, pi, pj)
            if val == zero(Tv)
                val = get_sparse_entry(A, pj, pi)
            end
            work[i] = val
        end

        for k in 1:(j - 1)
            L_jk = zero(Tv)
            for p in Lcolptr[k]:(Lcolptr[k + 1] - 1)
                if Lrowval[p] == j
                    L_jk = Lnzval[p]
                    break
                end
            end

            if L_jk != zero(Tv)
                D[j] -= L_jk * L_jk * D[k]

                for p in Lcolptr[k]:(Lcolptr[k + 1] - 1)
                    i = Lrowval[p]
                    if i > j
                        L_ik = Lnzval[p]
                        work[i] -= L_ik * L_jk * D[k]
                    end
                end
            end
        end

        D_raw[j] = D[j]

        if j > K
            u_idx = j - K
            D[j] = sabs(D[j], u[u_idx])
        end

        for p in Lcolptr[j]:(Lcolptr[j + 1] - 1)
            i = Lrowval[p]
            Lnzval[p] = work[i] / D[j]
            work[i] = zero(Tv)
        end
    end

    return F
end

"""
    modified_cholesky(A::SparseMatrixCSC, u::Vector, K::Integer=0; perm=nothing)

Compute the modified Cholesky factorization of a sparse symmetric matrix.

Arguments:
- `A`: Sparse symmetric matrix (negative Hessian)
- `u`: Regularization parameters, length n - K
- `K`: Number of leading columns known to be from positive definite block (default 0)
- `perm`: Optional fill-reducing permutation

Returns:
- `SparseLDLFactor` containing the factorization L̃ D L̃ᵀ = P A Pᵀ + J

Example:
```julia
A = your_sparse_hessian()
u = fill(1e-8, size(A, 1))
F = modified_cholesky(A, u)
L = get_L(F)  # Unit lower triangular factor
D = get_D(F)  # Diagonal matrix
```
"""
function modified_cholesky(
    A::SparseMatrixCSC{Tv,Ti}, u::Vector{<:Real}, K::Integer=0; perm=nothing
) where {Tv,Ti}
    symbolic = analyze_symbolic(A; perm=perm)
    u_typed = Tv.(u)
    F = SparseLDLFactor(symbolic, u_typed, K)
    modified_cholesky!(F, A)
    return F
end

"""
    modified_cholesky(A::SparseMatrixCSC, symbolic::SparseLDLSymbolic, u::Vector, K::Integer=0)

Compute modified Cholesky using pre-computed symbolic analysis.
Useful when factorizing multiple matrices with the same sparsity pattern.
"""
function modified_cholesky(
    A::SparseMatrixCSC{Tv,Ti},
    symbolic::SparseLDLSymbolic{Ti},
    u::Vector{<:Real},
    K::Integer=0,
) where {Tv,Ti}
    u_typed = Tv.(u)
    F = SparseLDLFactor(symbolic, u_typed, K)
    modified_cholesky!(F, A)
    return F
end
