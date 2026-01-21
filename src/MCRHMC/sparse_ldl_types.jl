"""
Data structures for sparse modified LDL^T factorization.
"""

using SparseArrays
using LinearAlgebra

"""
    SparseLDLSymbolic{Ti}

Symbolic (pattern-only) information for sparse LDL^T factorization.
Computed once from the sparsity pattern and reused for numeric factorizations.

Fields:
- `n`: Matrix dimension
- `perm`: Fill-reducing permutation (AMD ordering)
- `invperm`: Inverse permutation
- `parent`: Elimination tree parent pointers
- `Lcolptr`: Column pointers for L̃ (CSC format)
- `Lrowval`: Row indices for L̃ (CSC format), storing indices i > j for each column j
"""
struct SparseLDLSymbolic{Ti<:Integer}
    n::Ti
    perm::Vector{Ti}
    invperm::Vector{Ti}
    parent::Vector{Ti}
    Lcolptr::Vector{Ti}
    Lrowval::Vector{Ti}
end

Base.size(S::SparseLDLSymbolic) = (S.n, S.n)
nnz_L(S::SparseLDLSymbolic) = S.Lcolptr[end] - 1

"""
    SparseLDLFactor{Tv,Ti}

Numeric factorization result for sparse modified LDL^T.

The factorization represents: L̃ D L̃ᵀ = P A Pᵀ + J

where P is the permutation matrix corresponding to `symbolic.perm`,
L̃ is unit lower triangular, D is diagonal with positive entries,
and J is the diagonal modification.

Fields:
- `symbolic`: The symbolic factorization (sparsity pattern)
- `Lnzval`: Non-zero values of L̃ (unit diagonal implicit)
- `D`: Diagonal entries
- `D_raw`: Pre-sabs diagonal values (stored for adjoint computation)
- `u`: Regularization parameters (length n - K)
- `K`: Number of leading columns without regularization
- `work`: Workspace for numeric factorization
- `flag`: Integer workspace for marking
"""
mutable struct SparseLDLFactor{Tv<:AbstractFloat,Ti<:Integer}
    symbolic::SparseLDLSymbolic{Ti}
    Lnzval::Vector{Tv}
    D::Vector{Tv}
    D_raw::Vector{Tv}
    u::Vector{Tv}
    K::Ti
    work::Vector{Tv}
    flag::Vector{Ti}
end

Base.size(F::SparseLDLFactor) = size(F.symbolic)
Base.size(F::SparseLDLFactor, d::Integer) = d <= 2 ? F.symbolic.n : 1

"""
    SparseLDLFactor(symbolic, u, K)

Create a new numeric factorization structure with preallocated workspace.
"""
function SparseLDLFactor(
    symbolic::SparseLDLSymbolic{Ti}, u::Vector{Tv}, K::Integer
) where {Tv<:AbstractFloat,Ti<:Integer}
    n = symbolic.n
    nnz = nnz_L(symbolic)

    @assert length(u) == n - K "Regularization vector u must have length n - K = $(n - K), got $(length(u))"
    @assert 0 <= K <= n "K must satisfy 0 ≤ K ≤ n"

    Lnzval = zeros(Tv, nnz)
    D = zeros(Tv, n)
    D_raw = zeros(Tv, n)
    work = zeros(Tv, n)
    flag = zeros(Ti, n)

    return SparseLDLFactor(symbolic, Lnzval, D, D_raw, u, Ti(K), work, flag)
end

"""
    get_L(F::SparseLDLFactor)

Construct the unit lower triangular factor L̃ as a SparseMatrixCSC.
The diagonal is implicitly 1.
"""
function get_L(F::SparseLDLFactor{Tv,Ti}) where {Tv,Ti}
    n = F.symbolic.n
    sym = F.symbolic

    nz_L = nnz_L(sym) + n
    colptr = Vector{Ti}(undef, n + 1)
    rowval = Vector{Ti}(undef, nz_L)
    nzval = Vector{Tv}(undef, nz_L)

    colptr[1] = 1
    k = 1
    for j in 1:n
        rowval[k] = j
        nzval[k] = one(Tv)
        k += 1

        for p in sym.Lcolptr[j]:(sym.Lcolptr[j + 1] - 1)
            rowval[k] = sym.Lrowval[p]
            nzval[k] = F.Lnzval[p]
            k += 1
        end
        colptr[j + 1] = k
    end

    return SparseMatrixCSC(n, n, colptr, rowval, nzval)
end

"""
    get_D(F::SparseLDLFactor)

Return the diagonal as a Diagonal matrix.
"""
get_D(F::SparseLDLFactor) = Diagonal(F.D)

"""
    logdet(F::SparseLDLFactor)

Compute log|L̃ D L̃ᵀ| = log|D| = sum(log.(D)).
Since L̃ is unit triangular, det(L̃) = 1.
"""
function LinearAlgebra.logdet(F::SparseLDLFactor)
    return sum(log, F.D)
end

"""
    reconstruct(F::SparseLDLFactor)

Reconstruct the matrix P' * L̃ * D * L̃' * P from the factorization.
This equals A + J where J is the diagonal modification (in original ordering).

Returns a SparseMatrixCSC in the original (unpermuted) ordering.
"""
function reconstruct(F::SparseLDLFactor{Tv,Ti}) where {Tv,Ti}
    n = F.symbolic.n
    perm = F.symbolic.perm
    invperm = F.symbolic.invperm

    L = get_L(F)
    D_diag = get_D(F)
    G_perm = L * D_diag * L'

    I = Ti[]
    J = Ti[]
    V = Tv[]

    for j in 1:n
        for p in G_perm.colptr[j]:(G_perm.colptr[j + 1] - 1)
            i = G_perm.rowval[p]
            new_i = perm[i]
            new_j = perm[j]
            push!(I, new_i)
            push!(J, new_j)
            push!(V, G_perm.nzval[p])
        end
    end

    return sparse(I, J, V, n, n)
end

"""
    ldiv!(x::Vector, F::SparseLDLFactor, b::Vector)

Solve G x = b in-place where G = P' L̃ D L̃' P.

The solution proceeds as:
1. Permute: y = P b
2. Forward solve: L̃ z = y
3. Diagonal solve: D w = z
4. Backward solve: L̃' v = w
5. Unpermute: x = P' v
"""
function LinearAlgebra.ldiv!(
    x::Vector{Tv}, F::SparseLDLFactor{Tv,Ti}, b::Vector{Tv}
) where {Tv,Ti}
    sym = F.symbolic
    n = sym.n
    perm = sym.perm
    Lcolptr = sym.Lcolptr
    Lrowval = sym.Lrowval
    Lnzval = F.Lnzval
    D = F.D

    @assert length(b) == n "Vector length must match matrix dimension"
    @assert length(x) == n "Output vector length must match matrix dimension"

    # Step 1: Permute b -> work in permuted space
    @inbounds for i in 1:n
        x[i] = b[perm[i]]
    end

    # Step 2: Forward solve L̃ z = y (x holds y, then z)
    @inbounds for j in 1:n
        xj = x[j]
        for p in Lcolptr[j]:(Lcolptr[j + 1] - 1)
            i = Lrowval[p]
            x[i] -= Lnzval[p] * xj
        end
    end

    # Step 3: Diagonal solve D w = z
    @inbounds for i in 1:n
        x[i] /= D[i]
    end

    # Step 4: Backward solve L̃' v = w
    @inbounds for j in n:-1:1
        xj = x[j]
        for p in Lcolptr[j]:(Lcolptr[j + 1] - 1)
            i = Lrowval[p]
            xj -= Lnzval[p] * x[i]
        end
        x[j] = xj
    end

    # Step 5: Unpermute - copy to temp then back
    temp = copy(x)
    @inbounds for i in 1:n
        x[perm[i]] = temp[i]
    end

    return x
end

"""
    \\(F::SparseLDLFactor, b::Vector)

Solve G x = b where G = P' L̃ D L̃' P. Returns x = G⁻¹ b.
"""
function LinearAlgebra.:\(F::SparseLDLFactor{Tv,Ti}, b::Vector{Tv}) where {Tv,Ti}
    x = similar(b)
    ldiv!(x, F, b)
    return x
end

"""
    quadform(F::SparseLDLFactor, p::Vector)

Compute the quadratic form ½ pᵀ G⁻¹ p.

Returns (value, v) where:
- value = ½ pᵀ G⁻¹ p
- v = G⁻¹ p (needed for gradient: the pullback is -½ v vᵀ)
"""
function quadform(F::SparseLDLFactor{Tv,Ti}, p::Vector{Tv}) where {Tv,Ti}
    v = F \ p
    val = Tv(0.5) * dot(p, v)
    return val, v
end

"""
    compute_regularization_sensitivities(F::SparseLDLFactor)

Compute sensitivity of each regularized diagonal to changes in the regularization parameter u.

For each j > K (the regularized dimensions), computes:
    |d/dz[1/sabs(z; u_j)]|_{z=D_raw[j]}| = |sabs'(D_raw[j], u[j])| / sabs(D_raw[j], u[j])²

Higher sensitivity indicates that dimension contributes more to potential fixed-point
iteration instability. Used by the u-adaptation scheme to determine which dimensions
to adjust when divergences occur.

Returns a vector of length (n - K) containing the sensitivities.
"""
function compute_regularization_sensitivities(F::SparseLDLFactor{Tv,Ti}) where {Tv,Ti}
    K = F.K
    n = size(F, 1)
    n_reg = n - K
    sensitivities = zeros(Tv, n_reg)

    @inbounds for j in (K + 1):n
        u_idx = j - K
        raw_d = F.D_raw[j]
        u_j = F.u[u_idx]
        # Skip if inputs are invalid
        if !isfinite(raw_d) || !isfinite(u_j) || u_j <= 0
            sensitivities[u_idx] = zero(Tv)
            continue
        end
        # |d/dz[1/sabs(z; u)]| = |sabs'(z;u)| / sabs(z;u)^2
        sabs_val = sabs(raw_d, u_j)
        sabs_d = sabs_deriv(raw_d, u_j)
        s = abs(sabs_d) / (sabs_val * sabs_val)
        # Guard against NaN/Inf in result
        sensitivities[u_idx] = isfinite(s) ? s : zero(Tv)
    end

    return sensitivities
end

"""
    diagnose_regularization(F::SparseLDLFactor; top_k=5)

Diagnostic function to understand which dimensions need heavy regularization.
Returns information about the top_k dimensions with highest sensitivity.

For each problematic dimension, shows:
- Permuted index (position in factorization)
- Original index (position in unpermuted matrix)
- D_raw value (diagonal before sabs)
- u value (regularization parameter)
- D value (diagonal after sabs)
- Sensitivity
"""
function diagnose_regularization(F::SparseLDLFactor{Tv,Ti}; top_k::Int=5) where {Tv,Ti}
    sens = compute_regularization_sensitivities(F)
    K = F.K
    n = size(F, 1)
    perm = F.symbolic.perm

    # Get top_k indices by sensitivity
    top_indices = partialsortperm(sens, 1:min(top_k, length(sens)); rev=true)

    diagnostics = []
    for idx in top_indices
        j = idx + K  # Permuted index (1-based, in factorization order)
        orig_idx = perm[j]  # Original index before permutation
        raw_d = F.D_raw[j]
        u_j = F.u[idx]
        d_j = F.D[j]
        s = sens[idx]

        push!(
            diagnostics,
            (;
                perm_idx=j,
                orig_idx=orig_idx,
                D_raw=raw_d,
                u=u_j,
                D=d_j,
                sensitivity=s,
                regularization_active=(abs(raw_d) < u_j),
            ),
        )
    end

    return diagnostics
end
