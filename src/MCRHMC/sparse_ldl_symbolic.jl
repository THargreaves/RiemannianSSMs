"""
Symbolic analysis for sparse LDL^T factorization.

Computes fill-reducing permutation, elimination tree, and allocates
the sparsity pattern for L̃.
"""

using SparseArrays

"""
    compute_etree(A::SparseMatrixCSC)

Compute the elimination tree for the Cholesky factor of A.
A is assumed to be symmetric (stored as full or lower triangular).

Returns parent vector where parent[j] is the parent of node j in the
elimination tree, or 0 if j is a root.

The elimination tree satisfies: parent[j] = min{i > j : L[i,j] != 0}
where L is the Cholesky factor. This includes both direct connections
from A and fill-in connections.
"""
function compute_etree(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    n = Ti(size(A, 1))
    parent = zeros(Ti, n)

    for j in one(Ti):n
        for p in A.colptr[j]:(A.colptr[j + 1] - 1)
            i = A.rowval[p]
            if i > j
                # Walk up tree from j, updating parents to maintain minimum
                k = j
                while true
                    if parent[k] == zero(Ti)
                        # k is a root, connect it to i
                        parent[k] = i
                        break
                    elseif parent[k] >= i
                        # k's current parent is >= i; update to smaller value if needed
                        if i < parent[k]
                            parent[k] = i
                        end
                        break
                    else
                        # parent[k] < i, walk up the tree
                        k = parent[k]
                    end
                end
            end
        end
    end
    return parent
end

"""
    compute_reach(A::SparseMatrixCSC, j, parent, marker)

Compute which columns k < j contribute to column j through fill-in.
Updates marker and returns the set of columns.
"""
function compute_reach!(
    A::SparseMatrixCSC{Tv,Ti},
    j::Ti,
    parent::Vector{Ti},
    marker::Vector{Ti},
    reach::Vector{Ti},
) where {Tv,Ti}
    reach_count = zero(Ti)

    for p in A.colptr[j]:(A.colptr[j + 1] - 1)
        i = A.rowval[p]
        if i >= j
            continue
        end

        k = i
        while k != zero(Ti) && k < j && marker[k] != j
            reach_count += one(Ti)
            reach[reach_count] = k
            marker[k] = j
            k = parent[k]
        end
    end

    return reach_count
end

"""
    compute_column_counts(A::SparseMatrixCSC, parent::Vector)

Compute the number of nonzeros in each column of L (below the diagonal).
For column j, count how many rows i > j will have L[i,j] != 0.

Uses a row_added matrix (stored as set per column) to track which rows
have been added to each column, avoiding duplicates.
"""
function compute_column_counts(A::SparseMatrixCSC{Tv,Ti}, parent::Vector{Ti}) where {Tv,Ti}
    n = Ti(size(A, 1))

    # For each column k, track which rows have been added
    # Use a Vector of Sets for correctness
    rows_in_col = [Set{Ti}() for _ in 1:n]

    # For each A[i,j] with i > j, walk up the elimination tree
    # and mark that row i fills into each column along the path
    for j in one(Ti):n
        for p in A.colptr[j]:(A.colptr[j + 1] - 1)
            i = A.rowval[p]
            if i > j
                k = j
                while k != zero(Ti) && k < i
                    push!(rows_in_col[k], i)
                    k = parent[k]
                end
            end
        end
    end

    # Count unique rows per column
    counts = zeros(Ti, n)
    for k in one(Ti):n
        counts[k] = Ti(length(rows_in_col[k]))
    end

    return counts
end

"""
    compute_L_pattern!(Lrowval, Lcolptr, A, parent)

Fill in the row indices of L based on the elimination tree structure.
For each column j, find all rows i > j such that L[i,j] != 0.

Uses sets to collect unique row indices, then fills Lrowval.
"""
function compute_L_pattern!(
    Lrowval::Vector{Ti}, Lcolptr::Vector{Ti}, A::SparseMatrixCSC{Tv,Ti}, parent::Vector{Ti}
) where {Tv,Ti}
    n = Ti(size(A, 1))

    # Collect rows for each column using sets to avoid duplicates
    rows_in_col = [Set{Ti}() for _ in 1:n]

    for j in one(Ti):n
        for p in A.colptr[j]:(A.colptr[j + 1] - 1)
            i = A.rowval[p]
            if i > j
                k = j
                while k != zero(Ti) && k < i
                    push!(rows_in_col[k], i)
                    k = parent[k]
                end
            end
        end
    end

    # Fill Lrowval with sorted unique row indices
    for col in one(Ti):n
        rows = sort!(collect(rows_in_col[col]))
        for (idx, row) in enumerate(rows)
            Lrowval[Lcolptr[col] + idx - one(Ti)] = row
        end
    end
end

"""
    amd_order(A::SparseMatrixCSC)

Compute an approximate minimum degree ordering for A.
Uses Julia's built-in CHOLMOD interface.
"""
function amd_order(A::SparseMatrixCSC{Tv,Ti}) where {Tv,Ti}
    n = size(A, 1)

    A_sym = A + A'
    for i in 1:n
        A_sym[i, i] = one(Tv)
    end

    try
        F = ldlt(A_sym; check=false)
        if hasproperty(F, :p) || hasfield(typeof(F), :p)
            return Ti.(F.p)
        end
    catch
    end

    return Ti.(collect(1:n))
end

"""
    permute_matrix(A::SparseMatrixCSC, perm::Vector)

Compute P * A * P' where P is the permutation matrix.
Returns a new SparseMatrixCSC.
"""
function permute_matrix(A::SparseMatrixCSC{Tv,Ti}, perm::Vector{Ti}) where {Tv,Ti}
    n = Ti(size(A, 1))
    invp = invperm(perm)

    I = Ti[]
    J = Ti[]
    V = Tv[]

    for j in one(Ti):n
        for p in A.colptr[j]:(A.colptr[j + 1] - 1)
            i = A.rowval[p]
            new_i = invp[i]
            new_j = invp[j]
            push!(I, new_i)
            push!(J, new_j)
            push!(V, A.nzval[p])
        end
    end

    return sparse(I, J, V, n, n)
end

"""
    analyze_symbolic(A::SparseMatrixCSC; perm=nothing)

Perform symbolic analysis of sparse matrix A for LDL^T factorization.

Arguments:
- `A`: Sparse symmetric matrix (only lower triangle used)
- `perm`: Optional fill-reducing permutation. If nothing, AMD ordering is computed.

Returns:
- `SparseLDLSymbolic` structure containing pattern information
"""
function analyze_symbolic(
    A::SparseMatrixCSC{Tv,Ti}; perm::Union{Nothing,Vector{<:Integer}}=nothing
) where {Tv,Ti}
    n = Ti(size(A, 1))
    @assert size(A, 1) == size(A, 2) "Matrix must be square"

    if perm === nothing
        perm_use = amd_order(A)
    else
        perm_use = Ti.(perm)
    end
    invperm_use = Ti.(invperm(perm_use))

    A_sym = A + A'
    for i in 1:n
        A_sym[i, i] = A[i, i]
    end

    A_perm = permute_matrix(A_sym, perm_use)

    parent = compute_etree(A_perm)

    counts = compute_column_counts(A_perm, parent)

    Lcolptr = zeros(Ti, n + 1)
    Lcolptr[1] = one(Ti)
    for j in 1:n
        Lcolptr[j + 1] = Lcolptr[j] + counts[j]
    end
    nnz_L = Lcolptr[n + 1] - one(Ti)

    Lrowval = zeros(Ti, nnz_L)

    compute_L_pattern!(Lrowval, Lcolptr, A_perm, parent)

    return SparseLDLSymbolic(n, perm_use, invperm_use, parent, Lcolptr, Lrowval)
end
