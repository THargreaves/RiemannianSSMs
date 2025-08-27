"""Definitions of custom block matrix types backed by static arrays."""

export SymPSDBlockTridiag, BlockUpperBidiag, BlockVector

struct SymPSDBlockTridiag{T,D,L} <: AbstractMatrix{T}
    diag_blocks::Vector{SMatrix{D,D,T,L}}
    super_blocks::Vector{SMatrix{D,D,T,L}}

    function SymPSDBlockTridiag{T,D}(diag_blocks, super_blocks) where {T,D}
        L = D^2
        if length(diag_blocks) != length(super_blocks) + 1
            throw(ArgumentError("Number of diagonal blocks must be one more than number \
                                 of super-diagonal blocks"))
        end
        return new{T,D,L}(diag_blocks, super_blocks)
    end
end

block_dim(::SymPSDBlockTridiag{T,D}) where {T,D} = D

function Base.rand(rng::AbstractRNG, ::Type{SymPSDBlockTridiag{T,D}}, K::Int) where {T,D}
    U = zeros(T, D * K, D * K)
    @views for k in 1:K
        rand!(rng, U[(D * (k - 1) + 1):(D * k), (D * (k - 1) + 1):(D * k)])
        if k < K
            rand!(rng, U[(D * (k - 1) + 1):(D * k), (D * k + 1):(D * (k + 1))])
        end
    end
    S = U' * U + I
    diag_blocks = [
        SMatrix{D,D,T,D^2}(S[(D * (k - 1) + 1):(D * k), (D * (k - 1) + 1):(D * k)]) for
        k in 1:K
    ]
    super_blocks = [
        SMatrix{D,D,T,D^2}(S[(D * (k - 1) + 1):(D * k), (D * k + 1):(D * (k + 1))]) for
        k in 1:(K - 1)
    ]
    return SymPSDBlockTridiag{T,D}(diag_blocks, super_blocks)
end
function Base.rand(::Type{SymPSDBlockTridiag{T,D}}, K::Int) where {T,D}
    return rand(Random.default_rng(), SymPSDBlockTridiag{T,D}, K)
end

function Base.size(A::SymPSDBlockTridiag)
    K = length(A.diag_blocks)
    D = block_dim(A)
    return (K * D, K * D)
end

function Base.getindex(A::SymPSDBlockTridiag{T}, i::Int, j::Int) where {T}
    K = length(A.diag_blocks)
    D = block_dim(A)

    # Check bounds
    n = K * D
    @boundscheck checkbounds(Bool, 1:n, i) || throw(BoundsError(A, (i, j)))
    @boundscheck checkbounds(Bool, 1:n, j) || throw(BoundsError(A, (i, j)))

    # Determine which blocks the indices belong to
    block_row = div(i - 1, D) + 1
    block_col = div(j - 1, D) + 1

    # Local indices within the block
    local_i = mod(i - 1, D) + 1
    local_j = mod(j - 1, D) + 1

    if block_row == block_col
        # Diagonal block
        return A.diag_blocks[block_row][local_i, local_j]
    elseif block_col == block_row + 1
        # Super-diagonal block
        return A.super_blocks[block_row][local_i, local_j]
    elseif block_row == block_col + 1
        # Sub-diagonal block (transpose of super-diagonal due to symmetry)
        return A.super_blocks[block_col][local_j, local_i]
    else
        # Outside the tridiagonal structure
        return zero(T)
    end
end

Base.IndexStyle(::Type{<:SymPSDBlockTridiag}) = IndexCartesian()

Base.eltype(::SymPSDBlockTridiag{T}) where {T} = T

# Pretty printing
function Base.show(io::IO, ::MIME"text/plain", A::SymPSDBlockTridiag)
    K = length(A.diag_blocks)
    D = size(A.diag_blocks[1], 1)
    return println(
        io, "SymPSDBlockTridiag of size $(D*K) x $(D*K) with $K blocks of size $D x $D"
    )
end

struct BlockUpperBidiag{T,D,L} <: AbstractMatrix{T}
    diag_blocks::Vector{UpperTriangular{T,SMatrix{D,D,T,L}}}
    super_blocks::Vector{SMatrix{D,D,T,L}}

    function BlockUpperBidiag{T,D}(diag_blocks, super_blocks) where {T,D}
        L = D^2
        if length(diag_blocks) != length(super_blocks) + 1
            throw(ArgumentError("Number of diagonal blocks must be one more than number \
                                 of super-diagonal blocks"))
        end
        return new{T,D,L}(diag_blocks, super_blocks)
    end
end

block_dim(::BlockUpperBidiag{T,D}) where {T,D} = D

function Base.size(A::BlockUpperBidiag)
    K = length(A.diag_blocks)
    D = block_dim(A)
    return (K * D, K * D)
end

function Base.getindex(A::BlockUpperBidiag{T}, i::Int, j::Int) where {T}
    K = length(A.diag_blocks)
    D = block_dim(A)

    # Check bounds
    n = K * D
    @boundscheck checkbounds(Bool, 1:n, i) || throw(BoundsError(A, (i, j)))
    @boundscheck checkbounds(Bool, 1:n, j) || throw(BoundsError(A, (i, j)))

    # Determine which blocks the indices belong to
    block_row = div(i - 1, D) + 1
    block_col = div(j - 1, D) + 1

    # Local indices within the block
    local_i = mod(i - 1, D) + 1
    local_j = mod(j - 1, D) + 1

    if block_row == block_col
        # Diagonal block (upper triangular)
        return A.diag_blocks[block_row][local_i, local_j]
    elseif block_col == block_row + 1
        # Super-diagonal block
        return A.super_blocks[block_row][local_i, local_j]
    else
        # Outside the upper bidiagonal structure
        return zero(T)
    end
end

Base.IndexStyle(::Type{<:BlockUpperBidiag}) = IndexCartesian()

Base.eltype(::BlockUpperBidiag{T}) where {T} = T

function Base.show(io::IO, ::MIME"text/plain", A::BlockUpperBidiag)
    K = length(A.diag_blocks)
    D = size(A.diag_blocks[1], 1)
    return println(
        io, "BlockUpperBidiag of size $(D*K) x $(D*K) with $K blocks of size $D x $D"
    )
end

struct BlockVector{T,D} <: AbstractVector{T}
    blocks::Vector{SVector{D,T}}
end

block_dim(::BlockVector{T,D}) where {T,D} = D

function Base.size(v::BlockVector)
    K = length(v.blocks)
    D = block_dim(v)
    return (K * D,)
end

function Base.getindex(v::BlockVector{T}, i::Int) where {T}
    K = length(v.blocks)
    D = block_dim(v)

    # Check bounds
    n = K * D
    @boundscheck checkbounds(Bool, 1:n, i) || throw(BoundsError(v, i))

    # Determine which block the index belongs to
    block_idx = div(i - 1, D) + 1
    local_i = mod(i - 1, D) + 1

    return v.blocks[block_idx][local_i]
end

Base.IndexStyle(::Type{<:BlockVector}) = IndexCartesian()

Base.eltype(::BlockVector{T}) where {T} = T

function Base.show(io::IO, ::MIME"text/plain", v::BlockVector)
    K = length(v.blocks)
    D = size(v.blocks[1], 1)
    return println(io, "BlockVector of length $(D*K) with $K blocks of size $D")
end
