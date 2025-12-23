"""Definitions of custom block matrix types backed by static arrays."""

export SymPSDBlockTridiag, BlockUpperBidiag, BlockVector
export SymPSDBorderedBlockTridiag, BorderedBlockUpperBidiag, BorderedBlockVector
export to_block_vector,
    from_block_vector, to_bordered_block_vector, from_bordered_block_vector

struct SymPSDBlockTridiag{T,D,L,V<:AbstractVector{SMatrix{D,D,T,L}}} <: AbstractMatrix{T}
    diag_blocks::V
    super_blocks::V

    function SymPSDBlockTridiag{T,D}(
        diag_blocks::V, super_blocks::V
    ) where {T,D,L,V<:AbstractVector{SMatrix{D,D,T,L}}}
        if length(diag_blocks) != length(super_blocks) + 1
            throw(ArgumentError("Number of diagonal blocks must be one more than number \
                                 of super-diagonal blocks"))
        end
        return new{T,D,L,V}(diag_blocks, super_blocks)
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

struct BlockUpperBidiag{
    T,
    D,
    L,
    V1<:AbstractVector{UpperTriangular{T,SMatrix{D,D,T,L}}},
    V2<:AbstractVector{SMatrix{D,D,T,L}},
} <: AbstractMatrix{T}
    diag_blocks::V1
    super_blocks::V2

    function BlockUpperBidiag{T,D}(
        diag_blocks::V1, super_blocks::V2
    ) where {
        T,
        D,
        L,
        V1<:AbstractVector{UpperTriangular{T,SMatrix{D,D,T,L}}},
        V2<:AbstractVector{SMatrix{D,D,T,L}},
    }
        if length(diag_blocks) != length(super_blocks) + 1
            throw(ArgumentError("Number of diagonal blocks must be one more than number \
                                 of super-diagonal blocks"))
        end
        return new{T,D,L,V1,V2}(diag_blocks, super_blocks)
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

struct BlockVector{T,D,V<:AbstractVector{SVector{D,T}}} <: AbstractVector{T}
    blocks::V
end

# Convienence constructor from vector
function BlockVector{T,D}(v::Vector{T}) where {T,D}
    n = length(v)
    if n % D != 0
        throw(ArgumentError("Length of vector must be a multiple of block size D"))
    end
    K = div(n, D)
    blocks = [SVector{D,T}(v[(D * (k - 1) + 1):(D * k)]) for k in 1:K]
    return BlockVector{T,D}(blocks)
end

# Regular constructor
function BlockVector{T,D}(blocks::V) where {T,D,V<:AbstractVector{SVector{D,T}}}
    return BlockVector{T,D,V}(blocks)
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

##############################
#### BORDERED BLOCK TYPES ####
##############################

struct SymPSDBorderedBlockTridiag{
    T,
    D,
    P,
    LD,
    LP,
    LDP,
    VD<:AbstractVector{SMatrix{D,D,T,LD}},
    VB<:AbstractVector{SMatrix{D,P,T,LDP}},
} <: AbstractMatrix{T}
    diag_blocks::VD
    super_blocks::VD
    border_blocks::VB
    corner_block::SMatrix{P,P,T,LP}

    function SymPSDBorderedBlockTridiag{T,D,P}(
        diag_blocks::VD,
        super_blocks::VD,
        border_blocks::VB,
        corner_block::SMatrix{P,P,T,LP},
    ) where {
        T,
        D,
        P,
        LD,
        LP,
        LDP,
        VD<:AbstractVector{SMatrix{D,D,T,LD}},
        VB<:AbstractVector{SMatrix{D,P,T,LDP}},
    }
        if length(diag_blocks) != length(super_blocks) + 1
            throw(ArgumentError("Number of diagonal blocks must be one more than number \
                                 of super-diagonal blocks"))
        end
        if length(diag_blocks) != length(border_blocks)
            throw(
                ArgumentError(
                    "Number of border blocks must equal number of diagonal blocks"
                ),
            )
        end
        return new{T,D,P,LD,LP,LDP,VD,VB}(
            diag_blocks, super_blocks, border_blocks, corner_block
        )
    end
end

block_dim(::SymPSDBorderedBlockTridiag{T,D}) where {T,D} = D
param_dim(::SymPSDBorderedBlockTridiag{T,D,P}) where {T,D,P} = P

function Base.rand(
    rng::AbstractRNG, ::Type{SymPSDBorderedBlockTridiag{T,D,P}}, K::Int
) where {T,D,P}
    n = D * K + P
    U = zeros(T, n, n)
    @views for k in 1:K
        rand!(rng, U[(D * (k - 1) + 1):(D * k), (D * (k - 1) + 1):(D * k)])
        if k < K
            rand!(rng, U[(D * (k - 1) + 1):(D * k), (D * k + 1):(D * (k + 1))])
        end
        rand!(rng, U[(D * (k - 1) + 1):(D * k), (D * K + 1):n])
    end
    rand!(rng, U[(D * K + 1):n, (D * K + 1):n])
    S = U' * U + I

    diag_blocks = [
        SMatrix{D,D,T,D^2}(S[(D * (k - 1) + 1):(D * k), (D * (k - 1) + 1):(D * k)]) for
        k in 1:K
    ]
    super_blocks = [
        SMatrix{D,D,T,D^2}(S[(D * (k - 1) + 1):(D * k), (D * k + 1):(D * (k + 1))]) for
        k in 1:(K - 1)
    ]
    border_blocks = [
        SMatrix{D,P,T,D * P}(S[(D * (k - 1) + 1):(D * k), (D * K + 1):n]) for k in 1:K
    ]
    corner_block = SMatrix{P,P,T,P^2}(S[(D * K + 1):n, (D * K + 1):n])

    return SymPSDBorderedBlockTridiag{T,D,P}(
        diag_blocks, super_blocks, border_blocks, corner_block
    )
end

function Base.rand(::Type{SymPSDBorderedBlockTridiag{T,D,P}}, K::Int) where {T,D,P}
    return rand(Random.default_rng(), SymPSDBorderedBlockTridiag{T,D,P}, K)
end

function Base.size(A::SymPSDBorderedBlockTridiag)
    K = length(A.diag_blocks)
    D = block_dim(A)
    P = param_dim(A)
    return (K * D + P, K * D + P)
end

function Base.getindex(A::SymPSDBorderedBlockTridiag{T}, i::Int, j::Int) where {T}
    K = length(A.diag_blocks)
    D = block_dim(A)
    P = param_dim(A)
    n = K * D + P

    @boundscheck checkbounds(Bool, 1:n, i) || throw(BoundsError(A, (i, j)))
    @boundscheck checkbounds(Bool, 1:n, j) || throw(BoundsError(A, (i, j)))

    in_state_rows = i <= K * D
    in_state_cols = j <= K * D

    if in_state_rows && in_state_cols
        block_row = div(i - 1, D) + 1
        block_col = div(j - 1, D) + 1
        local_i = mod(i - 1, D) + 1
        local_j = mod(j - 1, D) + 1

        if block_row == block_col
            return A.diag_blocks[block_row][local_i, local_j]
        elseif block_col == block_row + 1
            return A.super_blocks[block_row][local_i, local_j]
        elseif block_row == block_col + 1
            return A.super_blocks[block_col][local_j, local_i]
        else
            return zero(T)
        end
    elseif in_state_rows && !in_state_cols
        block_row = div(i - 1, D) + 1
        local_i = mod(i - 1, D) + 1
        local_j = j - K * D
        return A.border_blocks[block_row][local_i, local_j]
    elseif !in_state_rows && in_state_cols
        block_col = div(j - 1, D) + 1
        local_i = i - K * D
        local_j = mod(j - 1, D) + 1
        return A.border_blocks[block_col][local_j, local_i]
    else
        local_i = i - K * D
        local_j = j - K * D
        return A.corner_block[local_i, local_j]
    end
end

Base.IndexStyle(::Type{<:SymPSDBorderedBlockTridiag}) = IndexCartesian()
Base.eltype(::SymPSDBorderedBlockTridiag{T}) where {T} = T

function Base.show(io::IO, ::MIME"text/plain", A::SymPSDBorderedBlockTridiag)
    K = length(A.diag_blocks)
    D = block_dim(A)
    P = param_dim(A)
    n = K * D + P
    return println(
        io,
        "SymPSDBorderedBlockTridiag of size $n x $n with $K state blocks of size $D x $D \
         and parameter block of size $P x $P",
    )
end

struct BorderedBlockUpperBidiag{
    T,
    D,
    P,
    LD,
    LP,
    LDP,
    V1<:AbstractVector{UpperTriangular{T,SMatrix{D,D,T,LD}}},
    V2<:AbstractVector{SMatrix{D,D,T,LD}},
    V3<:AbstractVector{SMatrix{D,P,T,LDP}},
} <: AbstractMatrix{T}
    diag_blocks::V1
    super_blocks::V2
    border_blocks::V3
    corner_factor::UpperTriangular{T,SMatrix{P,P,T,LP}}

    function BorderedBlockUpperBidiag{T,D,P}(
        diag_blocks::V1,
        super_blocks::V2,
        border_blocks::V3,
        corner_factor::UpperTriangular{T,SMatrix{P,P,T,LP}},
    ) where {
        T,
        D,
        P,
        LD,
        LP,
        LDP,
        V1<:AbstractVector{UpperTriangular{T,SMatrix{D,D,T,LD}}},
        V2<:AbstractVector{SMatrix{D,D,T,LD}},
        V3<:AbstractVector{SMatrix{D,P,T,LDP}},
    }
        if length(diag_blocks) != length(super_blocks) + 1
            throw(ArgumentError("Number of diagonal blocks must be one more than number \
                                 of super-diagonal blocks"))
        end
        if length(diag_blocks) != length(border_blocks)
            throw(
                ArgumentError(
                    "Number of border blocks must equal number of diagonal blocks"
                ),
            )
        end
        return new{T,D,P,LD,LP,LDP,V1,V2,V3}(
            diag_blocks, super_blocks, border_blocks, corner_factor
        )
    end
end

block_dim(::BorderedBlockUpperBidiag{T,D}) where {T,D} = D
param_dim(::BorderedBlockUpperBidiag{T,D,P}) where {T,D,P} = P

function Base.size(A::BorderedBlockUpperBidiag)
    K = length(A.diag_blocks)
    D = block_dim(A)
    P = param_dim(A)
    return (K * D + P, K * D + P)
end

function Base.getindex(A::BorderedBlockUpperBidiag{T}, i::Int, j::Int) where {T}
    K = length(A.diag_blocks)
    D = block_dim(A)
    P = param_dim(A)
    n = K * D + P

    @boundscheck checkbounds(Bool, 1:n, i) || throw(BoundsError(A, (i, j)))
    @boundscheck checkbounds(Bool, 1:n, j) || throw(BoundsError(A, (i, j)))

    in_state_rows = i <= K * D
    in_state_cols = j <= K * D

    if in_state_rows && in_state_cols
        block_row = div(i - 1, D) + 1
        block_col = div(j - 1, D) + 1
        local_i = mod(i - 1, D) + 1
        local_j = mod(j - 1, D) + 1

        if block_row == block_col
            return A.diag_blocks[block_row][local_i, local_j]
        elseif block_col == block_row + 1
            return A.super_blocks[block_row][local_i, local_j]
        else
            return zero(T)
        end
    elseif in_state_rows && !in_state_cols
        block_row = div(i - 1, D) + 1
        local_i = mod(i - 1, D) + 1
        local_j = j - K * D
        return A.border_blocks[block_row][local_i, local_j]
    elseif !in_state_rows && in_state_cols
        return zero(T)
    else
        local_i = i - K * D
        local_j = j - K * D
        return A.corner_factor[local_i, local_j]
    end
end

Base.IndexStyle(::Type{<:BorderedBlockUpperBidiag}) = IndexCartesian()
Base.eltype(::BorderedBlockUpperBidiag{T}) where {T} = T

function Base.show(io::IO, ::MIME"text/plain", A::BorderedBlockUpperBidiag)
    K = length(A.diag_blocks)
    D = block_dim(A)
    P = param_dim(A)
    n = K * D + P
    return println(
        io,
        "BorderedBlockUpperBidiag of size $n x $n with $K state blocks of size $D x $D \
         and corner factor of size $P x $P",
    )
end

struct BorderedBlockVector{T,D,P,V<:AbstractVector{SVector{D,T}}} <: AbstractVector{T}
    state_blocks::V
    param_block::SVector{P,T}
end

function BorderedBlockVector{T,D,P}(v::Vector{T}) where {T,D,P}
    n = length(v)
    K = div(n - P, D)
    if K * D + P != n
        throw(ArgumentError("Length of vector must be K*D + P for some integer K"))
    end
    state_blocks = [SVector{D,T}(v[(D * (k - 1) + 1):(D * k)]) for k in 1:K]
    param_block = SVector{P,T}(v[(K * D + 1):n])
    return BorderedBlockVector{T,D,P,Vector{SVector{D,T}}}(state_blocks, param_block)
end

function BorderedBlockVector{T,D,P}(
    state_blocks::V, param_block::SVector{P,T}
) where {T,D,P,V<:AbstractVector{SVector{D,T}}}
    return BorderedBlockVector{T,D,P,V}(state_blocks, param_block)
end

block_dim(::BorderedBlockVector{T,D}) where {T,D} = D
param_dim(::BorderedBlockVector{T,D,P}) where {T,D,P} = P

function Base.size(v::BorderedBlockVector)
    K = length(v.state_blocks)
    D = block_dim(v)
    P = param_dim(v)
    return (K * D + P,)
end

function Base.getindex(v::BorderedBlockVector{T}, i::Int) where {T}
    K = length(v.state_blocks)
    D = block_dim(v)
    P = param_dim(v)
    n = K * D + P

    @boundscheck checkbounds(Bool, 1:n, i) || throw(BoundsError(v, i))

    if i <= K * D
        block_idx = div(i - 1, D) + 1
        local_i = mod(i - 1, D) + 1
        return v.state_blocks[block_idx][local_i]
    else
        local_i = i - K * D
        return v.param_block[local_i]
    end
end

Base.IndexStyle(::Type{<:BorderedBlockVector}) = IndexCartesian()
Base.eltype(::BorderedBlockVector{T}) where {T} = T

function Base.show(io::IO, ::MIME"text/plain", v::BorderedBlockVector)
    K = length(v.state_blocks)
    D = block_dim(v)
    P = param_dim(v)
    return println(
        io,
        "BorderedBlockVector of length $(K * D + P) with $K state blocks of size $D \
         and parameter block of size $P",
    )
end

####################
#### CONVERTERS ####
####################

"""Convert a standard vector to a BlockVector."""
function to_block_vector(x::AbstractVector{T}, ::Val{D}) where {T,D}
    blocks = reinterpret(SVector{D,T}, x)
    return BlockVector{T,D}(blocks)
end

"""Convert a BlockVector to a standard vector."""
function from_block_vector(x_blocks::BlockVector{T,D}) where {T,D}
    K = length(x_blocks.blocks)
    x = Vector{T}(undef, K * D)
    @inbounds for k in 1:K
        x[(k - 1) * D .+ (1:D)] .= x_blocks.blocks[k]
    end
    return x
end

"""Convert a standard vector to a BorderedBlockVector."""
function to_bordered_block_vector(x::AbstractVector{T}, ::Val{D}, ::Val{P}) where {T,D,P}
    n = length(x)
    K = div(n - P, D)
    state_blocks = reinterpret(SVector{D,T}, @view x[1:(K * D)])
    param_block = SVector{P,T}(x[(K * D + 1):n])
    return BorderedBlockVector{T,D,P}(state_blocks, param_block)
end

"""Convert a BorderedBlockVector to a standard vector."""
function from_bordered_block_vector(x_blocks::BorderedBlockVector{T,D,P}) where {T,D,P}
    K = length(x_blocks.state_blocks)
    x = Vector{T}(undef, K * D + P)
    @inbounds for k in 1:K
        x[(k - 1) * D .+ (1:D)] .= x_blocks.state_blocks[k]
    end
    @inbounds x[(K * D + 1):end] .= x_blocks.param_block
    return x
end
