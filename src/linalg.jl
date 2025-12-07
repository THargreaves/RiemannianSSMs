"""Optimised linear algebra routines for symmetric PSD block tridiagonal matrices."""

export block_tridiag_selected_inv

function LinearAlgebra.cholesky(S::SymPSDBlockTridiag{T,D}) where {T,D}
    K = length(S.diag_blocks)

    # TODO: can be done in-place to reduce allocation
    U_diag = Vector{UpperTriangular{T,SMatrix{D,D,T,D^2}}}(undef, K)
    U_supdiag = Vector{SMatrix{D,D,T,D^2}}(undef, K - 1)

    # Perform in-place Cholesky decomposition
    @inbounds begin
        U_diag[1] = cholesky(Symmetric(S.diag_blocks[1])).U
        if K > 1
            U_supdiag[1] = U_diag[1]' \ S.super_blocks[1]
        end

        for k in 2:K
            # Schur complement update
            S̄ = Symmetric(S.diag_blocks[k] - U_supdiag[k - 1]' * U_supdiag[k - 1])
            U_diag[k] = cholesky(S̄).U
            if k < K
                U_supdiag[k] = U_diag[k]' \ S.super_blocks[k]
            end
        end
    end

    return Cholesky(BlockUpperBidiag{T,D}(U_diag, U_supdiag), 'U', 0)
end

function LinearAlgebra.cholesky!(S::SymPSDBlockTridiag{T,D}) where {T,D}
    K = length(S.diag_blocks)

    U_diag = reinterpret(UpperTriangular{T,SMatrix{D,D,T,D^2}}, S.diag_blocks)
    U_supdiag = S.super_blocks

    # Perform in-place Cholesky decomposition
    @inbounds begin
        U_diag[1] = cholesky(Symmetric(S.diag_blocks[1])).U
        if K > 1
            U_supdiag[1] = U_diag[1]' \ S.super_blocks[1]
        end

        for k in 2:K
            # Schur complement update
            S̄ = Symmetric(S.diag_blocks[k] - U_supdiag[k - 1]' * U_supdiag[k - 1])
            U_diag[k] = cholesky(S̄).U
            if k < K
                U_supdiag[k] = U_diag[k]' \ S.super_blocks[k]
            end
        end
    end

    return Cholesky(BlockUpperBidiag{T,D}(U_diag, U_supdiag), 'U', 0)
end

function Base.:*(L::Adjoint{T,<:BlockUpperBidiag{T,D}}, x::BlockVector{T,D}) where {T,D}
    U_diag = L.parent.diag_blocks
    U_supdiag = L.parent.super_blocks
    x_blocks = x.blocks
    K = length(U_diag)

    y_blocks = Vector{SVector{D,T}}(undef, K)

    @inbounds begin
        y_blocks[1] = U_diag[1]' * x_blocks[1]
        for i in 2:K
            y_blocks[i] = U_diag[i]' * x_blocks[i] + U_supdiag[i - 1]' * x_blocks[i - 1]
        end
    end

    return BlockVector{T,D}(y_blocks)
end

# Upper backward solve
function Base.:\(U::BlockUpperBidiag{T,D}, x::BlockVector{T,D}) where {T,D}
    U_diag = U.diag_blocks
    U_supdiag = U.super_blocks
    x_blocks = x.blocks
    K = length(U_diag)

    y_blocks = Vector{SVector{D,T}}(undef, K)

    @inbounds begin
        y_blocks[K] = U_diag[K] \ x_blocks[K]
        for i in (K - 1):-1:1
            rhs = x_blocks[i] - U_supdiag[i] * y_blocks[i + 1]
            y_blocks[i] = U_diag[i] \ rhs
        end
    end

    return BlockVector{T,D}(y_blocks)
end

# Lower forward solve
function Base.:\(L::Adjoint{T,<:BlockUpperBidiag{T,D}}, x::BlockVector{T,D}) where {T,D}
    U_diag = L.parent.diag_blocks
    U_supdiag = L.parent.super_blocks
    x_blocks = x.blocks
    K = length(U_diag)

    y_blocks = Vector{SVector{D,T}}(undef, K)

    @inbounds begin
        y_blocks[1] = U_diag[1]' \ x_blocks[1]
        for i in 2:K
            rhs = x_blocks[i] - U_supdiag[i - 1]' * y_blocks[i - 1]
            y_blocks[i] = U_diag[i]' \ rhs
        end
    end

    return BlockVector{T,D}(y_blocks)
end

# TODO: decide whether BlockUpperBidiag composes an UpperTriangular or _is_ one
function Base.:\(F::Cholesky{T,<:BlockUpperBidiag{T,D}}, x::BlockVector{T,D}) where {T,D}
    y = F.factors' \ x
    z = F.factors \ y
    return z
end

function LinearAlgebra.logdet(F::Cholesky{T,<:BlockUpperBidiag{T,D}}) where {T,D}
    U_diag = F.factors.diag_blocks
    K = length(U_diag)
    U_log_det = 0.0
    @inbounds for k in 1:K
        U_log_det += sum(log, diag(U_diag[k]))
    end
    return 2 * U_log_det
end

###########################
#### IN-PLACE VARIANTS ####
###########################

# Upper backward solve
function LinearAlgebra.ldiv!(U::BlockUpperBidiag{T,D}, x::BlockVector{T,D}) where {T,D}
    U_diag = U.diag_blocks
    U_supdiag = U.super_blocks
    x_blocks = x.blocks
    K = length(U_diag)

    @inbounds begin
        x_blocks[K] = U_diag[K] \ x_blocks[K]
        for i in (K - 1):-1:1
            x_blocks[i] -= U_supdiag[i] * x_blocks[i + 1]
            x_blocks[i] = U_diag[i] \ x_blocks[i]
        end
    end

    return x
end

# Lower forward solve
function LinearAlgebra.ldiv!(
    L::Adjoint{T,<:BlockUpperBidiag{T,D}}, x::BlockVector{T,D}
) where {T,D}
    U_diag = L.parent.diag_blocks
    U_supdiag = L.parent.super_blocks
    x_blocks = x.blocks
    K = length(U_diag)

    @inbounds begin
        x_blocks[1] = U_diag[1]' \ x_blocks[1]
        for i in 2:K
            x_blocks[i] -= U_supdiag[i - 1]' * x_blocks[i - 1]
            x_blocks[i] = U_diag[i]' \ x_blocks[i]
        end
    end

    return x
end

# Cholesky solve
function LinearAlgebra.ldiv!(
    F::Cholesky{T,<:BlockUpperBidiag{T,D}}, x::BlockVector{T,D}
) where {T,D}
    ldiv!(F.factors', x)
    ldiv!(F.factors, x)
    return x
end

"""
Compute the tridiagonal elements of the inverse of a symmetric PSD block tridiagonal matrix.
"""
function block_tridiag_selected_inv(S::SymPSDBlockTridiag{T,D}) where {T,D}
    K = length(S.diag_blocks)
    # TODO: should try to avoid allocations here
    S_inv_diag = Vector{SMatrix{D,D,T,D^2}}(undef, K)
    S_inv_super = Vector{SMatrix{D,D,T,D^2}}(undef, K - 1)

    @inbounds begin
        # Forward elimination
        S_forward = Vector{SMatrix{D,D,T,D^2}}(undef, K)
        S_forward[1] = zeros(SMatrix{D,D,T,D^2})
        for k in 2:K
            A = S.diag_blocks[k - 1]
            B = S.super_blocks[k - 1]

            F = cholesky(Symmetric(A - S_forward[k - 1]))
            S_forward[k] = B' * (F \ B)
        end

        # Backward elimination
        S_backward = Vector{SMatrix{D,D,T,D^2}}(undef, K)
        S_backward[K] = zeros(SMatrix{D,D,T,D^2})
        for k in (K - 1):-1:1
            A = S.diag_blocks[k + 1]
            B = S.super_blocks[k]
            F = cholesky(Symmetric(A - S_backward[k + 1]))
            S_backward[k] = B * (F \ B')
        end

        # Compute diagonal blocks of the inverse
        for k in 1:K
            A = S.diag_blocks[k]
            F = cholesky(Symmetric(A - S_forward[k] - S_backward[k]))
            S_inv_diag[k] = inv(F)
        end

        # Compute off-diagonal blocks of the inverse
        for k in 1:(K - 1)
            A = S.diag_blocks[k + 1]
            B = S.super_blocks[k]
            F = cholesky(Symmetric(A - S_backward[k + 1]))
            S_inv_super[k] = -S_inv_diag[k] * (B / F)
        end
    end

    return SymPSDBlockTridiag{T,D}(S_inv_diag, S_inv_super)
end

# Vector-vector operations
for op in (:+, :-)
    @eval function Base.$(op)(x::BlockVector{T,D}, y::BlockVector{T,D}) where {T<:Number,D}
        K = length(x.blocks)
        @assert K == length(y.blocks)

        z_blocks = Vector{SVector{D,T}}(undef, K)
        @inbounds for k in 1:K
            z_blocks[k] = $op(x.blocks[k], y.blocks[k])
        end

        return BlockVector{T,D}(z_blocks)
    end
end

# Vector-scalar operations
for op in (:+, :-, :*, :/)
    @eval function Base.$(op)(x::BlockVector{T,D}, a::T) where {T<:Number,D}
        K = length(x.blocks)

        z_blocks = Vector{SVector{D,T}}(undef, K)
        @inbounds for k in 1:K
            z_blocks[k] = $op(x.blocks[k], a)
        end

        return BlockVector{T,D}(z_blocks)
    end

    @eval function Base.$(op)(a::T, x::BlockVector{T,D}) where {T<:Number,D}
        K = length(x.blocks)

        z_blocks = Vector{SVector{D,T}}(undef, K)
        @inbounds for k in 1:K
            z_blocks[k] = $op(a, x.blocks[k])
        end

        return BlockVector{T,D}(z_blocks)
    end
end

Base.copy(x::BlockVector{T,D}) where {T,D} = BlockVector{T,D}(copy.(x.blocks))
