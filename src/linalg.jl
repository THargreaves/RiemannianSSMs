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

#########################################
#### BORDERED BLOCK TRIDIAGONAL OPS ####
#########################################

export bordered_block_selected_inv

function LinearAlgebra.cholesky(S::SymPSDBorderedBlockTridiag{T,D,P}) where {T,D,P}
    K = length(S.diag_blocks)

    # Step 1: Factor the tridiagonal part
    U_diag = Vector{UpperTriangular{T,SMatrix{D,D,T,D^2}}}(undef, K)
    U_supdiag = Vector{SMatrix{D,D,T,D^2}}(undef, K - 1)

    @inbounds begin
        U_diag[1] = cholesky(Symmetric(S.diag_blocks[1])).U
        if K > 1
            U_supdiag[1] = U_diag[1]' \ S.super_blocks[1]
        end

        for k in 2:K
            S̄ = Symmetric(S.diag_blocks[k] - U_supdiag[k - 1]' * U_supdiag[k - 1])
            U_diag[k] = cholesky(S̄).U
            if k < K
                U_supdiag[k] = U_diag[k]' \ S.super_blocks[k]
            end
        end
    end

    # Step 2: Compute W = L_T^{-T} B (forward solve each column of B against L_T')
    W_blocks = Vector{SMatrix{D,P,T,D * P}}(undef, K)
    @inbounds begin
        W_blocks[1] = U_diag[1]' \ S.border_blocks[1]
        for k in 2:K
            rhs = S.border_blocks[k] - U_supdiag[k - 1]' * W_blocks[k - 1]
            W_blocks[k] = U_diag[k]' \ rhs
        end
    end

    # Step 3: Compute Schur complement S = C - W'W
    schur = S.corner_block
    @inbounds for k in 1:K
        schur = schur - W_blocks[k]' * W_blocks[k]
    end

    # Step 4: Factor the Schur complement
    U_corner = cholesky(Symmetric(schur)).U

    return Cholesky(
        BorderedBlockUpperBidiag{T,D,P}(U_diag, U_supdiag, W_blocks, U_corner), 'U', 0
    )
end

function LinearAlgebra.logdet(
    F::Cholesky{T,<:BorderedBlockUpperBidiag{T,D,P}}
) where {T,D,P}
    U = F.factors
    K = length(U.diag_blocks)
    log_det = zero(T)

    @inbounds for k in 1:K
        log_det += sum(log, diag(U.diag_blocks[k]))
    end
    log_det += sum(log, diag(U.corner_factor))

    return 2 * log_det
end

# Lower multiplication: L' * x where L' is lower triangular
function Base.:*(
    L::Adjoint{T,<:BorderedBlockUpperBidiag{T,D,P}}, x::BorderedBlockVector{T,D,P}
) where {T,D,P}
    U = L.parent
    K = length(U.diag_blocks)

    y_state_blocks = Vector{SVector{D,T}}(undef, K)
    @inbounds begin
        y_state_blocks[1] = U.diag_blocks[1]' * x.state_blocks[1]
        for k in 2:K
            y_state_blocks[k] =
                U.diag_blocks[k]' * x.state_blocks[k] +
                U.super_blocks[k - 1]' * x.state_blocks[k - 1]
        end
    end

    # y_C = W' * x_T + L_C' * x_C
    y_param = U.corner_factor' * x.param_block
    @inbounds for k in 1:K
        y_param = y_param + U.border_blocks[k]' * x.state_blocks[k]
    end

    return BorderedBlockVector{T,D,P}(y_state_blocks, y_param)
end

# Upper backward solve: U \ x
function Base.:\(
    U::BorderedBlockUpperBidiag{T,D,P}, x::BorderedBlockVector{T,D,P}
) where {T,D,P}
    K = length(U.diag_blocks)

    # Step 1: z_C = U_C \ x_C
    z_param = U.corner_factor \ x.param_block

    # Step 2: z_T = U_T \ (x_T - W * z_C)
    z_state_blocks = Vector{SVector{D,T}}(undef, K)
    @inbounds begin
        rhs_K = x.state_blocks[K] - U.border_blocks[K] * z_param
        z_state_blocks[K] = U.diag_blocks[K] \ rhs_K
        for k in (K - 1):-1:1
            rhs =
                x.state_blocks[k] - U.super_blocks[k] * z_state_blocks[k + 1] -
                U.border_blocks[k] * z_param
            z_state_blocks[k] = U.diag_blocks[k] \ rhs
        end
    end

    return BorderedBlockVector{T,D,P}(z_state_blocks, z_param)
end

# Lower forward solve: L' \ x where L = U'
function Base.:\(
    L::Adjoint{T,<:BorderedBlockUpperBidiag{T,D,P}}, x::BorderedBlockVector{T,D,P}
) where {T,D,P}
    U = L.parent
    K = length(U.diag_blocks)

    # Step 1: y_T = L_T \ x_T (forward solve against L_T = U_T')
    y_state_blocks = Vector{SVector{D,T}}(undef, K)
    @inbounds begin
        y_state_blocks[1] = U.diag_blocks[1]' \ x.state_blocks[1]
        for k in 2:K
            rhs = x.state_blocks[k] - U.super_blocks[k - 1]' * y_state_blocks[k - 1]
            y_state_blocks[k] = U.diag_blocks[k]' \ rhs
        end
    end

    # Step 2: y_C = L_C \ (x_C - W' * y_T)
    rhs_param = x.param_block
    @inbounds for k in 1:K
        rhs_param = rhs_param - U.border_blocks[k]' * y_state_blocks[k]
    end
    y_param = U.corner_factor' \ rhs_param

    return BorderedBlockVector{T,D,P}(y_state_blocks, y_param)
end

# Full Cholesky solve
function Base.:\(
    F::Cholesky{T,<:BorderedBlockUpperBidiag{T,D,P}}, x::BorderedBlockVector{T,D,P}
) where {T,D,P}
    y = F.factors' \ x
    z = F.factors \ y
    return z
end

# In-place upper backward solve
function LinearAlgebra.ldiv!(
    U::BorderedBlockUpperBidiag{T,D,P}, x::BorderedBlockVector{T,D,P}
) where {T,D,P}
    K = length(U.diag_blocks)

    # Step 1: x_C = U_C \ x_C
    x_param = U.corner_factor \ x.param_block

    # Step 2: x_T = U_T \ (x_T - W * x_C)
    @inbounds begin
        x.state_blocks[K] -= U.border_blocks[K] * x_param
        x.state_blocks[K] = U.diag_blocks[K] \ x.state_blocks[K]
        for k in (K - 1):-1:1
            x.state_blocks[k] -= U.super_blocks[k] * x.state_blocks[k + 1]
            x.state_blocks[k] -= U.border_blocks[k] * x_param
            x.state_blocks[k] = U.diag_blocks[k] \ x.state_blocks[k]
        end
    end

    return BorderedBlockVector{T,D,P}(x.state_blocks, x_param)
end

# In-place lower forward solve
function LinearAlgebra.ldiv!(
    L::Adjoint{T,<:BorderedBlockUpperBidiag{T,D,P}}, x::BorderedBlockVector{T,D,P}
) where {T,D,P}
    U = L.parent
    K = length(U.diag_blocks)

    @inbounds begin
        x.state_blocks[1] = U.diag_blocks[1]' \ x.state_blocks[1]
        for k in 2:K
            x.state_blocks[k] -= U.super_blocks[k - 1]' * x.state_blocks[k - 1]
            x.state_blocks[k] = U.diag_blocks[k]' \ x.state_blocks[k]
        end
    end

    # x_C = L_C \ (x_C - W' * x_T)
    x_param = x.param_block
    @inbounds for k in 1:K
        x_param = x_param - U.border_blocks[k]' * x.state_blocks[k]
    end
    x_param = U.corner_factor' \ x_param

    return BorderedBlockVector{T,D,P}(x.state_blocks, x_param)
end

# In-place Cholesky solve
function LinearAlgebra.ldiv!(
    F::Cholesky{T,<:BorderedBlockUpperBidiag{T,D,P}}, x::BorderedBlockVector{T,D,P}
) where {T,D,P}
    x = ldiv!(F.factors', x)
    x = ldiv!(F.factors, x)
    return x
end

# BorderedBlockVector arithmetic
for op in (:+, :-)
    @eval function Base.$(op)(
        x::BorderedBlockVector{T,D,P}, y::BorderedBlockVector{T,D,P}
    ) where {T<:Number,D,P}
        K = length(x.state_blocks)
        @assert K == length(y.state_blocks)

        z_state_blocks = Vector{SVector{D,T}}(undef, K)
        @inbounds for k in 1:K
            z_state_blocks[k] = $op(x.state_blocks[k], y.state_blocks[k])
        end
        z_param = $op(x.param_block, y.param_block)

        return BorderedBlockVector{T,D,P}(z_state_blocks, z_param)
    end
end

for op in (:+, :-, :*, :/)
    @eval function Base.$(op)(x::BorderedBlockVector{T,D,P}, a::T) where {T<:Number,D,P}
        K = length(x.state_blocks)

        z_state_blocks = Vector{SVector{D,T}}(undef, K)
        @inbounds for k in 1:K
            z_state_blocks[k] = $op(x.state_blocks[k], a)
        end
        z_param = $op(x.param_block, a)

        return BorderedBlockVector{T,D,P}(z_state_blocks, z_param)
    end

    @eval function Base.$(op)(a::T, x::BorderedBlockVector{T,D,P}) where {T<:Number,D,P}
        K = length(x.state_blocks)

        z_state_blocks = Vector{SVector{D,T}}(undef, K)
        @inbounds for k in 1:K
            z_state_blocks[k] = $op(a, x.state_blocks[k])
        end
        z_param = $op(a, x.param_block)

        return BorderedBlockVector{T,D,P}(z_state_blocks, z_param)
    end
end

function Base.copy(x::BorderedBlockVector{T,D,P}) where {T,D,P}
    return BorderedBlockVector{T,D,P}(copy.(x.state_blocks), x.param_block)
end

"""
Compute the selected inverse of a bordered block tridiagonal matrix.

Returns a tuple (A_inv_tridiag, T_inv_B_blocks, S_inv) where:
- A_inv_tridiag: SymPSDBlockTridiag with the tridiagonal blocks of the state-state inverse
  (includes the correction term W S^{-1} W' where W = T^{-1}B)
- T_inv_B_blocks: Vector of D×P matrices representing T^{-1}B
- S_inv: P×P matrix representing the Schur complement inverse

The full inverse has structure:
    A^{-1} = [ T^{-1} + T^{-1}B S^{-1}B'T^{-1}  |  -T^{-1}B S^{-1} ]
             [ -S^{-1}B'T^{-1}                   |   S^{-1}         ]
"""
function bordered_block_selected_inv(A::SymPSDBorderedBlockTridiag{T,D,P}) where {T,D,P}
    K = length(A.diag_blocks)

    # Extract the tridiagonal part
    T_tridiag = SymPSDBlockTridiag{T,D}(A.diag_blocks, A.super_blocks)

    # Compute the tridiagonal selected inverse of T (this is T^{-1}, not yet A^{-1})
    T_inv_tridiag = block_tridiag_selected_inv(T_tridiag)

    # Compute W = T^{-1}B by solving T W = B
    F_T = cholesky(T_tridiag)
    W_blocks = Vector{SMatrix{D,P,T,D * P}}(undef, K)
    @inbounds for p in 1:P
        # Extract column p of B as a BlockVector
        col_blocks = [A.border_blocks[k][:, p] for k in 1:K]
        b_col = BlockVector{T,D}(col_blocks)

        # Solve T x = b
        x_col = F_T \ b_col

        # Store result
        for k in 1:K
            if p == 1
                W_blocks[k] = SMatrix{D,P,T,D * P}(
                    hcat(x_col.blocks[k], zeros(SMatrix{D,P - 1,T,(P - 1) * D}))
                )
            else
                # Update column p
                old_mat = Matrix(W_blocks[k])
                old_mat[:, p] = x_col.blocks[k]
                W_blocks[k] = SMatrix{D,P,T,D * P}(old_mat)
            end
        end
    end

    # Compute Schur complement: S = C - B'W
    schur = A.corner_block
    @inbounds for k in 1:K
        schur = schur - A.border_blocks[k]' * W_blocks[k]
    end
    S_inv = inv(cholesky(Symmetric(schur)))

    # Now compute the corrected state-state inverse:
    # A^{-1}_{zz} = T^{-1} + W S^{-1} W'
    # For diagonal block k: correction is W_k S^{-1} W_k'
    # For off-diagonal block (k, k+1): correction is W_k S^{-1} W_{k+1}'
    A_inv_diag = Vector{SMatrix{D,D,T,D^2}}(undef, K)
    A_inv_super = Vector{SMatrix{D,D,T,D^2}}(undef, K - 1)

    @inbounds for k in 1:K
        # Diagonal correction: W_k S^{-1} W_k'
        correction = W_blocks[k] * S_inv * W_blocks[k]'
        A_inv_diag[k] = T_inv_tridiag.diag_blocks[k] + correction
    end

    @inbounds for k in 1:(K - 1)
        # Off-diagonal correction: W_k S^{-1} W_{k+1}'
        correction = W_blocks[k] * S_inv * W_blocks[k + 1]'
        A_inv_super[k] = T_inv_tridiag.super_blocks[k] + correction
    end

    A_inv_tridiag = SymPSDBlockTridiag{T,D}(A_inv_diag, A_inv_super)

    return (A_inv_tridiag, W_blocks, S_inv)
end
