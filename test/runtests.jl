using Test
using TestItems
using TestItemRunner

@run_package_tests

@testitem "Cholesky factorisation" begin
    using RiemannianSSMs
    using LinearAlgebra
    using StableRNGs

    D = 3
    K = 5
    rng = StableRNG(1234)

    A = rand(rng, SymPSDBlockTridiag{Float64,D}, K)
    F = cholesky(A)

    # @test isa(F, Cholesky{Float64,BlockUpperBidiag{Float64,D,D^2}})

    # Compare to dense
    A_dense = Matrix(A)
    F_dense = cholesky(A_dense)

    @test maximum(abs.(F.U .- F_dense.U)) < 1e-10
end

@testitem "Lower multiplication" begin
    using RiemannianSSMs
    using LinearAlgebra
    using StableRNGs
    using StaticArrays

    D = 3
    K = 5
    rng = StableRNG(1234)

    U = BlockUpperBidiag{Float64,D}(
        UpperTriangular.([rand(rng, SMatrix{D,D,Float64,D^2}) for k in 1:K]),
        [rand(rng, SMatrix{D,D,Float64,D^2}) for k in 1:(K - 1)],
    )
    x = BlockVector([rand(rng, SVector{D,Float64}) for k in 1:K])
    y = U' * x

    @test isa(y, BlockVector{Float64,D})

    U_dense = Matrix(U)
    x_dense = Vector(x)
    y_dense = U_dense' * x_dense

    @test maximum(abs.(y .- y_dense)) < 1e-10
end

@testitem "Upper backward solve" begin
    using RiemannianSSMs
    using LinearAlgebra
    using StableRNGs
    using StaticArrays

    D = 3
    K = 5
    rng = StableRNG(1234)

    U = BlockUpperBidiag{Float64,D}(
        UpperTriangular.([rand(rng, SMatrix{D,D,Float64,D^2}) for k in 1:K]),
        [rand(rng, SMatrix{D,D,Float64,D^2}) for k in 1:(K - 1)],
    )
    x = BlockVector([rand(rng, SVector{D,Float64}) for k in 1:K])
    y = U \ x

    @test isa(y, BlockVector{Float64,D})

    U_dense = Matrix(U)
    x_dense = Vector(x)
    y_dense = U_dense \ x_dense

    @test maximum(abs.(y .- y_dense)) < 1e-10
end

@testitem "Lower forward solve" begin
    using RiemannianSSMs
    using LinearAlgebra
    using StableRNGs
    using StaticArrays

    D = 3
    K = 5
    rng = StableRNG(1234)

    U = BlockUpperBidiag{Float64,D}(
        UpperTriangular.([rand(rng, SMatrix{D,D,Float64,D^2}) for k in 1:K]),
        [rand(rng, SMatrix{D,D,Float64,D^2}) for k in 1:(K - 1)],
    )
    x = BlockVector([rand(rng, SVector{D,Float64}) for k in 1:K])
    y = U' \ x

    @test isa(y, BlockVector{Float64,D})

    U_dense = Matrix(U)
    x_dense = Vector(x)
    y_dense = U_dense' \ x_dense

    @test maximum(abs.(y .- y_dense)) < 1e-10
end

@testitem "Logdet" begin
    using RiemannianSSMs
    using LinearAlgebra
    using StableRNGs

    D = 3
    K = 5
    rng = StableRNG(1234)

    A = rand(rng, SymPSDBlockTridiag{Float64,D}, K)
    F = cholesky(A)

    log_det = logdet(F)

    A_dense = Matrix(A)
    F_dense = cholesky(A_dense)
    log_det_dense = 2 * sum(logdet, diag(F_dense.U))

    @test abs(log_det - log_det_dense) < 1e-10
end

@testitem "Selected inverse" begin
    using RiemannianSSMs
    using LinearAlgebra
    using StableRNGs
    using StaticArrays

    D = 3
    K = 5
    rng = StableRNG(1234)

    A = rand(rng, SymPSDBlockTridiag{Float64,D}, K)
    A_inv = block_tridiag_selected_inv(A)

    @test isa(A_inv, SymPSDBlockTridiag{Float64,D,D^2})

    A_dense = Matrix(A)
    A_inv_dense = inv(cholesky(A_dense))

    # Zero out non-block-tridiagonal elements
    for k in 1:K
        if k > 2
            A_inv_dense[(D * (k - 1) + 1):(D * k), 1:(D * (k - 2))] .= 0.0
        end
        if k < K
            A_inv_dense[(D * (k - 1) + 1):(D * k), (D * (k + 1) + 1):end] .= 0.0
        end
    end

    @test maximum(abs.(A_inv .- A_inv_dense)) < 1e-10
end

# @testitem "Hamiltonian" begin
#     using RiemannianSSMs
#     using LinearAlgebra
#     using StableRNGs
#     using StaticArrays

#     D = 3
#     K = 5
#     N = D * K
#     rng = StableRNG(1234)

#     G = rand(rng, SymPSDBlockTridiag{Float64,D}, K)
#     θs = BlockVector([rand(rng, SVector{D,Float64}) for k in 1:K])
#     ps = BlockVector([rand(rng, SVector{D,Float64}) for k in 1:K])
#     ll = rand(rng)

#     H = RiemannianSSMs._calc_hamiltonian(θs, ps, ll, G)

#     G_dense = Matrix(G)
#     ps_dense = Vector(ps)
#     H_dense = (
#         -ll + 0.5 * (N * log(2π) + logdet(G_dense) + ps_dense' * inv(G_dense) * ps_dense)
#     )

#     @test abs(H - H_dense) < 1e-10
# end

# @testitem "∇θ_H" begin
#     using RiemannianSSMs
#     using LinearAlgebra
#     using StableRNGs
#     using StaticArrays

#     D = 3
#     K = 5
#     N = D * K
#     rng = StableRNG(1234)

#     G = rand(rng, SymPSDBlockTridiag{Float64,D}, K)
#     dGs = [rand(rng, SymPSDBlockTridiag{Float64,D}, K) for d in 1:D]
#     ps = BlockVector([rand(rng, SVector{D,Float64}) for k in 1:K])
#     grad = BlockVector([rand(rng, SVector{D,Float64}) for k in 1:K])

#     ∇θ = RiemannianSSMs._∇θ_H(ps, grad, G, dGs)

#     G_dense = Matrix(G)
#     G_inv_dense = inv(cholesky(G_dense))
#     dGs_dense = [zeros(N, N) for d in 1:D, k in 1:K]
#     for d in 1:D
#         for k in 1:K
#             bi = (k - 1) * D + 1
#             bj = bi + D - 1
#             dGs_dense[d, k][bi:bj, bi:bj] = dGs[d].diag_blocks[k]
#             if k < K
#                 dGs_dense[d, k][bi:bj, (bj + 1):(bj + D)] = dGs[d].super_blocks[k]
#                 dGs_dense[d, k][(bj + 1):(bj + D), bi:bj] = dGs[d].super_blocks[k]'
#             end
#         end
#     end
#     ps_dense = Vector(ps)
#     grad_dense = Vector(grad)
#     ∇θ_dense = Vector{Float64}(undef, D * K)
#     for k in 1:K
#         for d in 1:D
#             n = (k - 1) * D + d
#             v = (
#                 -grad_dense[n] + 0.5 * tr(G_inv_dense * dGs_dense[d, k]) -
#                 0.5 * ps_dense' * G_inv_dense * dGs_dense[d, k] * G_inv_dense * ps_dense
#             )
#             ∇θ_dense[n] = v
#         end
#     end

#     @test maximum(abs.(∇θ .- ∇θ_dense)) < 1e-10
# end

# @testitem "∇p_H" begin
#     using RiemannianSSMs
#     using LinearAlgebra
#     using StableRNGs
#     using StaticArrays

#     D = 3
#     K = 5
#     N = D * K
#     rng = StableRNG(1234)

#     G = rand(rng, SymPSDBlockTridiag{Float64,D}, K)
#     ps = BlockVector([rand(rng, SVector{D,Float64}) for k in 1:K])

#     ∇p = RiemannianSSMs._∇p_H(ps, G)

#     G_dense = Matrix(G)
#     ps_dense = Vector(ps)
#     ∇p_dense = G_dense \ ps_dense

#     @test maximum(abs.(∇p .- ∇p_dense)) < 1e-10
# end

@testitem "Variable Restoring Force Dynamics Gradients" begin
    using RiemannianSSMs
    using FiniteDiff
    using StableRNGs
    using StaticArrays

    rng = StableRNG(1234)
    dyn = VariableRestoringForceDynamics{Float64}(rand(rng, 6)...)
    z = @SVector rand(rng, 4)

    Jf_analytical = calc_Jf(dyn, z)
    Jf_numerical = FiniteDiff.finite_difference_jacobian(z -> f(dyn, z), z)

    @test maximum(abs.(Jf_analytical .- Jf_numerical)) < 1e-7

    Hfs_analytical = calc_Hfs(dyn, z)
    Hfs_numerical = Vector{Matrix{Float64}}(undef, 4)
    for i in 1:4
        Hfs_numerical[i] = FiniteDiff.finite_difference_jacobian(
            z -> calc_Jf(dyn, z)[:, i], z
        )
    end

    @test all(maximum(abs.(Hfs_analytical[i] .- Hfs_numerical[i])) < 1e-7 for i in 1:4)
end

@testitem "Squared Landmark Distance Gradients" begin
    using RiemannianSSMs
    using FiniteDiff
    using StableRNGs
    using StaticArrays

    rng = StableRNG(1234)
    model = TwoLandmarkMeasurementModel{Float64}(rand(rng, 6)...)
    z = @SVector rand(rng, 4)

    Jh_analytical = calc_Jh(model, z)
    Jh_numerical = FiniteDiff.finite_difference_jacobian(z -> h(model, z), z)

    @test maximum(abs.(Jh_analytical .- Jh_numerical)) < 1e-7

    Hhs_analytical = calc_Hhs(model, z)
    Hhs_numerical = Vector{Matrix{Float64}}(undef, 4)
    for i in 1:4
        Hhs_numerical[i] = FiniteDiff.finite_difference_jacobian(
            z -> calc_Jh(model, z)[:, i], z
        )
    end

    @test all(maximum(abs.(Hhs_analytical[i] .- Hhs_numerical[i])) < 1e-7 for i in 1:4)
end

#####################################
#### BORDERED BLOCK TRIDIAG TESTS ###
#####################################

@testitem "Bordered Cholesky factorisation" begin
    using RiemannianSSMs
    using LinearAlgebra
    using StableRNGs

    D = 3
    P = 2
    K = 5
    rng = StableRNG(1234)

    A = rand(rng, SymPSDBorderedBlockTridiag{Float64,D,P}, K)
    F = cholesky(A)

    @test isa(F, Cholesky{Float64,<:BorderedBlockUpperBidiag{Float64,D,P}})

    A_dense = Matrix(A)
    F_dense = cholesky(A_dense)

    @test maximum(abs.(Matrix(F.U) .- F_dense.U)) < 1e-10
end

@testitem "Bordered lower multiplication" begin
    using RiemannianSSMs
    using LinearAlgebra
    using StableRNGs
    using StaticArrays

    D = 3
    P = 2
    K = 5
    rng = StableRNG(1234)

    A = rand(rng, SymPSDBorderedBlockTridiag{Float64,D,P}, K)
    F = cholesky(A)
    U = F.factors

    x = BorderedBlockVector{Float64,D,P}(
        [rand(rng, SVector{D,Float64}) for k in 1:K], rand(rng, SVector{P,Float64})
    )
    y = U' * x

    @test isa(y, BorderedBlockVector{Float64,D,P})

    U_dense = Matrix(U)
    x_dense = Vector(x)
    y_dense = U_dense' * x_dense

    @test maximum(abs.(y .- y_dense)) < 1e-10
end

@testitem "Bordered upper backward solve" begin
    using RiemannianSSMs
    using LinearAlgebra
    using StableRNGs
    using StaticArrays

    D = 3
    P = 2
    K = 5
    rng = StableRNG(1234)

    A = rand(rng, SymPSDBorderedBlockTridiag{Float64,D,P}, K)
    F = cholesky(A)
    U = F.factors

    x = BorderedBlockVector{Float64,D,P}(
        [rand(rng, SVector{D,Float64}) for k in 1:K], rand(rng, SVector{P,Float64})
    )
    y = U \ x

    @test isa(y, BorderedBlockVector{Float64,D,P})

    U_dense = Matrix(U)
    x_dense = Vector(x)
    y_dense = U_dense \ x_dense

    @test maximum(abs.(y .- y_dense)) < 1e-10
end

@testitem "Bordered lower forward solve" begin
    using RiemannianSSMs
    using LinearAlgebra
    using StableRNGs
    using StaticArrays

    D = 3
    P = 2
    K = 5
    rng = StableRNG(1234)

    A = rand(rng, SymPSDBorderedBlockTridiag{Float64,D,P}, K)
    F = cholesky(A)
    U = F.factors

    x = BorderedBlockVector{Float64,D,P}(
        [rand(rng, SVector{D,Float64}) for k in 1:K], rand(rng, SVector{P,Float64})
    )
    y = U' \ x

    @test isa(y, BorderedBlockVector{Float64,D,P})

    U_dense = Matrix(U)
    x_dense = Vector(x)
    y_dense = U_dense' \ x_dense

    @test maximum(abs.(y .- y_dense)) < 1e-10
end

@testitem "Bordered Cholesky solve" begin
    using RiemannianSSMs
    using LinearAlgebra
    using StableRNGs
    using StaticArrays

    D = 3
    P = 2
    K = 5
    rng = StableRNG(1234)

    A = rand(rng, SymPSDBorderedBlockTridiag{Float64,D,P}, K)
    F = cholesky(A)

    x = BorderedBlockVector{Float64,D,P}(
        [rand(rng, SVector{D,Float64}) for k in 1:K], rand(rng, SVector{P,Float64})
    )
    y = F \ x

    @test isa(y, BorderedBlockVector{Float64,D,P})

    A_dense = Matrix(A)
    x_dense = Vector(x)
    y_dense = A_dense \ x_dense

    @test maximum(abs.(y .- y_dense)) < 1e-10
end

@testitem "Bordered logdet" begin
    using RiemannianSSMs
    using LinearAlgebra
    using StableRNGs

    D = 3
    P = 2
    K = 5
    rng = StableRNG(1234)

    A = rand(rng, SymPSDBorderedBlockTridiag{Float64,D,P}, K)
    F = cholesky(A)

    log_det = logdet(F)

    A_dense = Matrix(A)
    log_det_dense = logdet(A_dense)

    @test abs(log_det - log_det_dense) < 1e-10
end

@testitem "Bordered selected inverse" begin
    using RiemannianSSMs
    using LinearAlgebra
    using StableRNGs
    using StaticArrays

    D = 3
    P = 2
    K = 5
    rng = StableRNG(1234)

    A = rand(rng, SymPSDBorderedBlockTridiag{Float64,D,P}, K)
    T_inv_tridiag, T_inv_B_blocks, S_inv = bordered_block_selected_inv(A)

    A_dense = Matrix(A)
    A_inv_dense = inv(cholesky(A_dense))

    # Check corner block inverse (S^{-1})
    @test maximum(abs.(S_inv .- A_inv_dense[(K * D + 1):end, (K * D + 1):end])) < 1e-10

    # Check T^{-1}B blocks (this is part of the border inverse calculation)
    T_dense = A_dense[1:(K * D), 1:(K * D)]
    B_dense = A_dense[1:(K * D), (K * D + 1):end]
    T_inv_B_dense = T_dense \ B_dense

    for k in 1:K
        @test maximum(
            abs.(T_inv_B_blocks[k] .- T_inv_B_dense[((k - 1) * D + 1):(k * D), :])
        ) < 1e-10
    end

    # Check that the tridiagonal blocks of T_inv_tridiag match the dense inverse
    for k in 1:K
        @test maximum(
            abs.(
                T_inv_tridiag.diag_blocks[k] .-
                A_inv_dense[((k - 1) * D + 1):(k * D), ((k - 1) * D + 1):(k * D)],
            ),
        ) < 1e-9
    end
end

@testitem "BorderedBlockVector conversion" begin
    using RiemannianSSMs
    using LinearAlgebra
    using StableRNGs
    using StaticArrays

    D = 3
    P = 2
    K = 5
    rng = StableRNG(1234)

    v = rand(rng, K * D + P)
    x = to_bordered_block_vector(v, Val(D), Val(P))

    @test isa(x, BorderedBlockVector{Float64,D,P})
    @test length(x) == K * D + P

    v_back = from_bordered_block_vector(x)
    @test maximum(abs.(v .- v_back)) < 1e-15
end

#####################################
### MCRHMC MODIFIED CHOLESKY TESTS ##
#####################################

@testitem "MCRHMC: Modified Cholesky factorization" begin
    using RiemannianSSMs
    using RiemannianSSMs.MCRHMC
    using LinearAlgebra
    using SparseArrays
    using StableRNGs

    rng = StableRNG(1234)
    n = 10

    # Create a symmetric positive definite sparse matrix
    A_dense = rand(rng, n, n)
    A_dense = A_dense * A_dense' + I
    A = sparse(A_dense)

    u = fill(1e-6, n)
    F = modified_cholesky(A, u, 0)

    # Reconstruct and check
    G_reconstructed = reconstruct(F)
    G_dense = Matrix(G_reconstructed)

    # Should be close to original for SPD matrix
    @test maximum(abs.(G_dense .- A_dense)) < 1e-10

    # Test solve
    b = rand(rng, n)
    x = F \ b
    x_ref = A_dense \ b
    @test maximum(abs.(x .- x_ref)) < 1e-10

    # Test logdet
    ld = MCRHMC.logdet(F)
    ld_ref = logdet(A_dense)
    @test abs(ld - ld_ref) < 1e-10
end

@testitem "MCRHMC: Modified Cholesky with indefinite matrix" begin
    using RiemannianSSMs
    using RiemannianSSMs.MCRHMC
    using LinearAlgebra
    using SparseArrays
    using StableRNGs

    rng = StableRNG(1234)
    n = 10

    # Create indefinite symmetric matrix
    A_dense = rand(rng, n, n)
    A_dense = A_dense + A_dense'  # Symmetric but not necessarily PD
    A_dense[1, 1] = -5.0  # Make it definitely indefinite
    A = sparse(A_dense)

    u = fill(1.0, n)  # Regularization
    F = modified_cholesky(A, u, 0)

    # Reconstructed matrix should be SPD
    G_reconstructed = reconstruct(F)
    G_dense = Matrix(G_reconstructed)

    # Check symmetry
    @test maximum(abs.(G_dense .- G_dense')) < 1e-14

    # Check positive definiteness (all eigenvalues > 0)
    eigs = eigvals(Symmetric(G_dense))
    @test all(eigs .> 0)

    # Diagonal D should be at least u (due to sabs regularization)
    D_diag = get_D(F)
    @test all(D_diag.diag .>= u .- 1e-10)
end

@testitem "MCRHMC: logdet_pullback correctness" begin
    using RiemannianSSMs
    using RiemannianSSMs.MCRHMC
    using LinearAlgebra
    using SparseArrays
    using FiniteDiff
    using StableRNGs

    rng = StableRNG(1234)
    n = 8

    # Create SPD matrix
    A_dense = rand(rng, n, n)
    A_dense = A_dense * A_dense' + 0.1I
    A = sparse(A_dense)

    u = fill(1e-6, n)

    # Function to compute logdet given perturbed matrix entries
    function logdet_fn(A_vec)
        A_pert = reshape(A_vec, n, n)
        A_pert = (A_pert + A_pert') / 2  # Ensure symmetric
        A_sparse = sparse(A_pert)
        F = modified_cholesky(A_sparse, u, 0)
        return MCRHMC.logdet(F)
    end

    A_vec = vec(A_dense)
    grad_numerical = FiniteDiff.finite_difference_gradient(logdet_fn, A_vec)
    grad_numerical_mat = reshape(grad_numerical, n, n)

    # Analytical gradient via pullback
    F = modified_cholesky(A, u, 0)
    grad_analytical = logdet_pullback(F, A)
    grad_analytical_dense = Matrix(grad_analytical)

    # For symmetric matrices, compare symmetrized gradients
    # The pullback returns gradient w.r.t. full matrix, but we perturb symmetrically
    grad_numerical_sym = (grad_numerical_mat + grad_numerical_mat') / 2

    # The analytical gradient should match numerical
    # Note: off-diagonal entries should be doubled in the pullback for symmetric matrices
    diff = abs.(grad_analytical_dense .- grad_numerical_sym)
    @test maximum(diff) < 1e-6
end

@testitem "MCRHMC: quadform_pullback correctness" begin
    using RiemannianSSMs
    using RiemannianSSMs.MCRHMC
    using LinearAlgebra
    using SparseArrays
    using FiniteDiff
    using StableRNGs

    rng = StableRNG(1234)
    n = 8

    # Create SPD matrix
    A_dense = rand(rng, n, n)
    A_dense = A_dense * A_dense' + 0.1I
    A = sparse(A_dense)

    u = fill(1e-6, n)
    p = rand(rng, n)  # momentum vector

    # Function to compute quadratic form given perturbed matrix entries
    function quadform_fn(A_vec)
        A_pert = reshape(A_vec, n, n)
        A_pert = (A_pert + A_pert') / 2
        A_sparse = sparse(A_pert)
        F = modified_cholesky(A_sparse, u, 0)
        v = F \ p
        return 0.5 * dot(p, v)  # 0.5 * p' * G^{-1} * p
    end

    A_vec = vec(A_dense)
    grad_numerical = FiniteDiff.finite_difference_gradient(quadform_fn, A_vec)
    grad_numerical_mat = reshape(grad_numerical, n, n)
    grad_numerical_sym = (grad_numerical_mat + grad_numerical_mat') / 2

    # Analytical gradient via pullback
    F = modified_cholesky(A, u, 0)
    v = F \ p
    grad_analytical = quadform_pullback(F, A, v)
    grad_analytical_dense = Matrix(grad_analytical)

    diff = abs.(grad_analytical_dense .- grad_numerical_sym)
    @test maximum(diff) < 1e-5
end

@testitem "MCRHMC: hamiltonian_grad correctness" begin
    using RiemannianSSMs
    using RiemannianSSMs.MCRHMC
    using LinearAlgebra
    using SparseArrays
    using FiniteDiff
    using StableRNGs

    rng = StableRNG(1234)
    n = 8

    # Create SPD matrix
    A_dense = rand(rng, n, n)
    A_dense = A_dense * A_dense' + 0.1I
    A = sparse(A_dense)

    u = fill(1e-6, n)
    p = rand(rng, n)  # momentum vector

    # Kinetic energy: H = 0.5 * log|G| + 0.5 * p' * G^{-1} * p
    function kinetic_energy(A_vec)
        A_pert = reshape(A_vec, n, n)
        A_pert = (A_pert + A_pert') / 2
        A_sparse = sparse(A_pert)
        F = modified_cholesky(A_sparse, u, 0)
        logdetG = MCRHMC.logdet(F)
        v = F \ p
        quadform = dot(p, v)
        return 0.5 * logdetG + 0.5 * quadform
    end

    A_vec = vec(A_dense)
    grad_numerical = FiniteDiff.finite_difference_gradient(kinetic_energy, A_vec)
    grad_numerical_mat = reshape(grad_numerical, n, n)
    grad_numerical_sym = (grad_numerical_mat + grad_numerical_mat') / 2

    # Analytical gradient
    F = modified_cholesky(A, u, 0)
    v = F \ p
    grad_analytical = hamiltonian_grad(F, A, v)
    grad_analytical_dense = Matrix(grad_analytical)

    diff = abs.(grad_analytical_dense .- grad_numerical_sym)
    max_diff = maximum(diff)
    @test max_diff < 1e-5
end

@testitem "MCRHMC: sparse_trace_symmetric correctness" begin
    using RiemannianSSMs
    using RiemannianSSMs.MCRHMC
    using LinearAlgebra
    using SparseArrays
    using StableRNGs

    rng = StableRNG(1234)
    n = 6

    # Create two symmetric sparse matrices (with un-doubled values in both triangles)
    A_dense = rand(rng, n, n)
    A_dense = A_dense + A_dense'
    B_dense = rand(rng, n, n)
    B_dense = B_dense + B_dense'

    A = sparse(A_dense)
    B = sparse(B_dense)

    # For symmetric matrices stored with both triangles and un-doubled values,
    # sparse_trace_symmetric sums over all entries: tr = Σ_{i,j} A[i,j] * B[i,j]
    trace_ref = sum(A_dense .* B_dense)
    trace_computed = sparse_trace_symmetric(A, B)

    @test abs(trace_computed - trace_ref) < 1e-10
end

#####################################
### KLEPPE OBSERVED HESSIAN TESTS ###
#####################################

@testitem "Kleppe: calc_observed_hessian symmetry" begin
    using RiemannianSSMs
    using LinearAlgebra
    using SparseArrays
    using Distributions
    using StaticArrays
    using StableRNGs

    rng = StableRNG(1234)

    # Small test case
    D = 2
    P = 1
    K = 5

    δt = 0.1
    σ_u = 0.1  # Use larger noise for numerical stability
    σ_v = 0.1
    dyn = VanDerPolDynamicsParam(σ_u, σ_v, δt)

    σ_obs = 0.5
    obs = PartialLinearObservationParam(σ_obs)

    μ0 = @SVector [1.0, 0.0]
    Σ0 = Diagonal(@SVector([0.1, 0.1]))
    prior = MvNormal(μ0, Σ0)
    ssm = SSMParam(prior, dyn, obs)

    prior_prec = [0.25]
    obs_indices = [2, 4]

    # Random state vector
    z = rand(rng, K * D + P)

    G = calc_observed_hessian(z, ssm, K, D, P, prior_prec, obs_indices)
    G_dense = Matrix(G)

    # Check symmetry
    @test maximum(abs.(G_dense .- G_dense')) < 1e-12
end

@testitem "Kleppe: calc_observed_hessian vs finite differences" begin
    using RiemannianSSMs
    using LinearAlgebra
    using SparseArrays
    using Distributions
    using StaticArrays
    using FiniteDiff
    using StableRNGs

    rng = StableRNG(1234)

    D = 2
    P = 1
    K = 4

    δt = 0.1
    σ_u = 0.1
    σ_v = 0.1
    dyn = VanDerPolDynamicsParam(σ_u, σ_v, δt)

    σ_obs = 0.5
    obs = PartialLinearObservationParam(σ_obs)

    μ0 = @SVector [1.0, 0.0]
    Σ0 = Diagonal(@SVector([0.1, 0.1]))
    prior = MvNormal(μ0, Σ0)
    ssm = SSMParam(prior, dyn, obs)
    ys = [rand(rng, SVector{1,Float64}) for _ in 1:2]

    prior_prec = [0.25]
    obs_indices = [2, 4]

    # Random state vector
    z = rand(rng, K * D + P)

    # Compute analytical observed Hessian (negative Hessian of log-likelihood)
    G = calc_observed_hessian(z, ssm, K, D, P, prior_prec, obs_indices)
    G_dense = Matrix(G)

    # Compute numerical Hessian of negative log-likelihood
    function neg_ll_fn(z_vec)
        z_blocks = to_bordered_block_vector(z, Val(D), Val(P))
        # return -calc_ll(z_blocks, ssm, K, D, P, prior_prec, obs_indices)
        return -calc_ll_observed(z_vec, ys, ssm, K, D, P, [0.0], prior_prec, obs_indices)
    end

    H_numerical = FiniteDiff.finite_difference_hessian(neg_ll_fn, z)

    # Compare
    max_diff = maximum(abs.(G_dense .- H_numerical))
    @test max_diff < 1e-5
end

@testitem "Kleppe: calc_ll_grad_observed vs finite differences" begin
    using RiemannianSSMs
    using LinearAlgebra
    using SparseArrays
    using Distributions
    using StaticArrays
    using FiniteDiff
    using StableRNGs

    rng = StableRNG(1234)

    D = 2
    P = 1
    K = 4

    δt = 0.1
    σ_u = 0.1
    σ_v = 0.1
    dyn = VanDerPolDynamicsParam(σ_u, σ_v, δt)

    σ_obs = 0.5
    obs = PartialLinearObservationParam(σ_obs)

    μ0 = @SVector [1.0, 0.0]
    Σ0 = Diagonal(@SVector([0.1, 0.1]))
    prior = MvNormal(μ0, Σ0)
    ssm = SSMParam(prior, dyn, obs)

    prior_mean = [0.0]
    prior_prec = [0.25]
    obs_indices = [2, 4]

    # Random state vector
    z = rand(rng, K * D + P)

    # Generate some observations
    ys = [rand(rng, SVector{1,Float64}) for _ in 1:length(obs_indices)]

    # Compute analytical log-likelihood gradient
    grad_analytical = calc_ll_grad_observed(
        z, ys, ssm, K, D, P, prior_mean, prior_prec, obs_indices
    )

    # Compute numerical gradient
    function ll_fn(z_vec)
        return calc_ll_observed(
            z_vec, ys, ssm, K, D, P, prior_mean, prior_prec, obs_indices
        )
    end

    grad_numerical = FiniteDiff.finite_difference_gradient(ll_fn, z)

    @test maximum(abs.(grad_analytical .- grad_numerical)) < 1e-6
end

@testitem "Kleppe: calc_observed_hessian_derivs vs finite differences" begin
    using RiemannianSSMs
    using LinearAlgebra
    using SparseArrays
    using Distributions
    using StaticArrays
    using FiniteDiff
    using StableRNGs

    rng = StableRNG(1234)

    D = 2
    P = 1
    K = 3  # Small for speed

    δt = 0.1
    σ_u = 0.1
    σ_v = 0.1
    dyn = VanDerPolDynamicsParam(σ_u, σ_v, δt)

    σ_obs = 0.5
    obs = PartialLinearObservationParam(σ_obs)

    μ0 = @SVector [1.0, 0.0]
    Σ0 = Diagonal(@SVector([0.1, 0.1]))
    prior = MvNormal(μ0, Σ0)
    ssm = SSMParam(prior, dyn, obs)

    prior_prec = [0.25]
    obs_indices = [2]

    z = rand(rng, K * D + P)
    n = K * D + P

    # Compute analytical derivatives
    dGs_analytical = calc_observed_hessian_derivs(z, ssm, K, D, P, obs_indices)

    # Check against finite differences for each component
    ε = 1e-6
    max_errors = Float64[]

    dGs_numerical = Vector{SparseMatrixCSC{Float64,Int}}(undef, n)
    max_errors = Float64[]

    for i in 1:n
        # Numerical derivative via finite differences
        z_plus = copy(z)
        z_plus[i] += ε
        z_minus = copy(z)
        z_minus[i] -= ε

        G_plus = calc_observed_hessian(z_plus, ssm, K, D, P, prior_prec, obs_indices)
        G_minus = calc_observed_hessian(z_minus, ssm, K, D, P, prior_prec, obs_indices)

        dG_numerical = (G_plus - G_minus) / (2ε)
        dGs_numerical[i] = dG_numerical

        # Compare
        diff = abs.(dGs_analytical[i] .- dG_numerical)
        max_error = maximum(diff)
        push!(max_errors, max_error)
    end

    # All errors should be small
    @test maximum(max_errors) < 1e-4
end

@testitem "Kleppe: Full ∂H∂θ gradient vs finite differences" begin
    using RiemannianSSMs
    using RiemannianSSMs.MCRHMC
    using AdvancedHMC
    using LinearAlgebra
    using SparseArrays
    using Distributions
    using StaticArrays
    using FiniteDiff
    using StableRNGs
    using LogDensityProblems

    rng = StableRNG(1234)

    D = 2
    P = 1
    K = 3

    δt = 0.1
    σ_u = 0.1  # Larger noise for stability
    σ_v = 0.1
    dyn = VanDerPolDynamicsParam(σ_u, σ_v, δt)

    σ_obs = 0.5
    obs = PartialLinearObservationParam(σ_obs)

    μ0 = @SVector [1.0, 0.0]
    Σ0 = Diagonal(@SVector([0.1, 0.1]))
    prior = MvNormal(μ0, Σ0)
    ssm = SSMParam(prior, dyn, obs)

    prior_mean = @SVector [0.0]
    prior_var = @SVector [4.0]
    prior_prec_vec = Vector(1.0 ./ prior_var)
    obs_indices = [2]
    n = K * D + P

    # Generate observations
    ys = [rand(rng, SVector{1,Float64}) for _ in 1:length(obs_indices)]

    # Regularization
    u = fill(0.5, n)

    # Create metric and Hamiltonian
    metric = ObservedHessianMetric(ssm, ys, D, P, K, prior_mean, prior_var, obs_indices, u)

    struct LogTargetDensityParamSparse{D,P,M,V,PM,PV,OI}
        dim::Int
        ssm::M
        ys::V
        prior_mean::PM
        prior_prec::PV
        obs_indices::OI
    end

    function LogTargetDensityParamSparse(
        K::Int, D::Int, P::Int, ssm, ys, prior_mean, prior_var, obs_indices
    )
        dim = K * D + P
        prior_prec = SVector{P,Float64}(1 ./ prior_var)
        return LogTargetDensityParamSparse{
            D,
            P,
            typeof(ssm),
            typeof(ys),
            typeof(prior_mean),
            typeof(prior_prec),
            typeof(obs_indices),
        }(
            dim, ssm, ys, prior_mean, prior_prec, obs_indices
        )
    end

    function LogDensityProblems.logdensity(
        p::LogTargetDensityParamSparse{D,P}, θ
    ) where {D,P}
        θ_blocks = to_bordered_block_vector(θ, Val(D), Val(P))
        return calc_ll_param(
            θ_blocks, p.ys, p.ssm, p.prior_mean, p.prior_prec; obs_indices=p.obs_indices
        )
    end

    function LogDensityProblems.logdensity_and_gradient(
        p::LogTargetDensityParamSparse{D,P}, θ
    ) where {D,P}
        θ_blocks = to_bordered_block_vector(θ, Val(D), Val(P))
        ll = calc_ll_param(
            θ_blocks, p.ys, p.ssm, p.prior_mean, p.prior_prec; obs_indices=p.obs_indices
        )
        grad = calc_ll_grad_param(
            θ_blocks, p.ys, p.ssm, p.prior_mean, p.prior_prec; obs_indices=p.obs_indices
        )
        return ll, from_bordered_block_vector(grad)
    end

    LogDensityProblems.dimension(p::LogTargetDensityParamSparse) = p.dim

    function LogDensityProblems.capabilities(::Type{<:LogTargetDensityParamSparse})
        return LogDensityProblems.LogDensityOrder{1}()
    end

    ℓπ = LogTargetDensityParamSparse(K, D, P, ssm, ys, prior_mean, prior_var, obs_indices)

    h = Hamiltonian(metric, ℓπ)

    # Test point
    z = rand(rng, n)
    r = rand(rng, n)  # momentum

    # Compute analytical gradient
    result = AdvancedHMC.∂H∂θ(h, z, r)
    grad_analytical = result.gradient

    # Compute numerical gradient of full Hamiltonian
    function hamiltonian_fn(z_vec)
        # Potential energy: -log π(z)
        U =
            -calc_ll_observed(
                z_vec, ys, ssm, K, D, P, collect(prior_mean), prior_prec_vec, obs_indices
            )

        # Kinetic energy: 0.5 * log|G| + 0.5 * r' G^{-1} r
        G = calc_observed_hessian(z_vec, ssm, K, D, P, prior_prec_vec, obs_indices)
        F = MCRHMC.modified_cholesky(G, u, 0)
        logdetG = MCRHMC.logdet(F)
        v = F \ r
        quadform = dot(r, v)

        K_energy = 0.5 * logdetG + 0.5 * quadform

        return U + K_energy
    end

    grad_numerical = FiniteDiff.finite_difference_gradient(hamiltonian_fn, z)

    # Compare
    diff = abs.(grad_analytical .- grad_numerical)
    max_diff = maximum(diff)

    @test max_diff < 1e-3
end

@testitem "Kleppe: Compare kinetic gradient components" begin
    using RiemannianSSMs
    using RiemannianSSMs.MCRHMC
    using LinearAlgebra
    using SparseArrays
    using Distributions
    using StaticArrays
    using FiniteDiff
    using StableRNGs

    rng = StableRNG(1234)

    D = 2
    P = 1
    K = 3

    δt = 0.1
    σ_u = 0.1
    σ_v = 0.1
    dyn = VanDerPolDynamicsParam(σ_u, σ_v, δt)

    σ_obs = 0.5
    obs = PartialLinearObservationParam(σ_obs)

    μ0 = @SVector [1.0, 0.0]
    Σ0 = Diagonal(@SVector([0.1, 0.1]))
    prior = MvNormal(μ0, Σ0)
    ssm = SSMParam(prior, dyn, obs)

    prior_prec = [0.25]
    obs_indices = [2]
    n = K * D + P

    z = rand(rng, n)
    r = rand(rng, n)
    u = fill(0.5, n)

    # Compute G and its factorization
    G = calc_observed_hessian(z, ssm, K, D, P, prior_prec, obs_indices)
    F = MCRHMC.modified_cholesky(G, u, 0)
    v = F \ r

    # Compute H_grad
    H_grad = MCRHMC.hamiltonian_grad(F, G, v)
    H_grad_dense = Matrix(H_grad)

    # Compute dGs
    dGs = calc_observed_hessian_derivs(z, ssm, K, D, P, obs_indices)

    # Compute kinetic gradient using sparse_trace_symmetric
    kinetic_grad_sparse = zeros(n)
    for i in 1:n
        kinetic_grad_sparse[i] = MCRHMC.sparse_trace_symmetric(H_grad, dGs[i])
    end

    # Compute kinetic gradient using dense operations for reference
    function kinetic_energy(z_vec)
        G_pert = calc_observed_hessian(z_vec, ssm, K, D, P, prior_prec, obs_indices)
        F_pert = MCRHMC.modified_cholesky(G_pert, u, 0)
        logdetG = MCRHMC.logdet(F_pert)
        v_pert = F_pert \ r
        quadform = dot(r, v_pert)
        return 0.5 * logdetG + 0.5 * quadform
    end

    kinetic_grad_numerical = FiniteDiff.finite_difference_gradient(kinetic_energy, z)

    # Compare
    diff = abs.(kinetic_grad_sparse .- kinetic_grad_numerical)
    max_diff = maximum(diff)

    @test max_diff < 1e-3
end
