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

    @test isa(F, Cholesky{Float64,BlockUpperBidiag{Float64,D,D^2}})

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

@testitem "Hamiltonian" begin
    using RiemannianSSMs
    using LinearAlgebra
    using StableRNGs
    using StaticArrays

    D = 3
    K = 5
    N = D * K
    rng = StableRNG(1234)

    G = rand(rng, SymPSDBlockTridiag{Float64,D}, K)
    θs = BlockVector([rand(rng, SVector{D,Float64}) for k in 1:K])
    ps = BlockVector([rand(rng, SVector{D,Float64}) for k in 1:K])
    ll = rand(rng)

    H = RiemannianSSMs._calc_hamiltonian(θs, ps, ll, G)

    G_dense = Matrix(G)
    ps_dense = Vector(ps)
    H_dense = (
        -ll + 0.5 * (N * log(2π) + logdet(G_dense) + ps_dense' * inv(G_dense) * ps_dense)
    )

    @test abs(H - H_dense) < 1e-10
end

@testitem "∇θ_H" begin
    using RiemannianSSMs
    using LinearAlgebra
    using StableRNGs
    using StaticArrays

    D = 3
    K = 5
    N = D * K
    rng = StableRNG(1234)

    G = rand(rng, SymPSDBlockTridiag{Float64,D}, K)
    dGs = [rand(rng, SymPSDBlockTridiag{Float64,D}, K) for d in 1:D]
    ps = BlockVector([rand(rng, SVector{D,Float64}) for k in 1:K])
    grad = BlockVector([rand(rng, SVector{D,Float64}) for k in 1:K])

    ∇θ = RiemannianSSMs._∇θ_H(ps, grad, G, dGs)

    G_dense = Matrix(G)
    G_inv_dense = inv(cholesky(G_dense))
    dGs_dense = [zeros(N, N) for d in 1:D, k in 1:K]
    for d in 1:D
        for k in 1:K
            bi = (k - 1) * D + 1
            bj = bi + D - 1
            dGs_dense[d, k][bi:bj, bi:bj] = dGs[d].diag_blocks[k]
            if k < K
                dGs_dense[d, k][bi:bj, (bj + 1):(bj + D)] = dGs[d].super_blocks[k]
                dGs_dense[d, k][(bj + 1):(bj + D), bi:bj] = dGs[d].super_blocks[k]'
            end
        end
    end
    ps_dense = Vector(ps)
    grad_dense = Vector(grad)
    ∇θ_dense = Vector{Float64}(undef, D * K)
    for k in 1:K
        for d in 1:D
            n = (k - 1) * D + d
            v = (
                -grad_dense[n] + 0.5 * tr(G_inv_dense * dGs_dense[d, k]) -
                0.5 * ps_dense' * G_inv_dense * dGs_dense[d, k] * G_inv_dense * ps_dense
            )
            ∇θ_dense[n] = v
        end
    end

    @test maximum(abs.(∇θ .- ∇θ_dense)) < 1e-10
end

@testitem "∇p_H" begin
    using RiemannianSSMs
    using LinearAlgebra
    using StableRNGs
    using StaticArrays

    D = 3
    K = 5
    N = D * K
    rng = StableRNG(1234)

    G = rand(rng, SymPSDBlockTridiag{Float64,D}, K)
    ps = BlockVector([rand(rng, SVector{D,Float64}) for k in 1:K])

    ∇p = RiemannianSSMs._∇p_H(ps, G)

    G_dense = Matrix(G)
    ps_dense = Vector(ps)
    ∇p_dense = G_dense \ ps_dense

    @test maximum(abs.(∇p .- ∇p_dense)) < 1e-10
end
