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
