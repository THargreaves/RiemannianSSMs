"""
Debug script for Kleppe's observed Hessian RHMC divergences.

This script diagnoses numerical issues by:
1. Checking the observed Hessian condition number at various points
2. Testing the modified Cholesky factorization stability
3. Tracing a single leapfrog trajectory to find where instability occurs
"""

using Distributions
using LinearAlgebra
using Random
using SparseArrays
using StaticArrays
using Printf

using AdvancedHMC
using LogDensityProblems

using RiemannianSSMs
using RiemannianSSMs.MCRHMC

# ============================================================================
# Setup (same as test script)
# ============================================================================

SEED = 42
rng = MersenneTwister(SEED)

D = 2
P = 1
δt = 0.1
K_y = 10
N_obs = 5
K = N_obs * K_y
obs_indices = collect(K_y:K_y:K)
n = K * D + P

μ0 = @SVector [1.0, 0.0]
Σ0 = Diagonal(@SVector([0.1, 0.1]))
prior = MvNormal(μ0, Σ0)

μ_true = 1.0
θ_true = @SVector [log(μ_true)]

σ_u = 0.1
σ_v = 0.1
dyn = VanDerPolDynamicsParam(σ_u, σ_v, δt)

σ_obs = 0.05
obs = PartialLinearObservationParam(σ_obs)

ssm = SSMParam(prior, dyn, obs)

prior_mean = @SVector [log(μ_true)]
prior_var = @SVector [4.0]
prior_prec = Vector(1.0 ./ prior_var)

function simulate(rng::AbstractRNG, ssm, θ, K::Int, obs_indices)
    zs = Vector{SVector{2,Float64}}(undef, K)
    ys = Vector{SVector{1,Float64}}(undef, length(obs_indices))
    obs_idx = 1
    for k in 1:K
        if k == 1
            z = SVector{2,Float64}(rand(rng, ssm.prior))
        else
            z =
                f_param(ssm.dyn, zs[k - 1], θ) +
                rand(rng, MvNormal(zeros(2), calc_Q_param(ssm.dyn)))
        end
        zs[k] = z
        if k in obs_indices
            y =
                h_param(ssm.sensor, z, θ) +
                rand(rng, MvNormal(zeros(1), calc_R_param(ssm.sensor)))
            ys[obs_idx] = y
            obs_idx += 1
        end
    end
    return zs, ys
end

zs_true, ys = simulate(rng, ssm, θ_true, K, obs_indices)
zs_true_block = BlockVector{Float64,2}(zs_true)
initial_θ = vcat(from_block_vector(zs_true_block), collect(θ_true))

println("="^70)
println("DIAGNOSTICS FOR KLEPPE'S OBSERVED HESSIAN MCRHMC")
println("="^70)
println("\nProblem size: n = $n (K=$K states × D=$D + P=$P parameters)")

# ============================================================================
# 1. Check observed Hessian at true point
# ============================================================================

println("\n" * "-"^70)
println("1. OBSERVED HESSIAN AT TRUE POINT")
println("-"^70)

G_true = calc_observed_hessian(initial_θ, ssm, K, Val(D), Val(P), prior_prec, obs_indices)
G_true_dense = Matrix(G_true)

println("\nMatrix properties:")
println("  Size: $(size(G_true))")
println("  nnz: $(nnz(G_true))")
println("  Symmetric: $(issymmetric(G_true_dense))")

eigs_true = eigvals(Symmetric(G_true_dense))
println("\nEigenvalue spectrum:")
println("  Min eigenvalue: $(minimum(eigs_true))")
println("  Max eigenvalue: $(maximum(eigs_true))")
println("  Condition number: $(maximum(eigs_true) / max(minimum(eigs_true), 1e-15))")
println("  Number of negative eigenvalues: $(sum(eigs_true .< 0))")
println("  Number of eigenvalues < 1e-6: $(sum(eigs_true .< 1e-6))")

# ============================================================================
# 2. Test modified Cholesky at true point
# ============================================================================

println("\n" * "-"^70)
println("2. MODIFIED CHOLESKY AT TRUE POINT")
println("-"^70)

u_values = [1e-6, 1e-4, 1e-2, 1e-1, 1.0]

for u_val in u_values
    u = fill(u_val, n)
    F = modified_cholesky(G_true, u, 0)

    D_diag = diag(get_D(F))
    G_reconstructed = reconstruct(F)
    recon_err = maximum(abs.(Matrix(G_true) - Matrix(G_reconstructed)))

    println("\nu = $u_val:")
    println("  Min D[i]: $(minimum(D_diag))")
    println("  Max D[i]: $(maximum(D_diag))")
    println("  Reconstruction error: $recon_err")

    # Check solve stability
    b = randn(n)
    x = F \ b
    residual = norm(Matrix(G_reconstructed) * x - b) / norm(b)
    println("  Solve residual: $residual")
end

# ============================================================================
# 3. Check Hessian along a random perturbation direction
# ============================================================================

println("\n" * "-"^70)
println("3. HESSIAN STABILITY ALONG PERTURBATION")
println("-"^70)

perturbation = randn(n)
perturbation = perturbation / norm(perturbation)

scales = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]

println("\nPerturbing from true point in random direction:")
println(
    @sprintf(
        "%-10s %15s %15s %15s %15s", "Scale", "Min eig", "Max eig", "Cond #", "Neg eigs"
    )
)

for scale in scales
    θ_pert = initial_θ + scale * perturbation
    G_pert = calc_observed_hessian(θ_pert, ssm, K, Val(D), Val(P), prior_prec, obs_indices)
    eigs = eigvals(Symmetric(Matrix(G_pert)))

    min_eig = minimum(eigs)
    max_eig = maximum(eigs)
    cond = max_eig / max(min_eig, 1e-15)
    neg_count = sum(eigs .< 0)

    println(
        @sprintf(
            "%-10.2f %15.2e %15.2e %15.2e %15d", scale, min_eig, max_eig, cond, neg_count
        )
    )
end

# ============================================================================
# 4. Check along trajectory from initial state
# ============================================================================

println("\n" * "-"^70)
println("4. HESSIAN ALONG SIMULATED TRAJECTORY")
println("-"^70)

# Simulate a longer trajectory with more noise to see what happens
function check_trajectory_stability(θ_start, n_steps, step_size, u_val)
    θ = copy(θ_start)

    println("\nStep size: $step_size, u = $u_val")
    println(
        @sprintf(
            "%-6s %12s %12s %12s %10s %12s",
            "Step",
            "Min eig",
            "Max eig",
            "Cond #",
            "Neg eigs",
            "Solve resid"
        )
    )

    u = fill(u_val, n)

    for step in 0:n_steps
        G = calc_observed_hessian(θ, ssm, K, Val(D), Val(P), prior_prec, obs_indices)
        eigs = eigvals(Symmetric(Matrix(G)))

        min_eig = minimum(eigs)
        max_eig = maximum(eigs)
        cond = max_eig / max(min_eig, 1e-15)
        neg_count = sum(eigs .< 0)

        # Test solve
        F = modified_cholesky(G, u, 0)
        b = randn(n)
        x = F \ b
        G_rec = reconstruct(F)
        residual = norm(Matrix(G_rec) * x - b) / norm(b)

        println(
            @sprintf(
                "%-6d %12.2e %12.2e %12.2e %10d %12.2e",
                step,
                min_eig,
                max_eig,
                cond,
                neg_count,
                residual
            )
        )

        if step < n_steps
            # Random step in momentum-like direction
            θ = θ + step_size * randn(n)
        end
    end
end

check_trajectory_stability(initial_θ, 5, 0.1, 1e-1)

# ============================================================================
# 5. Check gradient computation stability
# ============================================================================

println("\n" * "-"^70)
println("5. GRADIENT COMPUTATION STABILITY")
println("-"^70)

u = fill(1e-1, n)

for _ in 1:3
    θ_test = initial_θ + 0.5 * randn(n)
    r = randn(n)  # Random momentum

    G = calc_observed_hessian(θ_test, ssm, K, Val(D), Val(P), prior_prec, obs_indices)
    F = modified_cholesky(G, u, 0)

    # Check if factorization is valid
    D_diag = diag(get_D(F))
    if any(D_diag .<= 0)
        println("WARNING: Non-positive diagonal in D!")
        continue
    end

    # Compute v = G^{-1} r
    v = F \ r

    # Check solution quality
    G_rec = reconstruct(F)
    residual = norm(Matrix(G_rec) * v - r) / norm(r)

    # Compute Hamiltonian gradient
    H_grad = MCRHMC.hamiltonian_grad(F, G, v)
    H_grad_dense = Matrix(H_grad)

    # Check for NaN/Inf
    has_nan = any(isnan.(H_grad_dense))
    has_inf = any(isinf.(H_grad_dense))
    max_grad = maximum(abs.(H_grad_dense))

    # Compute dG terms
    dGs = calc_observed_hessian_derivs(θ_test, ssm, K, D, P, obs_indices)

    kinetic_grad = zeros(n)
    for i in 1:n
        kinetic_grad[i] = MCRHMC.sparse_trace_symmetric(H_grad, dGs[i])
    end

    has_nan_kg = any(isnan.(kinetic_grad))
    has_inf_kg = any(isinf.(kinetic_grad))
    max_kg = maximum(abs.(kinetic_grad))

    println("\nTest point:")
    println("  Solve residual: $residual")
    println("  H_grad: has_nan=$has_nan, has_inf=$has_inf, max=$(max_grad)")
    println("  Kinetic grad: has_nan=$has_nan_kg, has_inf=$has_inf_kg, max=$(max_kg)")
end

# ============================================================================
# 6. Single leapfrog step analysis
# ============================================================================

println("\n" * "-"^70)
println("6. SINGLE LEAPFROG STEP ANALYSIS")
println("-"^70)

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
        D,P,typeof(ssm),typeof(ys),typeof(prior_mean),typeof(prior_prec),typeof(obs_indices)
    }(
        dim, ssm, ys, prior_mean, prior_prec, obs_indices
    )
end

function LogDensityProblems.logdensity(p::LogTargetDensityParamSparse{D,P}, θ) where {D,P}
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

# Set up metric and Hamiltonian
u = fill(1e-1, n)
metric = ObservedHessianMetric(
    ssm, ys, Val(D), Val(P), K, prior_mean, prior_var, obs_indices, u
)
h = Hamiltonian(metric, ℓπ)

# Create phase point at initial position with random momentum
θ0 = initial_θ
r0 = randn(n)

println("\nInitial state:")
ll0, grad0 = LogDensityProblems.logdensity_and_gradient(ℓπ, θ0)
println("  log density: $ll0")
println("  gradient norm: $(norm(grad0))")
println("  max |grad|: $(maximum(abs.(grad0)))")

# Try to compute initial phase point
try
    z0 = AdvancedHMC.phasepoint(rng, θ0, h)
    println("  Initial Hamiltonian: $(AdvancedHMC.energy(z0))")

    # Try single leapfrog steps with different step sizes
    step_sizes = [0.001, 0.005, 0.01, 0.05]

    for ϵ in step_sizes
        println("\nStep size ϵ = $ϵ:")
        integrator = AdvancedHMC.Leapfrog(ϵ)

        try
            z1 = AdvancedHMC.step(rng, h, z0, 1, integrator)
            H0 = AdvancedHMC.energy(z0)
            H1 = AdvancedHMC.energy(z1)
            ΔH = H1 - H0

            println("  H0 = $H0")
            println("  H1 = $H1")
            println("  ΔH = $ΔH")
            println("  |θ1 - θ0| = $(norm(z1.θ - z0.θ))")
        catch e
            println("  ERROR: $e")
        end
    end
catch e
    println("  ERROR creating initial phase point: $e")
end

println("\n" * "="^70)
println("DIAGNOSTICS COMPLETE")
println("="^70)
