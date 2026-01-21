"""
Block dimension scaling experiment for LGCP AR(p) Linear model.

This experiment varies the AR order P from 1 to 10 while keeping the total
state dimension roughly constant (~600). This tests how the different samplers
scale with increasing block dimension in the block-tridiagonal structure.

For each P, we adjust N_obs so that K * P ≈ 600 (where K = N_obs * K_y).
"""

using Distributions
using LinearAlgebra
using Printf

BLAS.set_num_threads(1)
using Random
using Statistics
using Plots
using StaticArrays

using AbstractMCMC
using AdvancedHMC
using LogDensityProblems
using MCMCDiagnosticTools

using RiemannianSSMs

# ============================================================================
# Configuration
# ============================================================================

SEED = 42

# Time discretization
K_y = 5            # Latent steps between observations

# Target total state dimension
TARGET_STATE_DIM = 600

# P values to test (AR order = block dimension)
P_values = collect(1:10)

# Sampling configuration
N_samples = 2000
N_adapt = 1000

# Process noise
σ = 0.15

# ============================================================================
# Helper functions
# ============================================================================

"""
Generate decaying AR coefficients that sum to `total_sum`.
Uses exponential decay: φᵢ ∝ exp(-λ(i-1)) normalized to sum to total_sum.
"""
function generate_ar_coeffs(P::Int; total_sum=0.95, decay_rate=0.5)
    if P == 1
        return SVector{1,Float64}(total_sum)
    end
    weights = [exp(-decay_rate * (i - 1)) for i in 1:P]
    normalized = weights ./ sum(weights) .* total_sum
    return SVector{P,Float64}(normalized...)
end

"""
Compute N_obs to achieve target total state dimension.
Total state dim = K * P = N_obs * K_y * P
"""
function compute_n_obs(P::Int, target_dim::Int, K_y::Int)
    return max(5, round(Int, target_dim / (P * K_y)))
end

function compute_ess_stats(samples_array)
    ess_vals = ess(samples_array)
    return (
        min=minimum(ess_vals),
        median=median(ess_vals),
        mean=mean(ess_vals),
        ess_vec=vec(ess_vals),
    )
end

function samples_to_array(samples, total_dim)
    n = length(samples)
    arr = Array{Float64}(undef, n, 1, total_dim)
    for i in 1:n
        for j in 1:total_dim
            arr[i, 1, j] = samples[i][j]
        end
    end
    return arr
end

# ============================================================================
# Sampler functions
# ============================================================================

function run_rhmc(model, ssm_model, ys, K, prior_mean, prior_var, obs_indices, initial_θ)
    metric = RiemannianMetric(
        ssm_model, ys, K, prior_mean, prior_var; obs_indices=obs_indices
    )
    initial_ϵ = 0.01
    integrator = AdaptiveGeneralizedLeapfrog(initial_ϵ; max_iters=7)
    kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
    adaptor = StepSizeAdaptor(0.9, integrator)
    rhmc = HMCSampler(kernel, metric, adaptor)

    time_start = time()
    chains = AbstractMCMC.sample(
        model,
        rhmc,
        N_samples;
        n_adapts=N_adapt,
        initial_params=initial_θ,
        verbose=false,
        progress=true,
    )
    elapsed = time() - time_start

    samples = [s.z.θ for s in chains]
    return samples, elapsed
end

function run_hmc_dense(model, initial_θ)
    hmc = NUTS(0.8; metric=:dense)

    time_start = time()
    chains = AbstractMCMC.sample(
        model,
        hmc,
        N_samples;
        n_adapts=N_adapt,
        initial_params=initial_θ,
        verbose=false,
        progress=true,
    )
    elapsed = time() - time_start

    samples = [s.z.θ for s in chains]
    return samples, elapsed
end

function run_hmc_diagonal(model, initial_θ)
    metric = DiagEuclideanMetric(length(initial_θ))
    initial_ϵ = 0.001
    integrator = Leapfrog(initial_ϵ)
    kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
    adaptor = StanHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(0.8, integrator))
    hmc = HMCSampler(kernel, metric, adaptor)

    time_start = time()
    chains = AbstractMCMC.sample(
        model,
        hmc,
        N_samples;
        n_adapts=N_adapt,
        initial_params=initial_θ,
        verbose=false,
        progress=true,
    )
    elapsed = time() - time_start

    samples = [s.z.θ for s in chains]
    return samples, elapsed
end

function run_hmc_empirical(model, initial_θ, rhmc_samples)
    samples_mat = reduce(hcat, rhmc_samples)'
    emp_cov = cov(samples_mat)
    emp_cov_sym = Symmetric(emp_cov)

    metric = DenseEuclideanMetric(emp_cov_sym)
    initial_ϵ = 0.001
    integrator = Leapfrog(initial_ϵ)
    kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
    adaptor = StepSizeAdaptor(0.8, integrator)
    hmc = HMCSampler(kernel, metric, adaptor)

    time_start = time()
    chains = AbstractMCMC.sample(
        model,
        hmc,
        N_samples;
        n_adapts=N_adapt,
        initial_params=initial_θ,
        verbose=false,
        progress=true,
    )
    elapsed = time() - time_start

    samples = [s.z.θ for s in chains]
    return samples, elapsed
end

# ============================================================================
# Run benchmarks
# ============================================================================

results = Dict{Int,Dict{Symbol,NamedTuple}}()

for P in P_values
    println("\n" * "="^60)
    println("Running benchmark for P = $P (block dimension)")
    println("="^60)

    rng = MersenneTwister(SEED)

    # Compute dimensions
    N_obs = compute_n_obs(P, TARGET_STATE_DIM, K_y)
    K = N_obs * K_y
    obs_indices = collect(K_y:K_y:K)
    Dx = P  # State dimension equals AR order
    Dp = P + 2  # Parameter dimension
    total_dim = Dx * K + Dp
    actual_state_dim = Dx * K

    println("  N_obs = $N_obs, K = $K")
    println("  State dim (Dx) = $Dx, Total state dim = $actual_state_dim")
    println("  Total inference dim = $total_dim")

    # Generate AR coefficients
    φ_true = generate_ar_coeffs(P)
    μ_true = 0.5
    b_true = 1.0
    θ_true = SVector{P + 2,Float64}(φ_true..., μ_true, b_true)

    println(
        "  AR coeffs: φ = $(round.(φ_true, digits=3)) (sum = $(round(sum(φ_true), digits=3)))",
    )

    # Create model
    ssm_model = LGCPARpLinearModel{P}(; σ=σ, s0=1.0)

    # Simulate data
    zs_true, ys = simulate(rng, ssm_model, θ_true, K; obs_indices=obs_indices)
    zs_true_block = BlockVector{Float64,Dx}(zs_true)

    # Parameter prior
    prior_mean = SVector{P + 2,Float64}(ntuple(i -> i <= P ? 0.3 : 0.0, Val(P + 2)))
    prior_var = SVector{P + 2,Float64}(ntuple(_ -> 4.0, Val(P + 2)))

    # Setup log density
    ℓπ = RHMCLogDensity(
        ssm_model,
        ys,
        K;
        prior_mean=prior_mean,
        prior_var=prior_var,
        obs_indices=obs_indices,
    )
    adv_model = AdvancedHMC.LogDensityModel(ℓπ)

    # Initial state
    initial_θ = vcat(from_block_vector(zs_true_block), collect(θ_true))

    results[P] = Dict{Symbol,NamedTuple}()

    # Run RHMC
    println("  Running RHMC...")
    rhmc_samples, rhmc_time = run_rhmc(
        adv_model, ssm_model, ys, K, prior_mean, prior_var, obs_indices, initial_θ
    )
    rhmc_arr = samples_to_array(rhmc_samples, total_dim)
    rhmc_ess = compute_ess_stats(rhmc_arr)
    results[P][:rhmc] = (
        time=rhmc_time,
        ess_min=rhmc_ess.min,
        ess_median=rhmc_ess.median,
        ess_mean=rhmc_ess.mean,
        ess_vec=rhmc_ess.ess_vec,
    )
    println(
        "    Time: $(round(rhmc_time, digits=2))s, ESS min/med/mean: $(round(rhmc_ess.min, digits=1))/$(round(rhmc_ess.median, digits=1))/$(round(rhmc_ess.mean, digits=1))",
    )

    # Run HMC Dense
    println("  Running HMC (Dense)...")
    hmc_dense_samples, hmc_dense_time = run_hmc_dense(adv_model, initial_θ)
    hmc_dense_arr = samples_to_array(hmc_dense_samples, total_dim)
    hmc_dense_ess = compute_ess_stats(hmc_dense_arr)
    results[P][:hmc_dense] = (
        time=hmc_dense_time,
        ess_min=hmc_dense_ess.min,
        ess_median=hmc_dense_ess.median,
        ess_mean=hmc_dense_ess.mean,
        ess_vec=hmc_dense_ess.ess_vec,
    )
    println(
        "    Time: $(round(hmc_dense_time, digits=2))s, ESS min/med/mean: $(round(hmc_dense_ess.min, digits=1))/$(round(hmc_dense_ess.median, digits=1))/$(round(hmc_dense_ess.mean, digits=1))",
    )

    # Run HMC Diagonal
    println("  Running HMC (Diagonal)...")
    hmc_diag_samples, hmc_diag_time = run_hmc_diagonal(adv_model, initial_θ)
    hmc_diag_arr = samples_to_array(hmc_diag_samples, total_dim)
    hmc_diag_ess = compute_ess_stats(hmc_diag_arr)
    results[P][:hmc_diag] = (
        time=hmc_diag_time,
        ess_min=hmc_diag_ess.min,
        ess_median=hmc_diag_ess.median,
        ess_mean=hmc_diag_ess.mean,
        ess_vec=hmc_diag_ess.ess_vec,
    )
    println(
        "    Time: $(round(hmc_diag_time, digits=2))s, ESS min/med/mean: $(round(hmc_diag_ess.min, digits=1))/$(round(hmc_diag_ess.median, digits=1))/$(round(hmc_diag_ess.mean, digits=1))",
    )

    # Run HMC with empirical covariance from RHMC
    println("  Running HMC (Empirical)...")
    hmc_emp_samples, hmc_emp_time = run_hmc_empirical(adv_model, initial_θ, rhmc_samples)
    hmc_emp_arr = samples_to_array(hmc_emp_samples, total_dim)
    hmc_emp_ess = compute_ess_stats(hmc_emp_arr)
    results[P][:hmc_empirical] = (
        time=hmc_emp_time,
        ess_min=hmc_emp_ess.min,
        ess_median=hmc_emp_ess.median,
        ess_mean=hmc_emp_ess.mean,
        ess_vec=hmc_emp_ess.ess_vec,
    )
    println(
        "    Time: $(round(hmc_emp_time, digits=2))s, ESS min/med/mean: $(round(hmc_emp_ess.min, digits=1))/$(round(hmc_emp_ess.median, digits=1))/$(round(hmc_emp_ess.mean, digits=1))",
    )
end

# ============================================================================
# Compute ESS per second
# ============================================================================

ess_per_sec = Dict{Int,Dict{Symbol,NamedTuple}}()

for P in P_values
    ess_per_sec[P] = Dict{Symbol,NamedTuple}()
    for method in [:rhmc, :hmc_dense, :hmc_diag, :hmc_empirical]
        r = results[P][method]
        ess_per_sec[P][method] = (
            min=(r.ess_min / r.time),
            median=(r.ess_median / r.time),
            mean=(r.ess_mean / r.time),
        )
    end
end

# ============================================================================
# Print summary tables
# ============================================================================

println("\n" * "="^100)
println("SUMMARY: Raw ESS (min / median / mean)")
println("="^100)
println(
    "   P   |        RHMC         |        Dense        |        Diag         |      Empirical",
)
println("-"^100)
for P in P_values
    rhmc = results[P][:rhmc]
    dense = results[P][:hmc_dense]
    diag = results[P][:hmc_diag]
    emp = results[P][:hmc_empirical]
    @printf(
        "%5d  | %5.0f / %5.0f / %5.0f | %5.0f / %5.0f / %5.0f | %5.0f / %5.0f / %5.0f | %5.0f / %5.0f / %5.0f\n",
        P,
        rhmc.ess_min,
        rhmc.ess_median,
        rhmc.ess_mean,
        dense.ess_min,
        dense.ess_median,
        dense.ess_mean,
        diag.ess_min,
        diag.ess_median,
        diag.ess_mean,
        emp.ess_min,
        emp.ess_median,
        emp.ess_mean
    )
end

println("\n" * "="^100)
println("SUMMARY: ESS per second (min / median / mean)")
println("="^100)
println(
    "   P   |        RHMC         |        Dense        |        Diag         |      Empirical",
)
println("-"^100)
for P in P_values
    rhmc = ess_per_sec[P][:rhmc]
    dense = ess_per_sec[P][:hmc_dense]
    diag = ess_per_sec[P][:hmc_diag]
    emp = ess_per_sec[P][:hmc_empirical]
    @printf(
        "%5d  | %5.1f / %5.1f / %5.1f | %5.1f / %5.1f / %5.1f | %5.1f / %5.1f / %5.1f | %5.1f / %5.1f / %5.1f\n",
        P,
        rhmc.min,
        rhmc.median,
        rhmc.mean,
        dense.min,
        dense.median,
        dense.mean,
        diag.min,
        diag.median,
        diag.mean,
        emp.min,
        emp.median,
        emp.mean
    )
end

println("\n" * "="^60)
println("SUMMARY: Wall time (seconds)")
println("="^60)
println("   P   |   RHMC   |  Dense   |   Diag   | Empirical")
println("-"^60)
for P in P_values
    rhmc = results[P][:rhmc]
    dense = results[P][:hmc_dense]
    diag = results[P][:hmc_diag]
    emp = results[P][:hmc_empirical]
    @printf(
        "%5d  | %8.2f | %8.2f | %8.2f | %8.2f\n",
        P,
        rhmc.time,
        dense.time,
        diag.time,
        emp.time
    )
end

# ============================================================================
# Plotting
# ============================================================================

# Extract data for plotting
rhmc_ess_min = [results[p][:rhmc].ess_min for p in P_values]
rhmc_ess_med = [results[p][:rhmc].ess_median for p in P_values]
rhmc_ess_mean = [results[p][:rhmc].ess_mean for p in P_values]

dense_ess_min = [results[p][:hmc_dense].ess_min for p in P_values]
dense_ess_med = [results[p][:hmc_dense].ess_median for p in P_values]
dense_ess_mean = [results[p][:hmc_dense].ess_mean for p in P_values]

diag_ess_min = [results[p][:hmc_diag].ess_min for p in P_values]
diag_ess_med = [results[p][:hmc_diag].ess_median for p in P_values]
diag_ess_mean = [results[p][:hmc_diag].ess_mean for p in P_values]

emp_ess_min = [results[p][:hmc_empirical].ess_min for p in P_values]
emp_ess_med = [results[p][:hmc_empirical].ess_median for p in P_values]
emp_ess_mean = [results[p][:hmc_empirical].ess_mean for p in P_values]

rhmc_essps_min = [ess_per_sec[p][:rhmc].min for p in P_values]
rhmc_essps_med = [ess_per_sec[p][:rhmc].median for p in P_values]
rhmc_essps_mean = [ess_per_sec[p][:rhmc].mean for p in P_values]

dense_essps_min = [ess_per_sec[p][:hmc_dense].min for p in P_values]
dense_essps_med = [ess_per_sec[p][:hmc_dense].median for p in P_values]
dense_essps_mean = [ess_per_sec[p][:hmc_dense].mean for p in P_values]

diag_essps_min = [ess_per_sec[p][:hmc_diag].min for p in P_values]
diag_essps_med = [ess_per_sec[p][:hmc_diag].median for p in P_values]
diag_essps_mean = [ess_per_sec[p][:hmc_diag].mean for p in P_values]

emp_essps_min = [ess_per_sec[p][:hmc_empirical].min for p in P_values]
emp_essps_med = [ess_per_sec[p][:hmc_empirical].median for p in P_values]
emp_essps_mean = [ess_per_sec[p][:hmc_empirical].mean for p in P_values]

# Plot 1: Raw ESS (min)
p_ess_min = plot(;
    title="Minimum ESS vs Block Dim (P)",
    xlabel="Block Dimension (P)",
    ylabel="Min ESS",
    legend=:topright,
    yscale=:log10,
    size=(600, 400),
)
plot!(p_ess_min, P_values, rhmc_ess_min; label="RHMC", lw=2, marker=:circle, color=:blue)
plot!(
    p_ess_min,
    P_values,
    dense_ess_min;
    label="HMC (Dense)",
    lw=2,
    marker=:square,
    color=:green,
)
plot!(
    p_ess_min,
    P_values,
    diag_ess_min;
    label="HMC (Diagonal)",
    lw=2,
    marker=:diamond,
    color=:orange,
)
plot!(
    p_ess_min,
    P_values,
    emp_ess_min;
    label="HMC (Empirical)",
    lw=2,
    marker=:utriangle,
    color=:purple,
)

# Plot 2: Raw ESS (median)
p_ess_med = plot(;
    title="Median ESS vs Block Dim (P)",
    xlabel="Block Dimension (P)",
    ylabel="Median ESS",
    legend=:topright,
    yscale=:log10,
    size=(600, 400),
)
plot!(p_ess_med, P_values, rhmc_ess_med; label="RHMC", lw=2, marker=:circle, color=:blue)
plot!(
    p_ess_med,
    P_values,
    dense_ess_med;
    label="HMC (Dense)",
    lw=2,
    marker=:square,
    color=:green,
)
plot!(
    p_ess_med,
    P_values,
    diag_ess_med;
    label="HMC (Diagonal)",
    lw=2,
    marker=:diamond,
    color=:orange,
)
plot!(
    p_ess_med,
    P_values,
    emp_ess_med;
    label="HMC (Empirical)",
    lw=2,
    marker=:utriangle,
    color=:purple,
)

# Plot 3: Raw ESS (mean)
p_ess_mean = plot(;
    title="Mean ESS vs Block Dim (P)",
    xlabel="Block Dimension (P)",
    ylabel="Mean ESS",
    legend=:topright,
    yscale=:log10,
    size=(600, 400),
)
plot!(p_ess_mean, P_values, rhmc_ess_mean; label="RHMC", lw=2, marker=:circle, color=:blue)
plot!(
    p_ess_mean,
    P_values,
    dense_ess_mean;
    label="HMC (Dense)",
    lw=2,
    marker=:square,
    color=:green,
)
plot!(
    p_ess_mean,
    P_values,
    diag_ess_mean;
    label="HMC (Diagonal)",
    lw=2,
    marker=:diamond,
    color=:orange,
)
plot!(
    p_ess_mean,
    P_values,
    emp_ess_mean;
    label="HMC (Empirical)",
    lw=2,
    marker=:utriangle,
    color=:purple,
)

# Plot 4: ESS/s (min)
p_essps_min = plot(;
    title="Min ESS/s vs Block Dim (P)",
    xlabel="Block Dimension (P)",
    ylabel="Min ESS/s",
    legend=:topright,
    yscale=:log10,
    size=(600, 400),
)
plot!(
    p_essps_min, P_values, rhmc_essps_min; label="RHMC", lw=2, marker=:circle, color=:blue
)
plot!(
    p_essps_min,
    P_values,
    dense_essps_min;
    label="HMC (Dense)",
    lw=2,
    marker=:square,
    color=:green,
)
plot!(
    p_essps_min,
    P_values,
    diag_essps_min;
    label="HMC (Diagonal)",
    lw=2,
    marker=:diamond,
    color=:orange,
)
plot!(
    p_essps_min,
    P_values,
    emp_essps_min;
    label="HMC (Empirical)",
    lw=2,
    marker=:utriangle,
    color=:purple,
)

# Plot 5: ESS/s (median)
p_essps_med = plot(;
    title="Median ESS/s vs Block Dim (P)",
    xlabel="Block Dimension (P)",
    ylabel="Median ESS/s",
    legend=:topright,
    yscale=:log10,
    size=(600, 400),
)
plot!(
    p_essps_med, P_values, rhmc_essps_med; label="RHMC", lw=2, marker=:circle, color=:blue
)
plot!(
    p_essps_med,
    P_values,
    dense_essps_med;
    label="HMC (Dense)",
    lw=2,
    marker=:square,
    color=:green,
)
plot!(
    p_essps_med,
    P_values,
    diag_essps_med;
    label="HMC (Diagonal)",
    lw=2,
    marker=:diamond,
    color=:orange,
)
plot!(
    p_essps_med,
    P_values,
    emp_essps_med;
    label="HMC (Empirical)",
    lw=2,
    marker=:utriangle,
    color=:purple,
)

# Plot 6: ESS/s (mean)
p_essps_mean = plot(;
    title="Mean ESS/s vs Block Dim (P)",
    xlabel="Block Dimension (P)",
    ylabel="Mean ESS/s",
    legend=:topright,
    yscale=:log10,
    size=(600, 400),
)
plot!(
    p_essps_mean, P_values, rhmc_essps_mean; label="RHMC", lw=2, marker=:circle, color=:blue
)
plot!(
    p_essps_mean,
    P_values,
    dense_essps_mean;
    label="HMC (Dense)",
    lw=2,
    marker=:square,
    color=:green,
)
plot!(
    p_essps_mean,
    P_values,
    diag_essps_mean;
    label="HMC (Diagonal)",
    lw=2,
    marker=:diamond,
    color=:orange,
)
plot!(
    p_essps_mean,
    P_values,
    emp_essps_mean;
    label="HMC (Empirical)",
    lw=2,
    marker=:utriangle,
    color=:purple,
)

# Combined plot: Raw ESS
p_raw_ess = plot(
    p_ess_min,
    p_ess_med,
    p_ess_mean;
    layout=(1, 3),
    size=(1400, 400),
    plot_title="Raw ESS vs Block Dimension",
)
display(p_raw_ess)
savefig(p_raw_ess, "../plots/block_dim_raw_ess.png")

# Combined plot: ESS per second
p_ess_per_sec = plot(
    p_essps_min,
    p_essps_med,
    p_essps_mean;
    layout=(1, 3),
    size=(1400, 400),
    plot_title="ESS per Second vs Block Dimension",
)
display(p_ess_per_sec)
savefig(p_ess_per_sec, "../plots/block_dim_ess_per_sec.png")

# Wall time plot
rhmc_times = [results[p][:rhmc].time for p in P_values]
dense_times = [results[p][:hmc_dense].time for p in P_values]
diag_times = [results[p][:hmc_diag].time for p in P_values]
emp_times = [results[p][:hmc_empirical].time for p in P_values]

p_time = plot(;
    title="Wall Time vs Block Dim (P)",
    xlabel="Block Dimension (P)",
    ylabel="Time (s)",
    legend=:topleft,
    yscale=:log10,
    size=(600, 400),
)
plot!(p_time, P_values, rhmc_times; label="RHMC", lw=2, marker=:circle, color=:blue)
plot!(
    p_time, P_values, dense_times; label="HMC (Dense)", lw=2, marker=:square, color=:green
)
plot!(
    p_time,
    P_values,
    diag_times;
    label="HMC (Diagonal)",
    lw=2,
    marker=:diamond,
    color=:orange,
)
plot!(
    p_time,
    P_values,
    emp_times;
    label="HMC (Empirical)",
    lw=2,
    marker=:utriangle,
    color=:purple,
)
display(p_time)
savefig(p_time, "../plots/block_dim_wall_time.png")

println("\nPlots saved to:")
println("  - ../plots/block_dim_raw_ess.png")
println("  - ../plots/block_dim_ess_per_sec.png")
println("  - ../plots/block_dim_wall_time.png")

println("\nDone!")
