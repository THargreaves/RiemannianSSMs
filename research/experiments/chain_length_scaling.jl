"""
Chain length scaling experiment for Van der Pol oscillator model.

This experiment varies N_obs (number of observations) to test how the different
samplers scale with increasing chain length in the block-tridiagonal structure.
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
using MCMCDiagnosticTools

using RiemannianSSMs

# ============================================================================
# Configuration
# ============================================================================

SEED = 42

# Time discretization
δt = 0.1           # Euler step size
K_y = 10           # Latent steps between observations

# N_obs values to test
N_obs_values = [2, 5, 10, 20, 50]

# Sampling configuration
N_samples = 2000
N_adapt = 1000

# ============================================================================
# Model setup (constant across experiments)
# ============================================================================

# True parameter value
μ_true = 1.0
θ_true = @SVector [log(μ_true)]

# Create model using unified interface
ssm_model = VanDerPolModel(;
    δt=δt,
    σ_u=0.01,
    σ_v=0.1,
    σ_obs=0.5,
    μ0=SVector{2,Float64}(1.0, 0.0),
    Σ0_diag=SVector{2,Float64}(0.1, 0.1),
)

# Dimensions
Dx = state_dim(ssm_model)
Dp = param_dim(ssm_model)

# Parameter prior
prior_mean = @SVector [log(μ_true)]
prior_var = @SVector [4.0]

# ============================================================================
# Helper functions
# ============================================================================

function compute_ess_stats(samples_array)
    ess_vals = ess(samples_array)
    return (
        min=minimum(ess_vals),
        median=median(ess_vals),
        mean=mean(ess_vals),
        ess_vec=vec(ess_vals),
    )
end

function run_rhmc(model, ssm_model, ys, K, prior_mean, prior_var, obs_indices, initial_θ)
    metric = RiemannianMetric(
        ssm_model, ys, K, prior_mean, prior_var; obs_indices=obs_indices
    )
    initial_ϵ = 0.005
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
        progress=false,
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
        progress=false,
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
        progress=false,
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
        progress=false,
    )
    elapsed = time() - time_start

    samples = [s.z.θ for s in chains]
    return samples, elapsed
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
# Run benchmarks
# ============================================================================

results = Dict{Int,Dict{Symbol,NamedTuple}}()

for N_obs in N_obs_values
    println("\n" * "="^60)
    println("Running benchmark for N_obs = $N_obs")
    println("="^60)

    rng = MersenneTwister(SEED)

    K = N_obs * K_y
    obs_indices = collect(K_y:K_y:K)
    total_dim = Dx * K + Dp

    # Simulate data using unified interface
    zs_true, ys = simulate(rng, ssm_model, θ_true, K; obs_indices=obs_indices)
    zs_true_block = BlockVector{Float64,Dx}(zs_true)

    # Setup model using unified interface
    ℓπ = RHMCLogDensity(
        ssm_model, ys, K; prior_mean=prior_mean, prior_var=prior_var, obs_indices=obs_indices
    )
    adv_model = AdvancedHMC.LogDensityModel(ℓπ)
    initial_θ = vcat(from_block_vector(zs_true_block), collect(θ_true))

    results[N_obs] = Dict{Symbol,NamedTuple}()

    # Run RHMC
    println("  Running RHMC...")
    rhmc_samples, rhmc_time = run_rhmc(
        adv_model, ssm_model, ys, K, prior_mean, prior_var, obs_indices, initial_θ
    )
    rhmc_arr = samples_to_array(rhmc_samples, total_dim)
    rhmc_ess = compute_ess_stats(rhmc_arr)
    results[N_obs][:rhmc] = (
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
    results[N_obs][:hmc_dense] = (
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
    results[N_obs][:hmc_diag] = (
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
    results[N_obs][:hmc_empirical] = (
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

for N_obs in N_obs_values
    ess_per_sec[N_obs] = Dict{Symbol,NamedTuple}()
    for method in [:rhmc, :hmc_dense, :hmc_diag, :hmc_empirical]
        r = results[N_obs][method]
        ess_per_sec[N_obs][method] = (
            min=(r.ess_min / r.time),
            median=(r.ess_median / r.time),
            mean=(r.ess_mean / r.time),
        )
    end
end

# ============================================================================
# Print summary table
# ============================================================================

println("\n" * "="^100)
println("SUMMARY: Raw ESS (min / median / mean)")
println("="^100)
println(
    "N_obs  |        RHMC         |        Dense        |        Diag         |      Empirical",
)
println("-"^100)
for N_obs in N_obs_values
    rhmc = results[N_obs][:rhmc]
    dense = results[N_obs][:hmc_dense]
    diag = results[N_obs][:hmc_diag]
    emp = results[N_obs][:hmc_empirical]
    @printf(
        "%5d  | %5.0f / %5.0f / %5.0f | %5.0f / %5.0f / %5.0f | %5.0f / %5.0f / %5.0f | %5.0f / %5.0f / %5.0f\n",
        N_obs,
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
    "N_obs  |        RHMC         |        Dense        |        Diag         |      Empirical",
)
println("-"^100)
for N_obs in N_obs_values
    rhmc = ess_per_sec[N_obs][:rhmc]
    dense = ess_per_sec[N_obs][:hmc_dense]
    diag = ess_per_sec[N_obs][:hmc_diag]
    emp = ess_per_sec[N_obs][:hmc_empirical]
    @printf(
        "%5d  | %5.1f / %5.1f / %5.1f | %5.1f / %5.1f / %5.1f | %5.1f / %5.1f / %5.1f | %5.1f / %5.1f / %5.1f\n",
        N_obs,
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
println("N_obs  |   RHMC   |  Dense   |   Diag   | Empirical")
println("-"^60)
for N_obs in N_obs_values
    rhmc = results[N_obs][:rhmc]
    dense = results[N_obs][:hmc_dense]
    diag = results[N_obs][:hmc_diag]
    emp = results[N_obs][:hmc_empirical]
    @printf(
        "%5d  | %8.2f | %8.2f | %8.2f | %8.2f\n",
        N_obs,
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
rhmc_ess_min = [results[n][:rhmc].ess_min for n in N_obs_values]
rhmc_ess_med = [results[n][:rhmc].ess_median for n in N_obs_values]
rhmc_ess_mean = [results[n][:rhmc].ess_mean for n in N_obs_values]

dense_ess_min = [results[n][:hmc_dense].ess_min for n in N_obs_values]
dense_ess_med = [results[n][:hmc_dense].ess_median for n in N_obs_values]
dense_ess_mean = [results[n][:hmc_dense].ess_mean for n in N_obs_values]

diag_ess_min = [results[n][:hmc_diag].ess_min for n in N_obs_values]
diag_ess_med = [results[n][:hmc_diag].ess_median for n in N_obs_values]
diag_ess_mean = [results[n][:hmc_diag].ess_mean for n in N_obs_values]

emp_ess_min = [results[n][:hmc_empirical].ess_min for n in N_obs_values]
emp_ess_med = [results[n][:hmc_empirical].ess_median for n in N_obs_values]
emp_ess_mean = [results[n][:hmc_empirical].ess_mean for n in N_obs_values]

rhmc_essps_min = [ess_per_sec[n][:rhmc].min for n in N_obs_values]
rhmc_essps_med = [ess_per_sec[n][:rhmc].median for n in N_obs_values]
rhmc_essps_mean = [ess_per_sec[n][:rhmc].mean for n in N_obs_values]

dense_essps_min = [ess_per_sec[n][:hmc_dense].min for n in N_obs_values]
dense_essps_med = [ess_per_sec[n][:hmc_dense].median for n in N_obs_values]
dense_essps_mean = [ess_per_sec[n][:hmc_dense].mean for n in N_obs_values]

diag_essps_min = [ess_per_sec[n][:hmc_diag].min for n in N_obs_values]
diag_essps_med = [ess_per_sec[n][:hmc_diag].median for n in N_obs_values]
diag_essps_mean = [ess_per_sec[n][:hmc_diag].mean for n in N_obs_values]

emp_essps_min = [ess_per_sec[n][:hmc_empirical].min for n in N_obs_values]
emp_essps_med = [ess_per_sec[n][:hmc_empirical].median for n in N_obs_values]
emp_essps_mean = [ess_per_sec[n][:hmc_empirical].mean for n in N_obs_values]

# Plot 1: Raw ESS (min)
p_ess_min = plot(;
    title="Minimum ESS vs N_obs",
    xlabel="N_obs",
    ylabel="Min ESS",
    legend=:topright,
    xscale=:log10,
    yscale=:log10,
    size=(600, 400),
)
plot!(
    p_ess_min, N_obs_values, rhmc_ess_min; label="RHMC", lw=2, marker=:circle, color=:blue
)
plot!(
    p_ess_min,
    N_obs_values,
    dense_ess_min;
    label="HMC (Dense)",
    lw=2,
    marker=:square,
    color=:green,
)
plot!(
    p_ess_min,
    N_obs_values,
    diag_ess_min;
    label="HMC (Diagonal)",
    lw=2,
    marker=:diamond,
    color=:orange,
)
plot!(
    p_ess_min,
    N_obs_values,
    emp_ess_min;
    label="HMC (Empirical)",
    lw=2,
    marker=:utriangle,
    color=:purple,
)

# Plot 2: Raw ESS (median)
p_ess_med = plot(;
    title="Median ESS vs N_obs",
    xlabel="N_obs",
    ylabel="Median ESS",
    legend=:topright,
    xscale=:log10,
    yscale=:log10,
    size=(600, 400),
)
plot!(
    p_ess_med, N_obs_values, rhmc_ess_med; label="RHMC", lw=2, marker=:circle, color=:blue
)
plot!(
    p_ess_med,
    N_obs_values,
    dense_ess_med;
    label="HMC (Dense)",
    lw=2,
    marker=:square,
    color=:green,
)
plot!(
    p_ess_med,
    N_obs_values,
    diag_ess_med;
    label="HMC (Diagonal)",
    lw=2,
    marker=:diamond,
    color=:orange,
)
plot!(
    p_ess_med,
    N_obs_values,
    emp_ess_med;
    label="HMC (Empirical)",
    lw=2,
    marker=:utriangle,
    color=:purple,
)

# Plot 3: Raw ESS (mean)
p_ess_mean = plot(;
    title="Mean ESS vs N_obs",
    xlabel="N_obs",
    ylabel="Mean ESS",
    legend=:topright,
    xscale=:log10,
    yscale=:log10,
    size=(600, 400),
)
plot!(
    p_ess_mean, N_obs_values, rhmc_ess_mean; label="RHMC", lw=2, marker=:circle, color=:blue
)
plot!(
    p_ess_mean,
    N_obs_values,
    dense_ess_mean;
    label="HMC (Dense)",
    lw=2,
    marker=:square,
    color=:green,
)
plot!(
    p_ess_mean,
    N_obs_values,
    diag_ess_mean;
    label="HMC (Diagonal)",
    lw=2,
    marker=:diamond,
    color=:orange,
)
plot!(
    p_ess_mean,
    N_obs_values,
    emp_ess_mean;
    label="HMC (Empirical)",
    lw=2,
    marker=:utriangle,
    color=:purple,
)

# Plot 4: ESS/s (min)
p_essps_min = plot(;
    title="Min ESS/s vs N_obs",
    xlabel="N_obs",
    ylabel="Min ESS/s",
    legend=:topright,
    xscale=:log10,
    yscale=:log10,
    size=(600, 400),
)
plot!(
    p_essps_min,
    N_obs_values,
    rhmc_essps_min;
    label="RHMC",
    lw=2,
    marker=:circle,
    color=:blue,
)
plot!(
    p_essps_min,
    N_obs_values,
    dense_essps_min;
    label="HMC (Dense)",
    lw=2,
    marker=:square,
    color=:green,
)
plot!(
    p_essps_min,
    N_obs_values,
    diag_essps_min;
    label="HMC (Diagonal)",
    lw=2,
    marker=:diamond,
    color=:orange,
)
plot!(
    p_essps_min,
    N_obs_values,
    emp_essps_min;
    label="HMC (Empirical)",
    lw=2,
    marker=:utriangle,
    color=:purple,
)

# Plot 5: ESS/s (median)
p_essps_med = plot(;
    title="Median ESS/s vs N_obs",
    xlabel="N_obs",
    ylabel="Median ESS/s",
    legend=:topright,
    xscale=:log10,
    yscale=:log10,
    size=(600, 400),
)
plot!(
    p_essps_med,
    N_obs_values,
    rhmc_essps_med;
    label="RHMC",
    lw=2,
    marker=:circle,
    color=:blue,
)
plot!(
    p_essps_med,
    N_obs_values,
    dense_essps_med;
    label="HMC (Dense)",
    lw=2,
    marker=:square,
    color=:green,
)
plot!(
    p_essps_med,
    N_obs_values,
    diag_essps_med;
    label="HMC (Diagonal)",
    lw=2,
    marker=:diamond,
    color=:orange,
)
plot!(
    p_essps_med,
    N_obs_values,
    emp_essps_med;
    label="HMC (Empirical)",
    lw=2,
    marker=:utriangle,
    color=:purple,
)

# Plot 6: ESS/s (mean)
p_essps_mean = plot(;
    title="Mean ESS/s vs N_obs",
    xlabel="N_obs",
    ylabel="Mean ESS/s",
    legend=:topright,
    xscale=:log10,
    yscale=:log10,
    size=(600, 400),
)
plot!(
    p_essps_mean,
    N_obs_values,
    rhmc_essps_mean;
    label="RHMC",
    lw=2,
    marker=:circle,
    color=:blue,
)
plot!(
    p_essps_mean,
    N_obs_values,
    dense_essps_mean;
    label="HMC (Dense)",
    lw=2,
    marker=:square,
    color=:green,
)
plot!(
    p_essps_mean,
    N_obs_values,
    diag_essps_mean;
    label="HMC (Diagonal)",
    lw=2,
    marker=:diamond,
    color=:orange,
)
plot!(
    p_essps_mean,
    N_obs_values,
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
    plot_title="Raw ESS Comparison",
)
display(p_raw_ess)
savefig(p_raw_ess, "../plots/vdp_benchmark_raw_ess.png")

# Combined plot: ESS per second
p_ess_per_sec = plot(
    p_essps_min,
    p_essps_med,
    p_essps_mean;
    layout=(1, 3),
    size=(1400, 400),
    plot_title="ESS per Second Comparison",
)
display(p_ess_per_sec)
savefig(p_ess_per_sec, "../plots/vdp_benchmark_ess_per_sec.png")

# Wall time plot
rhmc_times = [results[n][:rhmc].time for n in N_obs_values]
dense_times = [results[n][:hmc_dense].time for n in N_obs_values]
diag_times = [results[n][:hmc_diag].time for n in N_obs_values]
emp_times = [results[n][:hmc_empirical].time for n in N_obs_values]

p_time = plot(;
    title="Wall Time vs N_obs",
    xlabel="N_obs",
    ylabel="Time (s)",
    legend=:topleft,
    xscale=:log10,
    yscale=:log10,
    size=(600, 400),
)
plot!(p_time, N_obs_values, rhmc_times; label="RHMC", lw=2, marker=:circle, color=:blue)
plot!(
    p_time,
    N_obs_values,
    dense_times;
    label="HMC (Dense)",
    lw=2,
    marker=:square,
    color=:green,
)
plot!(
    p_time,
    N_obs_values,
    diag_times;
    label="HMC (Diagonal)",
    lw=2,
    marker=:diamond,
    color=:orange,
)
plot!(
    p_time,
    N_obs_values,
    emp_times;
    label="HMC (Empirical)",
    lw=2,
    marker=:utriangle,
    color=:purple,
)
display(p_time)
savefig(p_time, "../plots/vdp_benchmark_wall_time.png")

println("\nPlots saved to:")
println("  - ../plots/vdp_benchmark_raw_ess.png")
println("  - ../plots/vdp_benchmark_ess_per_sec.png")
println("  - ../plots/vdp_benchmark_wall_time.png")

println("\nDone!")
