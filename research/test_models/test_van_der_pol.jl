"""
Test script for the unified RHMC interface with Van der Pol oscillator.

This replicates test_vdp_benchmark_param.jl using the new unified interface.
"""

using Dates
using Distributions
using JLD2
using LinearAlgebra
using Random
using Plots
using StaticArrays

using AbstractMCMC
using AdvancedHMC
using MCMCDiagnosticTools

using RiemannianSSMs

# ============================================================================
# Test Configuration
# ============================================================================

RUN_RHMC = true           # My method (Fisher/GGN)
RUN_MCRHMC = false        # MC-RHMC (Kleppe's Observed Hessian)
RUN_HMC_DIAGONAL = true   # HMC with diagonal metric
RUN_HMC_DENSE = true      # HMC with dense metric
RUN_HMC_EMPIRICAL = true  # HMC with empirical covariance from RHMC samples

# ============================================================================
# General Configuration
# ============================================================================

SEED = 42
rng = MersenneTwister(SEED)

# Time discretization
δt = 0.01           # Euler step size
K_y = 100           # Latent steps between observations
N_obs = 20         # Number of observations
K = N_obs * K_y    # Total number of latent states (200)

# Observation indices (1-indexed, every K_y-th state starting from K_y)
obs_indices = collect(K_y:K_y:K)

# Sampling parameters
N_samples = 2000
N_adapt = 1000
burn_in = N_adapt

# ============================================================================
# Model Setup
# ============================================================================

# True parameter value
μ_true = 3.0
θ_true = @SVector [log(μ_true)]

# Create model using new interface
model = VanDerPolModel(;
    δt=δt,
    σ_u=0.001,
    σ_v=0.01,
    σ_obs=0.1,
    μ0=SVector{2,Float64}(1.0, 0.0),
    Σ0_diag=SVector{2,Float64}(0.1, 0.1),
)

# Dimensions
Dx = state_dim(model)
Dy = obs_dim(model)
Dp = param_dim(model)
total_dim = Dx * K + Dp

println("Model: VanDerPolModel")
println("State dim: $Dx, Obs dim: $Dy, Param dim: $Dp")

# ============================================================================
# Ground truth simulation
# ============================================================================

println("\nSimulating Van der Pol oscillator with μ = $(μ_true)...")
println("Total latent states: K = $K")
println("Number of observations: $(length(obs_indices))")
println("Observations at indices: $(obs_indices[1]):$(K_y):$(obs_indices[end])")

zs_true, ys = simulate(rng, model, θ_true, K; obs_indices=obs_indices)
zs_true_block = BlockVector{Float64,2}(zs_true)

# Plot the trajectory
p1 = plot(;
    title="Van der Pol Oscillator (μ=$(μ_true))",
    xlabel="u",
    ylabel="v",
    legend=:topright,
    size=(800, 600),
    aspect_ratio=1,
)
plot!(
    p1, [z[1] for z in zs_true], [z[2] for z in zs_true]; label="Truth", lw=2, color=:black
)

# Plot observed u coordinate with true v
scatter!(
    p1,
    only.(ys),
    getindex.(zs_true[obs_indices], 2);
    label="Observations",
    color=:blue,
    ms=4,
)

# Add dashed line between true and observed points
for (k, y) in zip(obs_indices, ys)
    plot!(
        p1,
        [zs_true[k][1], y[1]],
        [zs_true[k][2], zs_true[k][2]];
        l=:dot,
        color=:black,
        alpha=0.8,
        label="",
    )
end

display(p1)

# ============================================================================
# LogDensityProblems Model (Joint state + parameter inference)
# ============================================================================

# Parameter prior (diffuse Gaussian on log scale)
prior_mean = @SVector [log(μ_true)]  # Centered near true value
prior_var = @SVector [4.0]           # Wide variance

ℓπ = RHMCLogDensity(
    model, ys, K; prior_mean=prior_mean, prior_var=prior_var, obs_indices=obs_indices
)
adv_model = AdvancedHMC.LogDensityModel(ℓπ)

# Initial state: true states + true parameters
initial_θ = vcat(from_block_vector(zs_true_block), collect(θ_true))

# Storage for results
results = Dict{String,NamedTuple}()

# ============================================================================
# RHMC Sampling (My method - Fisher/GGN)
# ============================================================================

if RUN_RHMC
    println("\n" * "="^60)
    println("RHMC (MY METHOD - FISHER/GGN)")
    println("="^60)

    println("\nSetting up RHMC sampler with unified interface...")
    metric = RiemannianMetric(model, ys, K, prior_mean, prior_var; obs_indices=obs_indices)
    hamiltonian = Hamiltonian(metric, ℓπ)
    initial_ϵ = 0.005
    integrator = AdaptiveGeneralizedLeapfrog(initial_ϵ; max_iters=7)
    kernel = HMCKernel(Trajectory{MultinomialTS}(integrator, GeneralisedNoUTurn()))
    adaptor = StepSizeAdaptor(0.9, integrator)
    rhmc = HMCSampler(kernel, metric, adaptor)

    println("Running RHMC with joint state+parameter inference...")
    elapsed_time = @elapsed begin
        chains = AbstractMCMC.sample(
            adv_model,
            rhmc,
            N_samples;
            n_adapts=N_adapt,
            initial_params=initial_θ,
            verbose=false,
            progress=true,
        )
    end
    samples = [s.z.θ for s in chains]
    println("Wall time: $(round(elapsed_time, digits=2)) seconds")

    # Extract parameter samples (last Dp elements)
    μ_samples = [exp(s[K * Dx + 1]) for s in samples]

    # Compute ESS for all dimensions
    rhmc_samples = Array{Float64}(undef, N_samples, 1, total_dim)
    for i in 1:N_samples
        for j in 1:total_dim
            rhmc_samples[i, 1, j] = samples[i][j]
        end
    end
    rhmc_ess = ess(rhmc_samples) ./ N_samples

    println("\n=== RHMC State ESS Statistics ===")
    state_ess = rhmc_ess[1:(Dx * K)]
    println("Minimum State ESS: ", minimum(state_ess))
    println("Median State ESS: ", median(state_ess))
    println("Mean State ESS: ", mean(state_ess))

    println("\n=== RHMC Parameter ESS ===")
    param_ess = rhmc_ess[(Dx * K + 1):end]
    println("log(μ) ESS: ", param_ess[1])

    println("\n=== RHMC Parameter Posterior Summary ===")
    μ_post = μ_samples[(burn_in + 1):end]
    println("μ: true=$(μ_true), posterior mean=$(mean(μ_post)), std=$(std(μ_post))")

    results["RHMC"] = (
        samples=samples,
        μ_samples=μ_samples,
        μ_post=μ_post,
        state_ess=state_ess,
        param_ess=param_ess,
        elapsed_time=elapsed_time,
        state_ess_per_sec=state_ess ./ elapsed_time,
        param_ess_per_sec=param_ess ./ elapsed_time,
        color=:blue,
        label="RHMC (Fisher/GGN)",
    )
end

# ============================================================================
# MCRHMC Sampling (Kleppe's Observed Hessian)
# ============================================================================

if RUN_MCRHMC
    println("\n" * "="^60)
    println("MC-RHMC (KLEPPE'S OBSERVED HESSIAN)")
    println("="^60)

    n = K * Dx + Dp
    u_init = fill(1e-5, n)
    u_init[end] = 10.0  # Larger regularization for parameter

    println(
        "\nSetting up Kleppe's Observed Hessian metric (GeneralizedObservedHessianMetric)...",
    )
    mcrhmc_metric = GeneralizedObservedHessianMetric(
        model, ys, K, prior_mean, prior_var; obs_indices=obs_indices, u_init=u_init
    )
    println(mcrhmc_metric)

    mcrhmc_hamiltonian = Hamiltonian(mcrhmc_metric, ℓπ)
    mcrhmc_integrator = AdaptiveGeneralizedLeapfrog(0.005; max_iters=7)
    mcrhmc_kernel = HMCKernel(
        Trajectory{MultinomialTS}(mcrhmc_integrator, GeneralisedNoUTurn())
    )
    mcrhmc_adaptor = StepSizeAdaptor(0.8, mcrhmc_integrator)
    mcrhmc_sampler = HMCSampler(mcrhmc_kernel, mcrhmc_metric, mcrhmc_adaptor)

    println("Running MC-RHMC (N_samples=$N_samples, N_adapt=$N_adapt)...")
    elapsed_time = @elapsed begin
        mcrhmc_chains = AbstractMCMC.sample(
            adv_model,
            mcrhmc_sampler,
            N_samples;
            n_adapts=N_adapt,
            initial_params=initial_θ,
            verbose=false,
            progress=true,
        )
    end
    mcrhmc_samples_raw = [s.z.θ for s in mcrhmc_chains]
    println("Wall time: $(round(elapsed_time, digits=2)) seconds")

    # Print acceptance statistics
    mcrhmc_accept = [s.stat.is_accept for s in mcrhmc_chains]
    println("MC-RHMC Acceptance Rate: ", mean(mcrhmc_accept))

    mcrhmc_accept_prop = [s.stat.acceptance_rate for s in mcrhmc_chains]
    println("MC-RHMC Mean Acceptance Probability: ", mean(mcrhmc_accept_prop))

    # Extract parameter samples
    mcrhmc_μ_samples = [exp(s[K * Dx + 1]) for s in mcrhmc_samples_raw]

    # Compute ESS
    mcrhmc_samples_arr = Array{Float64}(undef, N_samples, 1, total_dim)
    for i in 1:N_samples
        for j in 1:total_dim
            mcrhmc_samples_arr[i, 1, j] = mcrhmc_samples_raw[i][j]
        end
    end
    mcrhmc_ess = ess(mcrhmc_samples_arr) ./ N_samples

    println("\n=== MC-RHMC State ESS Statistics ===")
    mcrhmc_state_ess = mcrhmc_ess[1:(Dx * K)]
    println("Minimum State ESS: ", minimum(mcrhmc_state_ess))
    println("Median State ESS: ", median(mcrhmc_state_ess))
    println("Mean State ESS: ", mean(mcrhmc_state_ess))

    println("\n=== MC-RHMC Parameter ESS ===")
    mcrhmc_param_ess = mcrhmc_ess[(Dx * K + 1):end]
    println("log(μ) ESS: ", mcrhmc_param_ess[1])

    println("\n=== MC-RHMC Parameter Posterior Summary ===")
    mcrhmc_μ_post = mcrhmc_μ_samples[(burn_in + 1):end]
    println(
        "μ: true=$(μ_true), posterior mean=$(mean(mcrhmc_μ_post)), std=$(std(mcrhmc_μ_post))",
    )

    results["MCRHMC"] = (
        samples=mcrhmc_samples_raw,
        μ_samples=mcrhmc_μ_samples,
        μ_post=mcrhmc_μ_post,
        state_ess=mcrhmc_state_ess,
        param_ess=mcrhmc_param_ess,
        elapsed_time=elapsed_time,
        state_ess_per_sec=mcrhmc_state_ess ./ elapsed_time,
        param_ess_per_sec=mcrhmc_param_ess ./ elapsed_time,
        color=:purple,
        label="MC-RHMC (Obs Hessian)",
    )
end

# ============================================================================
# HMC Sampling (Diagonal Metric)
# ============================================================================

if RUN_HMC_DIAGONAL
    println("\n" * "="^60)
    println("HMC (DIAGONAL METRIC)")
    println("="^60)

    println("\nRunning HMC with diagonal metric...")
    hmc_diag_metric = AdvancedHMC.DiagEuclideanMetric(total_dim)
    hmc_diag_integrator = Leapfrog(0.001)
    hmc_diag_kernel = HMCKernel(
        Trajectory{MultinomialTS}(hmc_diag_integrator, GeneralisedNoUTurn())
    )
    hmc_diag_adaptor = StanHMCAdaptor(
        MassMatrixAdaptor(hmc_diag_metric), StepSizeAdaptor(0.8, hmc_diag_integrator)
    )
    hmc_diag = HMCSampler(hmc_diag_kernel, hmc_diag_metric, hmc_diag_adaptor)

    elapsed_time = @elapsed begin
        hmc_diag_chains = AbstractMCMC.sample(
            adv_model,
            hmc_diag,
            N_samples;
            n_adapts=N_adapt,
            initial_params=initial_θ,
            verbose=false,
            progress=true,
        )
    end
    hmc_diag_samples_raw = [s.z.θ for s in hmc_diag_chains]
    println("Wall time: $(round(elapsed_time, digits=2)) seconds")

    # Extract parameter samples
    μ_samples_hmc_diag = [exp(s[K * Dx + 1]) for s in hmc_diag_samples_raw]

    # Compute ESS
    hmc_diag_samples = Array{Float64}(undef, N_samples, 1, total_dim)
    for i in 1:N_samples
        for j in 1:total_dim
            hmc_diag_samples[i, 1, j] = hmc_diag_samples_raw[i][j]
        end
    end
    hmc_diag_ess = ess(hmc_diag_samples) ./ N_samples

    println("\n=== HMC (Diagonal) State ESS Statistics ===")
    hmc_diag_state_ess = hmc_diag_ess[1:(Dx * K)]
    println("Minimum State ESS: ", minimum(hmc_diag_state_ess))
    println("Median State ESS: ", median(hmc_diag_state_ess))
    println("Mean State ESS: ", mean(hmc_diag_state_ess))

    println("\n=== HMC (Diagonal) Parameter ESS ===")
    hmc_diag_param_ess = hmc_diag_ess[(Dx * K + 1):end]
    println("log(μ) ESS: ", hmc_diag_param_ess[1])

    println("\n=== HMC (Diagonal) Parameter Posterior Summary ===")
    μ_post_hmc_diag = μ_samples_hmc_diag[(burn_in + 1):end]
    println(
        "μ: true=$(μ_true), posterior mean=$(mean(μ_post_hmc_diag)), std=$(std(μ_post_hmc_diag))",
    )

    results["HMC_Diagonal"] = (
        samples=hmc_diag_samples_raw,
        μ_samples=μ_samples_hmc_diag,
        μ_post=μ_post_hmc_diag,
        state_ess=hmc_diag_state_ess,
        param_ess=hmc_diag_param_ess,
        elapsed_time=elapsed_time,
        state_ess_per_sec=hmc_diag_state_ess ./ elapsed_time,
        param_ess_per_sec=hmc_diag_param_ess ./ elapsed_time,
        color=:orange,
        label="HMC (Diagonal)",
    )
end

# ============================================================================
# HMC Sampling (Dense Metric)
# ============================================================================

if RUN_HMC_DENSE
    println("\n" * "="^60)
    println("HMC (DENSE METRIC)")
    println("="^60)

    println("\nRunning HMC with dense metric...")
    hmc_dense_metric = AdvancedHMC.DenseEuclideanMetric(total_dim)
    hmc_dense_integrator = Leapfrog(0.001)
    hmc_dense_kernel = HMCKernel(
        Trajectory{MultinomialTS}(hmc_dense_integrator, GeneralisedNoUTurn())
    )
    hmc_dense_adaptor = StanHMCAdaptor(
        MassMatrixAdaptor(hmc_dense_metric), StepSizeAdaptor(0.8, hmc_dense_integrator)
    )
    hmc_dense = HMCSampler(hmc_dense_kernel, hmc_dense_metric, hmc_dense_adaptor)

    elapsed_time = @elapsed begin
        hmc_dense_chains = AbstractMCMC.sample(
            adv_model,
            hmc_dense,
            N_samples;
            n_adapts=N_adapt,
            initial_params=initial_θ,
            verbose=false,
            progress=true,
        )
    end
    hmc_dense_samples_raw = [s.z.θ for s in hmc_dense_chains]
    println("Wall time: $(round(elapsed_time, digits=2)) seconds")

    # Extract parameter samples
    μ_samples_hmc_dense = [exp(s[K * Dx + 1]) for s in hmc_dense_samples_raw]

    # Compute ESS
    hmc_dense_samples = Array{Float64}(undef, N_samples, 1, total_dim)
    for i in 1:N_samples
        for j in 1:total_dim
            hmc_dense_samples[i, 1, j] = hmc_dense_samples_raw[i][j]
        end
    end
    hmc_dense_ess = ess(hmc_dense_samples) ./ N_samples

    println("\n=== HMC (Dense) State ESS Statistics ===")
    hmc_dense_state_ess = hmc_dense_ess[1:(Dx * K)]
    println("Minimum State ESS: ", minimum(hmc_dense_state_ess))
    println("Median State ESS: ", median(hmc_dense_state_ess))
    println("Mean State ESS: ", mean(hmc_dense_state_ess))

    println("\n=== HMC (Dense) Parameter ESS ===")
    hmc_dense_param_ess = hmc_dense_ess[(Dx * K + 1):end]
    println("log(μ) ESS: ", hmc_dense_param_ess[1])

    println("\n=== HMC (Dense) Parameter Posterior Summary ===")
    μ_post_hmc_dense = μ_samples_hmc_dense[(burn_in + 1):end]
    println(
        "μ: true=$(μ_true), posterior mean=$(mean(μ_post_hmc_dense)), std=$(std(μ_post_hmc_dense))",
    )

    results["HMC_Dense"] = (
        samples=hmc_dense_samples_raw,
        μ_samples=μ_samples_hmc_dense,
        μ_post=μ_post_hmc_dense,
        state_ess=hmc_dense_state_ess,
        param_ess=hmc_dense_param_ess,
        elapsed_time=elapsed_time,
        state_ess_per_sec=hmc_dense_state_ess ./ elapsed_time,
        param_ess_per_sec=hmc_dense_param_ess ./ elapsed_time,
        color=:green,
        label="HMC (Dense)",
    )
end

# ============================================================================
# HMC Sampling (Empirical Covariance from RHMC)
# ============================================================================

if RUN_HMC_EMPIRICAL
    if !RUN_RHMC
        @warn "HMC (Empirical) requires RHMC to run first. Skipping..."
    else
        println("\n" * "="^60)
        println("HMC (EMPIRICAL COVARIANCE FROM RHMC)")
        println("="^60)

        println("\nRunning HMC with empirical covariance from RHMC samples...")
        rhmc_samples_for_cov = results["RHMC"].samples
        emp_cov = cov(hcat(rhmc_samples_for_cov...)') + 1e-3 * I
        emp_cov = (emp_cov + emp_cov') / 2  # Ensure symmetry
        hmc_emp_metric = AdvancedHMC.DenseEuclideanMetric(emp_cov)
        hmc_emp_integrator = Leapfrog(0.001)
        hmc_emp_kernel = HMCKernel(
            Trajectory{MultinomialTS}(hmc_emp_integrator, GeneralisedNoUTurn())
        )
        hmc_emp_adaptor = StanHMCAdaptor(
            MassMatrixAdaptor(hmc_emp_metric), StepSizeAdaptor(0.8, hmc_emp_integrator)
        )
        hmc_emp = HMCSampler(hmc_emp_kernel, hmc_emp_metric, hmc_emp_adaptor)

        elapsed_time = @elapsed begin
            hmc_emp_chains = AbstractMCMC.sample(
                adv_model,
                hmc_emp,
                N_samples;
                n_adapts=N_adapt,
                initial_params=initial_θ,
                verbose=false,
                progress=true,
            )
        end
        hmc_emp_samples_raw = [s.z.θ for s in hmc_emp_chains]
        println("Wall time: $(round(elapsed_time, digits=2)) seconds")

        # Extract parameter samples
        μ_samples_hmc_emp = [exp(s[K * Dx + 1]) for s in hmc_emp_samples_raw]

        # Compute ESS
        hmc_emp_samples = Array{Float64}(undef, N_samples, 1, total_dim)
        for i in 1:N_samples
            for j in 1:total_dim
                hmc_emp_samples[i, 1, j] = hmc_emp_samples_raw[i][j]
            end
        end
        hmc_emp_ess = ess(hmc_emp_samples) ./ N_samples

        println("\n=== HMC (Empirical) State ESS Statistics ===")
        hmc_emp_state_ess = hmc_emp_ess[1:(Dx * K)]
        println("Minimum State ESS: ", minimum(hmc_emp_state_ess))
        println("Median State ESS: ", median(hmc_emp_state_ess))
        println("Mean State ESS: ", mean(hmc_emp_state_ess))

        println("\n=== HMC (Empirical) Parameter ESS ===")
        hmc_emp_param_ess = hmc_emp_ess[(Dx * K + 1):end]
        println("log(μ) ESS: ", hmc_emp_param_ess[1])

        println("\n=== HMC (Empirical) Parameter Posterior Summary ===")
        μ_post_hmc_emp = μ_samples_hmc_emp[(burn_in + 1):end]
        println(
            "μ: true=$(μ_true), posterior mean=$(mean(μ_post_hmc_emp)), std=$(std(μ_post_hmc_emp))",
        )

        results["HMC_Empirical"] = (
            samples=hmc_emp_samples_raw,
            μ_samples=μ_samples_hmc_emp,
            μ_post=μ_post_hmc_emp,
            state_ess=hmc_emp_state_ess,
            param_ess=hmc_emp_param_ess,
            elapsed_time=elapsed_time,
            state_ess_per_sec=hmc_emp_state_ess ./ elapsed_time,
            param_ess_per_sec=hmc_emp_param_ess ./ elapsed_time,
            color=:red,
            label="HMC (Empirical)",
        )
    end
end

# ============================================================================
# Comparison Summary
# ============================================================================

function generate_summary(results, μ_true)
    io = IOBuffer()

    println(io, "="^60)
    println(io, "COMPARISON SUMMARY")
    println(io, "="^60)

    println(io, "\n--- Wall Time ---")
    for (name, res) in sort(collect(results); by=x -> x[1])
        println(
            io, "$(rpad(res.label * ":", 25)) $(round(res.elapsed_time, digits=2)) seconds"
        )
    end

    println(io, "\n--- State ESS ---")
    for (name, res) in sort(collect(results); by=x -> x[1])
        println(
            io,
            "$(rpad(res.label * ":", 25)) min=$(round(minimum(res.state_ess), digits=4)), median=$(round(median(res.state_ess), digits=4)), mean=$(round(mean(res.state_ess), digits=4))",
        )
    end

    println(io, "\n--- State ESS/sec ---")
    for (name, res) in sort(collect(results); by=x -> x[1])
        println(
            io,
            "$(rpad(res.label * ":", 25)) min=$(round(minimum(res.state_ess_per_sec), digits=6)), median=$(round(median(res.state_ess_per_sec), digits=6)), mean=$(round(mean(res.state_ess_per_sec), digits=6))",
        )
    end

    println(io, "\n--- Parameter ESS ---")
    for (name, res) in sort(collect(results); by=x -> x[1])
        println(
            io, "$(rpad(res.label * ":", 25)) log(μ)=$(round(res.param_ess[1], digits=4))"
        )
    end

    println(io, "\n--- Parameter ESS/sec ---")
    for (name, res) in sort(collect(results); by=x -> x[1])
        println(
            io,
            "$(rpad(res.label * ":", 25)) log(μ)=$(round(res.param_ess_per_sec[1], digits=6))",
        )
    end

    println(io, "\n--- Parameter Posterior ---")
    for (name, res) in sort(collect(results); by=x -> x[1])
        println(
            io,
            "$(rpad(res.label * ":", 25)) μ mean=$(round(mean(res.μ_post), digits=4)), std=$(round(std(res.μ_post), digits=4))",
        )
    end
    println(io, "True μ: $(μ_true)")

    return String(take!(io))
end

if !isempty(results)
    summary_text = generate_summary(results, μ_true)
    println("\n" * summary_text)
end

# ============================================================================
# Plots
# ============================================================================

if !isempty(results)
    # Comparison trace plots for parameter
    p_μ_compare = plot(;
        title="μ trace comparison",
        xlabel="Sample",
        ylabel="μ",
        legend=:topright,
        size=(1000, 400),
    )
    for (name, res) in results
        plot!(p_μ_compare, res.μ_samples; label=res.label, lw=1, color=res.color, alpha=0.5)
    end
    hline!(p_μ_compare, [μ_true]; label="True", lw=2, color=:black, linestyle=:dash)

    display(p_μ_compare)

    # Histogram comparison of posterior parameter (post burn-in)
    n_methods = length(results)
    p_μ_hist = plot(; layout=(1, n_methods), size=(400 * n_methods, 400))

    for (i, (name, res)) in enumerate(sort(collect(results); by=x -> x[1]))
        histogram!(
            p_μ_hist[i],
            res.μ_post;
            title=res.label,
            xlabel="μ",
            ylabel="Count",
            label="",
            bins=30,
            color=res.color,
            alpha=0.7,
        )
        vline!(p_μ_hist[i], [μ_true]; label="True", lw=2, color=:black, linestyle=:dash)
    end

    display(p_μ_hist)

    # Trajectory comparison plots
    n_plot_samples = min(100, minimum(length(res.μ_post) for (_, res) in results))
    plot_idxs = round.(Int, LinRange(burn_in + 1, N_samples, n_plot_samples))

    p_traj = plot(; layout=(1, n_methods), size=(500 * n_methods, 500))

    for (i, (name, res)) in enumerate(sort(collect(results); by=x -> x[1]))
        for idx in plot_idxs
            s = res.samples[idx]
            plot!(
                p_traj[i],
                [s[1 + (k - 1) * Dx] for k in 1:K],
                [s[2 + (k - 1) * Dx] for k in 1:K];
                label="",
                lw=1,
                alpha=0.05,
                color=res.color,
            )
        end
        plot!(
            p_traj[i],
            [z[1] for z in zs_true],
            [z[2] for z in zs_true];
            label="Truth",
            lw=2,
            color=:black,
            title=res.label,
            xlabel="u",
            ylabel="v",
        )
    end

    display(p_traj)
end

# ============================================================================
# Save Results
# ============================================================================

if !isempty(results)
    output_dir = @__DIR__
    timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
    base_filename = "vdp_results_$(timestamp)"

    # Save full results to JLD2
    jld2_path = joinpath(output_dir, "$(base_filename).jld2")
    jldsave(
        jld2_path;
        results=results,
        config=Dict(
            :N_samples => N_samples,
            :N_adapt => N_adapt,
            :K => K,
            :K_y => K_y,
            :N_obs => N_obs,
            :δt => δt,
            :μ_true => μ_true,
            :SEED => SEED,
            :RUN_RHMC => RUN_RHMC,
            :RUN_MCRHMC => RUN_MCRHMC,
            :RUN_HMC_DIAGONAL => RUN_HMC_DIAGONAL,
            :RUN_HMC_DENSE => RUN_HMC_DENSE,
            :RUN_HMC_EMPIRICAL => RUN_HMC_EMPIRICAL,
        ),
        observations=ys,
        true_states=zs_true,
    )
    println("\nResults saved to: $jld2_path")

    # Save summary to text file
    txt_path = joinpath(output_dir, "$(base_filename).txt")
    open(txt_path, "w") do f
        println(f, "Van der Pol Oscillator - Sampler Comparison Results")
        println(f, "Generated: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        println(f, "\n" * "="^60)
        println(f, "CONFIGURATION")
        println(f, "="^60)
        println(f, "N_samples: $N_samples")
        println(f, "N_adapt: $N_adapt")
        println(f, "K (total latent states): $K")
        println(f, "K_y (steps between obs): $K_y")
        println(f, "N_obs: $N_obs")
        println(f, "δt: $δt")
        println(f, "μ_true: $μ_true")
        println(f, "SEED: $SEED")
        println(f, "\nMethods run:")
        println(f, "  RUN_RHMC: $RUN_RHMC")
        println(f, "  RUN_MCRHMC: $RUN_MCRHMC")
        println(f, "  RUN_HMC_DIAGONAL: $RUN_HMC_DIAGONAL")
        println(f, "  RUN_HMC_DENSE: $RUN_HMC_DENSE")
        println(f, "  RUN_HMC_EMPIRICAL: $RUN_HMC_EMPIRICAL")
        println(f, "\n" * summary_text)
    end
    println("Summary saved to: $txt_path")
end

println("\nDone!")
