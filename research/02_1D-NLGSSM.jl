using Distributions
using Random
using Plots
using LinearAlgebra
using SparseArrays

using FiniteDiff
using ProgressMeter

# Particle filtering
using StatsBase
using LogExpFunctions

####################
#### PARAMATERS ####
####################

SEED = 1728
rng = MersenneTwister(SEED)

T = 10
dt = 1.0

μ = 3.0
θ = 0.5
σ = 0.8
σ_x = sqrt(σ^2 / (2 * θ) * (1 - exp(-2 * θ * dt)))  # Warning this depends on dt

σ_y = 2.0

# Initialise with marginal
μ0 = μ
σ0 = σ_x / sqrt(2θ)

# Marginal probability of negative sample
p_neg = cdf(Normal(μ, σ_x / sqrt(2θ)), 0.0)
println("Marginal probability of negative sample: $p_neg")

##########################
#### MODEL DEFINITION ####
##########################

# OU Dynamics
f(x, t) = μ + (x - μ) * exp(-θ * t)
df(x, t) = exp(-θ * t)
d2f(x, t) = 0.0

# Quadratic measurement model
h(x) = x^2
dh(x) = 2 * x
d2h(x) = 2.0

####################
#### SIMULATION ####
####################

xs = Vector{Float64}(undef, T)
ȳs = Vector{Float64}(undef, T)  # unnoised measurements
ys = Vector{Float64}(undef, T)

for t in 1:T
    if t == 1
        xs[t] = μ0 + σ0 * randn(rng)
    else
        xs[t] = f(xs[t - 1], dt) + σ_x * randn(rng)
    end

    ȳs[t] = h(xs[t])
    ys[t] = ȳs[t] + σ_y * randn(rng)
end

# Test around a slightly perterbed trajectory
x̂s = xs .+ σ0 / 10.0 * randn(rng, T)

# Plot trajectory and measurements
p_traj = scatter(1:T, xs; label="State Trajectory", xlabel="Time", ylabel="State")
scatter!(p_traj, 1:T, x̂s; label="Perturbed Trajectory")
p_meas = scatter(
    1:T, ȳs; label="Projected Trajectory", xlabel="Time", ylabel="Measurement"
)
scatter!(p_meas, 1:T, ys; label="Noised Measurements")
display(plot(p_traj, p_meas; layout=(2, 1), size=(800, 600)))

####################
#### LIKELIHOOD ####
####################

function likelihood(xs, ys)
    T = length(xs)
    log_likelihood = 0.0

    # Prior
    log_likelihood += logpdf(Normal(μ0, σ0), xs[1])
    for t in 2:T
        log_likelihood += logpdf(Normal(f(xs[t - 1], dt), σ_x), xs[t])
    end

    # Likelihood
    for t in 1:T
        log_likelihood += logpdf(Normal(h(xs[t]), σ_y), ys[t])
    end

    return log_likelihood
end

likelihood(x̂s, ys)

#################
#### HESSIAN ####
#################

# Compute Hessian numerically
H = FiniteDiff.finite_difference_hessian(xs -> likelihood(xs, ys), x̂s)

# Plot sparsity structure
println("Hessian sparsity structure:")
display(abs.(H) .> 1e-6)
println()

# Check definiteness
is_psd = isposdef(H)
println("Hessian is positive semi-definite: $is_psd \n")

######################
#### GAUSS-NEWTON ####
######################

# A Gauss-Newton approximation of the Hessian 

Λ_gn = zeros(T, T)

# Add diagonal terms
for t in 1:T
    λ = 0.0

    # Base term
    if t == 1
        λ += 1 / σ0^2
    else
        λ += 1 / σ_x^2
    end

    # Observation term
    λ += dh(x̂s[t])^2 / σ_y^2

    # Transition term
    if t < T
        λ += df(x̂s[t], dt)^2 / σ_x^2
    end

    Λ_gn[t, t] = λ
end

# Add secondary terms
for t in 2:T
    λ = -df(x̂s[t - 1], dt) / σ_x^2
    Λ_gn[t, t - 1] = λ
    Λ_gn[t - 1, t] = λ
end

# Correct on off-diagonal but off on diagonal due to missing Hessian tensor term
gn_error_mat = H - (-Λ_gn)
gn_error_mat[abs.(gn_error_mat) .< 1e-6] .= 0.0
println("Gauss-Newton error matrix:")
display(sparse(gn_error_mat))
println()

# Think the result of this approximation is that the matrix will always be PSD
is_psd_gn = isposdef(Λ_gn)
println("Gauss-Newton approximation is positive semi-definite: $is_psd_gn \n")

#######################
#### EXACT HESSIAN ####
#######################

# Easy to compute exact Hessian tensor in 1D case is just the second derivative
Λ_exact = copy(Λ_gn)
for t in 1:T
    # Add observation term
    y_res = ys[t] - h(x̂s[t])
    Λ_exact[t, t] -= y_res * d2h(x̂s[t]) / σ_y^2

    # Add dynamics term — actually zero for this case
    if t < T
        x_res = x̂s[t + 1] - f(x̂s[t], dt)
        Λ_exact[t, t] -= x_res * d2f(x̂s[t], dt) / σ_x^2
    end
end

exact_error_mat = H - (-Λ_exact)
exact_error_mat[abs.(exact_error_mat) .< 1e-6] .= 0.0
println("Exact Hessian error matrix:")
display(sparse(exact_error_mat))
println()

###############
#### NOTES ####
###############

# The Gauss-Newton approximation can be motivated for the observation terms since this is
# equivalent to marginalising over the observations to obtain the Fisher information matrix.
# For the dynamics however, no motivation is given so we have to view it as a pure
# approximation or a method for obtaining a PSD matrix.

# The alternative would be to compute the full Hessian (should be tractable if a bit slower
# since O(D^3)) but then we'd need to use the SoftAbs metric to ensure the matrix is PSD.

# Note that the observations do not affect the curvature when using the Gauss-Newton
# approximation. It might be worth contemplating why that is the case.

###########################
#### METRIC DERIVATIVE ####
###########################

struct ModelParams
    T::Int
    dt::Float64
    σ_x::Float64
    σ_y::Float64
    σ0::Float64
end

# The stochastic volatity example in the paper has G = I + C^{-1} where C^{-1} is the
# precison matrix so it seems correct to set G to be Λ

function calc_G(xs, params)
    # Unpack parameters
    T, dt, σ_x, σ_y, σ0 = params.T, params.dt, params.σ_x, params.σ_y, params.σ0

    # Compute diagonal
    dv = Vector{Float64}(undef, T)
    for t in 1:T
        v = t == 1 ? 1 / σ0^2 : 1 / σ_x^2
        v += dh(xs[t])^2 / σ_y^2
        if t < T
            v += df(xs[t], dt)^2 / σ_x^2
        end
        dv[t] = v
    end

    # Compute off-diagonal
    ev = Vector{Float64}(undef, T - 1)
    for t in 2:T
        ev[t - 1] = -df(xs[t - 1], dt) / σ_x^2
    end

    return SymTridiagonal(dv, ev)
end

params = ModelParams(T, dt, σ_x, σ_y, σ0)
G_test = calc_G(x̂s, params)

# Derivative of G w.r.t tth state
# Currently storing as tridiagonal even though it is far more sparse
function calc_dG(xs, t, params)
    # Unpack parameters
    T, dt, σ_x, σ_y = params.T, params.dt, params.σ_x, params.σ_y

    # Compute diagonal
    dv = zeros(T)
    v = 2 * dh(xs[t]) * d2h(xs[t]) / σ_y^2
    if t < T
        v += 2 * df(xs[t], dt) * d2f(xs[t], dt) / σ_x^2
    end
    dv[t] = v

    ev = zeros(T - 1)
    if t > 1
        ev[t - 1] = -d2f(xs[t - 1], dt) / σ_x^2
    end

    return SymTridiagonal(dv, ev)
end

# Compare to numerical derivative
t_test = 1
ϵ = 1e-8
x̂s_plus = [t == t_test ? x̂s[t] + ϵ : x̂s[t] for t in 1:T]
x̂s_minus = [t == t_test ? x̂s[t] - ϵ : x̂s[t] for t in 1:T]
G_plus = calc_G(x̂s_plus, params)
G_minus = calc_G(x̂s_minus, params)
dG_num = (G_plus - G_minus) / (2 * ϵ)
dG_test = calc_dG(x̂s, t_test, params)

println("Derivative error: $(norm(dG_num - dG_test)) \n")

##################
#### GRADIENT ####
##################

function ll_grad(xs, ys, params)
    T, dt, σ_x, σ_y, σ0 = params.T, params.dt, params.σ_x, params.σ_y, params.σ0
    grad = zeros(T)

    # Base term
    grad[1] = -(xs[1] - μ0) / σ0^2
    for t in 2:T
        x_res = xs[t] - f(xs[t - 1], dt)
        grad[t] = -x_res / σ_x^2
    end

    # Observation term
    for t in 1:T
        y_res = ys[t] - h(xs[t])
        grad[t] += y_res / σ_y^2 * dh(xs[t])
    end

    # Transition term
    for t in 2:T
        x_res = xs[t] - f(xs[t - 1], dt)
        grad[t - 1] += x_res / σ_x^2 * df(xs[t - 1], dt)
    end

    return grad
end

grad = FiniteDiff.finite_difference_gradient(xs -> likelihood(xs, ys), x̂s)
grad_test = ll_grad(x̂s, ys, params)
println("Gradient error: $(norm(grad - grad_test)) \n")

######################
#### GLF DYNAMICS ####
######################

# Compute gradient expression using inverses
p_test = randn(rng, T)
neg_dH_test_inv = (
    ll_grad(x̂s, ys, params)[t_test] - 0.5 * tr(inv(G_test) * dG_test) +
    0.5 * p_test' * inv(G_test) * dG_test * inv(G_test) * p_test
)

# Compute gradient using solves
G_chol = cholesky(G_test)
G_inv_p = G_chol \ p_test
neg_dH_test_solve = (
    ll_grad(x̂s, ys, params)[t_test] - 0.5 * tr(G_chol \ dG_test) +
    0.5 * G_inv_p' * dG_test * G_inv_p
)

println("GLF expression error: $(neg_dH_test_inv - neg_dH_test_solve) \n")

###########################
#### MOMENTUM SAMPLING ####
###########################

# p = G_chol.L * randn(rng, T)

################################
#### FIXED POINT INTERATION ####
################################

# NOTE: flipped this around from before, to hopefully fix sign error.
function ∇θ_H(xs, ps, ys, params)
    grad = ll_grad(xs, ys, params)
    G = calc_G(xs, params)
    G_chol = cholesky(G)
    ∇θ = Vector{Float64}(undef, length(xs))
    for t in 1:length(xs)
        dG_t = calc_dG(xs, t, params)
        G_inv_p = G_chol \ ps
        ∇θ[t] = (-grad[t] + 0.5 * tr(G_chol \ dG_t) - 0.5 * G_inv_p' * dG_t * G_inv_p)
    end
    return ∇θ
end

function ∇p_H(xs, ps, params)
    G = calc_G(xs, params)
    G_chol = cholesky(G)
    ∇p = G_chol \ ps
    return ∇p
end

function calc_hamiltonian(xs, ps, ys, params)
    G = calc_G(xs, params)
    G_chol = cholesky(G)
    hamiltonian = (
        -likelihood(xs, ys) +
        0.5 * (params.T * log(2π) + logdet(G_chol)) +
        0.5 * ps' * (G_chol \ ps)
    )
    return hamiltonian
end

ϵ = 0.1
fp_tol = 1e-8
max_reps = 100

# Fixed point iteration for momentum sampling
ps_old = G_chol.L * randn(rng, T)
ps_new_1 = copy(ps_old)
reps = 0
while reps < max_reps
    reps += 1
    ps_new_2 = ps_old - ϵ / 2 * ∇θ_H(x̂s, ps_new_1, ys, params)
    if norm(ps_new_1 - ps_new_2) < fp_tol
        break
    end
    ps_new_1 = ps_new_2
end
println("Fixed point iteration converged in $reps repetitions.\n")

# Fixed point iteration for state sampling
xs_old = copy(x̂s)
xs_new_1 = copy(xs_old)
reps = 0
while reps < max_reps
    reps += 1
    xs_new_2 =
        xs_old + ϵ / 2 * (∇p_H(xs_new_1, ps_old, params) + ∇p_H(xs_old, ps_old, params))
    if norm(xs_new_1 - xs_new_2) < fp_tol
        break
    end
    xs_new_1 = xs_new_2
end
println("Fixed point iteration converged in $reps repetitions.\n")

##############
#### RHMC ####
##############

# Sanity check — overwrite G with identity
# calc_G(xs, params) = Diagonal(ones(params.T))
# calc_dG(xs, t, params) = Diagonal(zeros(params.T))

N_samples = 300000
N_burnin = 10000
x_samples = Vector{Vector{Float64}}(undef, N_samples)
n_steps = 3

struct LFParams
    ϵ::Float64
    fp_tol::Float64
    max_reps::Int
end

lf_params = LFParams(ϵ, fp_tol, max_reps)

# Rewrite the above algorithm cleanly
function p_step(ps, xs, ys, params, lf_params)
    ps_new_1 = copy(ps)
    ps_new_2 = Vector{Float64}(undef, length(ps))
    reps = 0
    while reps < lf_params.max_reps
        reps += 1
        ps_new_2 = ps - lf_params.ϵ / 2 * ∇θ_H(xs, ps_new_1, ys, params)
        if norm(ps_new_1 - ps_new_2) < lf_params.fp_tol
            break
        end
        ps_new_1 = ps_new_2
    end
    if reps == lf_params.max_reps
        println("Warning: failed to converge in $(lf_params.max_reps) repetitions.")
    end
    return ps_new_2
end

function θ_step(xs, ps, lf_params)
    xs_new_1 = copy(xs)
    xs_new_2 = Vector{Float64}(undef, length(xs))
    reps = 0
    while reps < lf_params.max_reps
        reps += 1
        xs_new_2 =
            xs + lf_params.ϵ / 2 * (∇p_H(xs_new_1, ps, params) + ∇p_H(xs, ps, params))
        if norm(xs_new_1 - xs_new_2) < lf_params.fp_tol
            break
        end
        xs_new_1 = xs_new_2
    end
    if reps == lf_params.max_reps
        println("Warning: failed to converge in $(lf_params.max_reps) repetitions.")
    end
    return xs_new_2
end

function glf_step(xs, ps, ys, params, lf_params)
    # Half step for momentum
    ps = p_step(ps, xs, ys, params, lf_params)
    # Full step for state
    xs = θ_step(xs, ps, lf_params)
    # Half step for momentum again
    ps = p_step(ps, xs, ys, params, lf_params)
    return xs, ps
end

xs_curr = copy(x̂s)
n_accept = 0
@showprogress desc = "Running RHMC" for i in 1:N_samples
    # Resample momentum
    G = calc_G(xs_curr, params)
    G_chol = cholesky(G)
    ps_curr = G_chol.L * randn(rng, T)
    xs_new, ps_new = copy(xs_curr), copy(ps_curr)
    # Perform GLF steps
    for j in 1:n_steps
        xs_new, ps_new = glf_step(xs_new, ps_new, ys, params, lf_params)
    end
    # Accept or reject
    H_curr = calc_hamiltonian(xs_curr, ps_curr, ys, params)
    H_new = calc_hamiltonian(xs_new, ps_new, ys, params)
    if exp(H_curr - H_new) > rand(rng)
        xs_curr = xs_new
        n_accept += 1
    end

    # Store sample
    x_samples[i] = copy(xs_curr)
end
println("Acceptance rate: $(n_accept / N_samples) \n")

#########################
#### PARTICLE FILTER ####
#########################

# Think there is a bug with the initial state
# N_particles = 10^7
# states_history = Vector{Vector{Float64}}(undef, T)
# logweights_history = Vector{Vector{Float64}}(undef, T)

# states = [μ0 + σ0 * randn(rng) for _ in 1:N_particles]
# logweights = [logpdf(Normal(h(states[i]), σ_y), ys[1]) for i in 1:N_particles]
# states_history[1] = deepcopy(states)
# logweights_history[1] = deepcopy(logweights)

# @showprogress desc = "Running particle filter" for t in 2:T
#     # Perform resampling
#     weights = softmax(logweights)
#     idxs = sample(1:N_particles, Weights(weights), N_particles)
#     states = states[idxs]

#     # Propagate states
#     for i in 1:N_particles
#         states[i] = f(states[i], dt) + σ_x * randn(rng)
#         logweights[i] = logpdf(Normal(h(states[i]), σ_y), ys[t])
#     end

#     # Save state
#     states_history[t] = deepcopy(states)
#     logweights_history[t] = deepcopy(logweights)
# end

#########################
#### NUTS COMPARISON ####
#########################

using AdvancedHMC, AbstractMCMC
using LogDensityProblems, LogDensityProblemsAD, ADTypes # For defining the target distribution & its gradient
using ForwardDiff # An example AD backend
using Random # For initial parameters
using DistributionsAD

struct LogTargetDensity
    dim::Int
    ys::Vector{Float64}
end
LogDensityProblems.logdensity(p::LogTargetDensity, θ) = likelihood(θ, p.ys)
LogDensityProblems.dimension(p::LogTargetDensity) = p.dim
function LogDensityProblems.capabilities(::Type{LogTargetDensity})
    return LogDensityProblems.LogDensityOrder{0}()
end

# 2. Wrap the log density function and specify the AD backend.
#    This creates a callable struct that computes the log density and its gradient.
ℓπ = LogTargetDensity(T, ys)
model = AdvancedHMC.LogDensityModel(LogDensityProblemsAD.ADgradient(AutoForwardDiff(), ℓπ))

sampler = NUTS(0.8)
n_adapts, n_samples = 2000, 100000
initial_θ = randn(T)

samples = AbstractMCMC.sample(
    Random.default_rng(),
    model,
    sampler,
    n_adapts + n_samples;
    n_adapts=n_adapts,
    initial_params=initial_θ,
    progress=true, # Optional: Show a progress bar
);

nuts_states = [s.z.θ for s in samples]
nuts_means = mean(nuts_states)

# Can maybe believe these are correct
println("NUTS errors:")
println(mean(x_samples) - nuts_means)
println("Standard error: $(std(x_samples) * 1.96 ./ sqrt(length(x_samples)))")

# Perhaps try hypothesis testing
using HypothesisTests

p_values = Vector{Float64}(undef, T)
for t in 1:T
    test = UnequalVarianceTTest(getindex.(x_samples, t), getindex.(nuts_states, t))
    p_values[t] = pvalue(test)
end
# Why do these decay through time?
println("P-values for hypothesis test:")
println(p_values)

###############
#### PLOTS ####
###############

# Reproject measurements into state space
f_inv(y) = sqrt(y)  # assume x is non-negative
xs_recon = f_inv.(ys)

# # Plot the samples
p_samples = plot(; size=(800, 600))
scatter!(p_samples, 1:T, xs; label="State Trajectory", xlabel="Time", ylabel="State")
scatter!(p_samples, 1:T, xs_recon; label="Reconstructed Trajectory")

scatter!(
    p_samples, 1:T, mean(x_samples[(N_burnin + 1):end]); label="Mean Sampled Trajectory"
)

scatter!(p_samples, 1:T, nuts_means; label="NUTS means")

# display(p_samples)
