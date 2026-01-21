"""
Test Kleppe's observed Hessian metric for RMHMC on Van der Pol oscillator
with a QUADRATIC observation function h(z) = z₁² - z₂².

This introduces negative curvature in the observation Hessian, which tests
the modified Cholesky regularization more thoroughly.

Compares the ObservedHessianMetric (using modified Cholesky on the observed Hessian)
against the existing BorderedBlockTridiagonalRiemannianMetric (Fisher/GGN approximation).
"""

using Distributions
using LinearAlgebra
using Random
using Plots
using StaticArrays
using SparseArrays

using AbstractMCMC
using AdvancedHMC
using AdvancedHMC: Hamiltonian, DualValue
using LogDensityProblems
using MCMCDiagnosticTools

using RiemannianSSMs
using RiemannianSSMs:
    MCRHMC, to_bordered_block_vector, from_bordered_block_vector, from_block_vector

# ============================================================================
# Configuration
# ============================================================================

SEED = 42
rng = MersenneTwister(SEED)

# Model dimensions
D = 2      # State dimension (u, v)
P = 1      # Parameter dimension (log μ)

# Time discretization
δt = 0.1           # Euler step size
K_y = 10           # Latent steps between observations
N_obs = 10         # Number of observations
K = N_obs * K_y    # Total number of latent states (200)

# Observation indices (1-indexed, every K_y-th state starting from K_y)
obs_indices = collect(K_y:K_y:K)

# Observation noise (scaled proportionally since h ranges ±4 vs ±2)
σ_obs = 0.3

# ============================================================================
# Quadratic Observation Functions (h(z) = z₁² - z₂²)
# ============================================================================

"""
Quadratic difference observation: h(z) = z₁² - z₂²
Returns a 1-element SVector for scalar observation.
"""
function h_quadratic(z::SVector{2,T}) where {T}
    return @SVector [z[1]^2 - z[2]^2]
end

"""
Jacobian ∂h/∂z (1×2 matrix).
Jh = [2z₁, -2z₂]
"""
function calc_Jh_quadratic(z::SVector{2,T}) where {T}
    return @SMatrix [2*z[1] -2*z[2]]
end

"""
Derivative-of-Jacobian structure: ∂Jh/∂z_d (D matrices, each 1×D).
Hh₁ = ∂Jh/∂z₁ = [2, 0]
Hh₂ = ∂Jh/∂z₂ = [0, -2]
"""
function calc_Hhs_quadratic(::Type{T}) where {T}
    Hh1 = @SMatrix [T(2) T(0)]
    Hh2 = @SMatrix [T(0) T(-2)]
    return (Hh1, Hh2)
end

"""
Component Hessian: ∇²h (D×D matrix).
For scalar observation (m=1), this is the single Hessian of h w.r.t. z.
∇²h = [[2, 0], [0, -2]]  -- introduces negative curvature!
"""
function calc_obs_hessian_quadratic(::Type{T}) where {T}
    return @SMatrix [T(2) T(0); T(0) T(-2)]
end

calc_R_quadratic(σ) = Diagonal(@SVector([σ^2]))
calc_Rinv_quadratic(σ) = Diagonal(@SVector([1.0 / σ^2]))

# ============================================================================
# Ground truth simulation
# ============================================================================

# Prior for initial state
μ0 = @SVector [1.0, 0.0]  # Start near the limit cycle
Σ0 = Diagonal(@SVector([0.1, 0.1]))
prior = MvNormal(μ0, Σ0)

# True parameter value
μ_true = 1.0
θ_true = @SVector [log(μ_true)]

# Dynamics
σ_u = 0.1  # Small noise on u for numerical stability
σ_v = 0.1  # Main process noise on v
dyn = VanDerPolDynamicsParam(σ_u, σ_v, δt)

# Parameter prior (diffuse Gaussian on log scale)
prior_mean = @SVector [log(μ_true)]  # Centered near true value
prior_var = @SVector [4.0]           # Wide variance

function simulate_quadratic(rng::AbstractRNG, prior, dyn, θ, K::Int, obs_indices, σ_obs)
    zs = Vector{SVector{2,Float64}}(undef, K)
    ys = Vector{SVector{1,Float64}}(undef, length(obs_indices))

    obs_idx = 1
    for k in 1:K
        if k == 1
            z = SVector{2,Float64}(rand(rng, prior))
        else
            z =
                f_param(dyn, zs[k - 1], θ) +
                rand(rng, MvNormal(zeros(2), calc_Q_param(dyn)))
        end
        zs[k] = z

        # Only observe at specified indices
        if k in obs_indices
            y = h_quadratic(z) + rand(rng, MvNormal(zeros(1), calc_R_quadratic(σ_obs)))
            ys[obs_idx] = y
            obs_idx += 1
        end
    end

    return zs, ys
end

println("Simulating Van der Pol oscillator with μ = $(μ_true)...")
println("Observation function: h(z) = z₁² - z₂²")
println("Total latent states: K = $K")
println("Number of observations: $(length(obs_indices))")
println("Observations at indices: $(obs_indices[1]):$(K_y):$(obs_indices[end])")

zs_true, ys = simulate_quadratic(rng, prior, dyn, θ_true, K, obs_indices, σ_obs)
zs_true_block = BlockVector{Float64,2}(zs_true)

# Plot the trajectory
p1 = plot(;
    title="Van der Pol Oscillator (μ=$(μ_true), h=z₁²-z₂²)",
    xlabel="u",
    ylabel="v",
    legend=:topright,
    size=(800, 600),
    aspect_ratio=1,
)
plot!(
    p1, [z[1] for z in zs_true], [z[2] for z in zs_true]; label="Truth", lw=2, color=:black
)

# Show observation function values at observed points
obs_h_vals = [h_quadratic(zs_true[k])[1] for k in obs_indices]
println(
    "\nObservation function h values at observed states: min=$(minimum(obs_h_vals)), max=$(maximum(obs_h_vals))",
)

display(p1)

# ============================================================================
# Custom Log-Likelihood and Gradient for Quadratic Observation
# ============================================================================

"""
Log-likelihood for quadratic observation model.
"""
function calc_ll_quadratic(
    z::AbstractVector{T}, ys, prior, dyn, K::Int, prior_mean, prior_prec, obs_indices, σ_obs
) where {T}
    D = 2
    P = 1
    θ = @SVector [z[K * D + 1]]

    R_inv = calc_Rinv_quadratic(σ_obs)
    Q_inv = calc_Qinv_param(dyn)

    ll = zero(T)

    # Parameter prior
    ll -= 0.5 * prior_prec[1] * (θ[1] - prior_mean[1])^2

    # State prior (k=1)
    z1 = @SVector [z[1], z[2]]
    Σ0_inv = inv(prior.Σ)
    diff = z1 - SVector{2,T}(prior.μ)
    ll -= 0.5 * dot(diff, Σ0_inv * diff)

    obs_idx = 1
    @inbounds for k in 1:K
        idx = (k - 1) * D
        z_k = @SVector [z[idx + 1], z[idx + 2]]

        # Dynamics (k > 1)
        if k > 1
            z_km1 = @SVector [z[idx - D + 1], z[idx - D + 2]]
            μ = f_param(dyn, z_km1, θ)
            residual = z_k - μ
            ll -= 0.5 * dot(residual, Q_inv * residual)
        end

        # Observation (quadratic)
        if k in obs_indices
            h_val = h_quadratic(z_k)
            obs_residual = ys[obs_idx] - h_val
            ll -= 0.5 * dot(obs_residual, R_inv * obs_residual)
            obs_idx += 1
        end
    end

    return ll
end

"""
Gradient of log-likelihood for quadratic observation model.
"""
function calc_ll_grad_quadratic(
    z::AbstractVector{T}, ys, prior, dyn, K::Int, prior_mean, prior_prec, obs_indices, σ_obs
) where {T}
    D = 2
    P = 1
    n = K * D + P
    grad = zeros(T, n)
    θ = @SVector [z[K * D + 1]]

    R_inv = calc_Rinv_quadratic(σ_obs)
    Q_inv = calc_Qinv_param(dyn)

    # Parameter prior gradient
    grad[K * D + 1] = -prior_prec[1] * (θ[1] - prior_mean[1])

    # State prior gradient (k=1)
    z1 = @SVector [z[1], z[2]]
    Σ0_inv = inv(prior.Σ)
    diff = z1 - SVector{2,T}(prior.μ)
    g_prior = -Σ0_inv * diff
    grad[1] += g_prior[1]
    grad[2] += g_prior[2]

    obs_set = Set(obs_indices)
    obs_idx = 1

    @inbounds for k in 1:K
        idx = (k - 1) * D
        z_k = @SVector [z[idx + 1], z[idx + 2]]

        # Incoming dynamics gradient (k > 1)
        if k > 1
            z_km1 = @SVector [z[idx - D + 1], z[idx - D + 2]]
            μ = f_param(dyn, z_km1, θ)
            residual = z_k - μ
            g_in = -Q_inv * residual
            grad[idx + 1] += g_in[1]
            grad[idx + 2] += g_in[2]
        end

        # Outgoing dynamics gradient (k < K)
        if k < K
            z_kp1 = @SVector [z[idx + D + 1], z[idx + D + 2]]
            μ = f_param(dyn, z_k, θ)
            residual = z_kp1 - μ
            Jf = calc_Jf_param(dyn, z_k, θ)
            f_θ = calc_f_θ(dyn, z_k, θ)
            g_out = Jf' * Q_inv * residual
            grad[idx + 1] += g_out[1]
            grad[idx + 2] += g_out[2]
            # Parameter gradient from dynamics
            grad[K * D + 1] += dot(f_θ[:, 1], Q_inv * residual)
        end

        # Observation gradient (quadratic)
        if k in obs_set
            h_val = h_quadratic(z_k)
            Jh = calc_Jh_quadratic(z_k)
            obs_residual = ys[obs_idx] - h_val
            g_obs = Jh' * R_inv * obs_residual
            grad[idx + 1] += g_obs[1]
            grad[idx + 2] += g_obs[2]
            obs_idx += 1
        end
    end

    return grad
end

# ============================================================================
# Custom Observed Hessian with Quadratic Observation Correction
# ============================================================================

"""
Compute observed Hessian for quadratic observation model.
Includes the observation Hessian correction term: -R⁻¹r_obs × ∇²h
"""
function calc_observed_hessian_quadratic(
    z::AbstractVector{T}, ys, prior, dyn, K::Int, prior_prec, obs_indices, σ_obs
) where {T}
    D = 2
    P = 1
    n = K * D + P
    θ = @SVector [z[K * D + 1]]

    R_inv = calc_Rinv_quadratic(σ_obs)
    Q_inv = SMatrix{D,D,T}(calc_Qinv_param(dyn))

    I_idx = Int[]
    J_idx = Int[]
    V_val = T[]

    obs_set = Set(obs_indices)
    obs_idx_map = Dict(k => i for (i, k) in enumerate(obs_indices))

    # ----- DIAGONAL BLOCKS -----
    @inbounds for k in 1:K
        idx_base = (k - 1) * D
        z_k = @SVector [z[idx_base + 1], z[idx_base + 2]]

        H_kk = zeros(SMatrix{D,D,T})

        # Prior term (k=1) or incoming dynamics term (k>1)
        if k == 1
            H_kk += SMatrix{D,D,T}(inv(prior.Σ))
        else
            H_kk += SMatrix{D,D,T}(Q_inv)
        end

        # Observation term (quadratic)
        if k in obs_set
            Jh = calc_Jh_quadratic(z_k)
            # GGN term: Jh' R⁻¹ Jh
            H_kk += Jh' * R_inv * Jh

            # Observation Hessian correction: -R⁻¹r_obs × ∇²h
            obs_idx = obs_idx_map[k]
            h_val = h_quadratic(z_k)
            r_obs = ys[obs_idx] - h_val
            R_inv_r = (R_inv * r_obs)[1]  # scalar for m=1
            obs_hess = calc_obs_hessian_quadratic(T)  # [[2,0],[0,-2]]
            H_kk -= R_inv_r * obs_hess
        end

        # Outgoing dynamics term (k < K)
        if k < K
            z_kp1 = @SVector [z[idx_base + D + 1], z[idx_base + D + 2]]
            μ = f_param(dyn, z_k, θ)
            residual = z_kp1 - μ
            Jf = calc_Jf_param(dyn, z_k, θ)
            Hfs = calc_dyn_hessians(dyn, z_k, θ)

            # GGN term: Jf' Q⁻¹ Jf
            H_kk += Jf' * Q_inv * Jf

            # Observed Hessian correction: -Σ_d [Q⁻¹ r]_d ∇²f_d
            Q_inv_r = Q_inv * residual
            for d in 1:D
                H_kk -= Q_inv_r[d] * Hfs[d]
            end
        end

        # Add diagonal block entries
        for i in 1:D
            for j in 1:D
                push!(I_idx, idx_base + i)
                push!(J_idx, idx_base + j)
                push!(V_val, H_kk[i, j])
            end
        end

        # Off-diagonal block G_{k, k+1} (super-diagonal)
        if k < K
            Jf = calc_Jf_param(dyn, z_k, θ)
            H_k_kp1 = -Jf' * Q_inv

            for i in 1:D
                for j in 1:D
                    push!(I_idx, idx_base + i)
                    push!(J_idx, idx_base + D + j)
                    push!(V_val, H_k_kp1[i, j])
                    push!(I_idx, idx_base + D + j)
                    push!(J_idx, idx_base + i)
                    push!(V_val, H_k_kp1[i, j])
                end
            end
        end
    end

    # ----- BORDER BLOCKS (state-parameter coupling) -----
    @inbounds for k in 1:K
        idx_base = (k - 1) * D
        z_k = @SVector [z[idx_base + 1], z[idx_base + 2]]

        H_k_θ = zeros(SMatrix{D,P,T})

        # Incoming dynamics term (k > 1): -Q⁻¹ f_θ
        if k > 1
            z_km1 = @SVector [z[idx_base - D + 1], z[idx_base - D + 2]]
            f_θ = calc_f_θ(dyn, z_km1, θ)
            H_k_θ -= SMatrix{D,P,T}(Q_inv * f_θ)
        end

        # Outgoing dynamics term (k < K)
        if k < K
            z_kp1 = @SVector [z[idx_base + D + 1], z[idx_base + D + 2]]
            μ = f_param(dyn, z_k, θ)
            residual = z_kp1 - μ
            Jf = calc_Jf_param(dyn, z_k, θ)
            f_θ = calc_f_θ(dyn, z_k, θ)
            mixed_hess = calc_mixed_hessians(dyn, z_k, θ)

            # GGN term
            H_k_θ += Jf' * Q_inv * f_θ

            # Observed Hessian correction
            Q_inv_r = Q_inv * residual
            for i in 1:D
                H_k_θ -= Q_inv_r[i] * mixed_hess[i]
            end
        end

        for i in 1:D
            for p in 1:P
                if abs(H_k_θ[i, p]) > 1e-15
                    push!(I_idx, idx_base + i)
                    push!(J_idx, K * D + p)
                    push!(V_val, H_k_θ[i, p])
                    push!(I_idx, K * D + p)
                    push!(J_idx, idx_base + i)
                    push!(V_val, H_k_θ[i, p])
                end
            end
        end
    end

    # ----- CORNER BLOCK (parameter-parameter) -----
    H_θθ = zeros(SMatrix{P,P,T})

    # Prior
    H_θθ += Diagonal(SVector{P,T}(prior_prec))

    @inbounds for k in 1:(K - 1)
        idx_base = (k - 1) * D
        z_k = @SVector [z[idx_base + 1], z[idx_base + 2]]
        z_kp1 = @SVector [z[idx_base + D + 1], z[idx_base + D + 2]]

        μ = f_param(dyn, z_k, θ)
        residual = z_kp1 - μ
        f_θ = calc_f_θ(dyn, z_k, θ)

        # GGN term
        H_θθ += f_θ' * Q_inv * f_θ

        # Observed Hessian correction
        # For VDP with log-param: ∇²_θ f_d = f_θ[d, 1] (since ∂²(exp(θ)·g)/∂θ² = exp(θ)·g)
        Q_inv_r = Q_inv * residual
        for i in 1:D
            H_θθ -= Q_inv_r[i] * @SMatrix([f_θ[i, 1]])
        end
    end

    for i in 1:P
        for j in 1:P
            push!(I_idx, K * D + i)
            push!(J_idx, K * D + j)
            push!(V_val, H_θθ[i, j])
        end
    end

    return sparse(I_idx, J_idx, V_val, n, n)
end

# ============================================================================
# Custom Hessian Derivatives for Quadratic Observation
# ============================================================================

"""
Compute derivatives of the observed Hessian w.r.t. each component of z.
"""
function calc_observed_hessian_derivs_quadratic(
    z::AbstractVector{T}, ys, prior, dyn, K::Int, prior_prec, obs_indices, σ_obs
) where {T}
    D = 2
    P = 1
    n = K * D + P
    θ = @SVector [z[K * D + 1]]

    R_inv = calc_Rinv_quadratic(σ_obs)
    Q_inv = SMatrix{D,D,T}(calc_Qinv_param(dyn))

    obs_set = Set(obs_indices)
    obs_idx_map = Dict(k => i for (i, k) in enumerate(obs_indices))

    dGs = Vector{SparseMatrixCSC{T,Int}}(undef, n)

    # State derivatives
    for k in 1:K
        for d in 1:D
            idx = (k - 1) * D + d
            dGs[idx] = calc_dG_dz_kd_quadratic(
                z, ys, dyn, K, k, d, θ, Q_inv, R_inv, obs_set, obs_idx_map, σ_obs
            )
        end
    end

    # Parameter derivatives
    for p in 1:P
        dGs[K * D + p] = calc_dG_dθ_p_quadratic(z, ys, dyn, K, p, θ, Q_inv, R_inv, obs_set)
    end

    return dGs
end

"""
Compute ∂G/∂z_k^{(d)} for quadratic observation model.
"""
function calc_dG_dz_kd_quadratic(
    z::AbstractVector{T},
    ys,
    dyn,
    K::Int,
    k::Int,
    d::Int,
    θ,
    Q_inv,
    R_inv,
    obs_set,
    obs_idx_map,
    σ_obs,
) where {T}
    D = 2
    P = 1
    n = K * D + P
    I_idx = Int[]
    J_idx = Int[]
    V_val = T[]

    idx_base = (k - 1) * D
    z_k = @SVector [z[idx_base + 1], z[idx_base + 2]]

    # ===== STATE-STATE BLOCKS =====

    # ----- Effect on G_{k,k} from outgoing dynamics -----
    if k < K
        z_kp1 = @SVector [z[idx_base + D + 1], z[idx_base + D + 2]]
        μ = f_param(dyn, z_k, θ)
        residual = z_kp1 - μ
        Jf = calc_Jf_param(dyn, z_k, θ)
        Hfs = calc_Hfs_param(dyn, z_k, θ)
        dyn_hess = calc_dyn_hessians(dyn, z_k, θ)
        dyn_hess_derivs = calc_dyn_hessian_derivs(dyn, z_k, θ)

        # GGN term derivative
        Mf = Q_inv * Jf
        dH_kk = Hfs[d]' * Mf + Mf' * Hfs[d]

        # Observed correction derivative
        Q_inv_Jf_col = Q_inv * Jf[:, d]
        for i in 1:D
            dH_kk += Q_inv_Jf_col[i] * dyn_hess[i]
        end

        Q_inv_r = Q_inv * residual
        for i in 1:D
            dH_kk -= Q_inv_r[i] * dyn_hess_derivs[i][d]
        end

        for i in 1:D
            for j in 1:D
                if abs(dH_kk[i, j]) > 1e-15
                    push!(I_idx, idx_base + i)
                    push!(J_idx, idx_base + j)
                    push!(V_val, dH_kk[i, j])
                end
            end
        end

        # ----- Effect on G_{k,k+1} -----
        dH_k_kp1 = -Hfs[d]' * Q_inv

        for i in 1:D
            for j in 1:D
                if abs(dH_k_kp1[i, j]) > 1e-15
                    push!(I_idx, idx_base + i)
                    push!(J_idx, idx_base + D + j)
                    push!(V_val, dH_k_kp1[i, j])
                    push!(I_idx, idx_base + D + j)
                    push!(J_idx, idx_base + i)
                    push!(V_val, dH_k_kp1[i, j])
                end
            end
        end
    end

    # ----- Effect on G_{k,k} from observation term -----
    if k in obs_set
        obs_idx = obs_idx_map[k]
        Jh = calc_Jh_quadratic(z_k)
        Hhs = calc_Hhs_quadratic(T)

        # GGN term derivative: ∂(Jh' R⁻¹ Jh)/∂z_k^{(d)}
        R_inv_scalar = R_inv[1, 1]
        dH_obs_ggn = Hhs[d]' * R_inv_scalar * Jh + Jh' * R_inv_scalar * Hhs[d]

        # Observation Hessian correction derivative: ∂(-R_inv_r * ∇²h)/∂z_k^{(d)}
        # ∇²h is constant, so only R_inv_r changes
        # ∂r_obs/∂z_k^{(d)} = -Jh[1, d], so ∂(R_inv_r)/∂z_k^{(d)} = -R_inv[1,1] * Jh[1, d]
        obs_hess = calc_obs_hessian_quadratic(T)
        dH_obs_corr = R_inv_scalar * Jh[1, d] * obs_hess  # positive because -(-) = +

        dH_kk_obs = dH_obs_ggn + dH_obs_corr

        for i in 1:D
            for j in 1:D
                if abs(dH_kk_obs[i, j]) > 1e-15
                    push!(I_idx, idx_base + i)
                    push!(J_idx, idx_base + j)
                    push!(V_val, dH_kk_obs[i, j])
                end
            end
        end
    end

    # ----- Effect on G_{k-1,k-1} from incoming dynamics -----
    if k > 1
        idx_base_km1 = (k - 2) * D
        z_km1 = @SVector [z[idx_base_km1 + 1], z[idx_base_km1 + 2]]
        dyn_hess_km1 = calc_dyn_hessians(dyn, z_km1, θ)

        e_d = zeros(SVector{D,T})
        e_d = setindex(e_d, one(T), d)
        Q_inv_ed = Q_inv * e_d

        dH_km1_km1 = zeros(SMatrix{D,D,T})
        for i in 1:D
            dH_km1_km1 -= Q_inv_ed[i] * dyn_hess_km1[i]
        end

        for i in 1:D
            for j in 1:D
                if abs(dH_km1_km1[i, j]) > 1e-15
                    push!(I_idx, idx_base_km1 + i)
                    push!(J_idx, idx_base_km1 + j)
                    push!(V_val, dH_km1_km1[i, j])
                end
            end
        end
    end

    # ===== BORDER BLOCKS (state-parameter) =====

    # ----- Effect on G_{k-1,θ} from outgoing dynamics at k-1 -----
    if k > 1
        idx_base_km1 = (k - 2) * D
        z_km1 = @SVector [z[idx_base_km1 + 1], z[idx_base_km1 + 2]]
        mixed_hess_km1 = calc_mixed_hessians(dyn, z_km1, θ)

        e_d = zeros(SVector{D,T})
        e_d = setindex(e_d, one(T), d)
        Q_inv_ed = Q_inv * e_d

        dH_km1_θ = zeros(SMatrix{D,P,T})
        for i in 1:D
            dH_km1_θ -= Q_inv_ed[i] * mixed_hess_km1[i]
        end

        for i in 1:D
            for p in 1:P
                if abs(dH_km1_θ[i, p]) > 1e-15
                    push!(I_idx, idx_base_km1 + i)
                    push!(J_idx, K * D + p)
                    push!(V_val, dH_km1_θ[i, p])
                    push!(I_idx, K * D + p)
                    push!(J_idx, idx_base_km1 + i)
                    push!(V_val, dH_km1_θ[i, p])
                end
            end
        end
    end

    # ----- Effect on G_{k,θ} from outgoing dynamics at k -----
    if k < K
        z_kp1 = @SVector [z[idx_base + D + 1], z[idx_base + D + 2]]
        μ = f_param(dyn, z_k, θ)
        residual = z_kp1 - μ
        Jf = calc_Jf_param(dyn, z_k, θ)
        f_θ = calc_f_θ(dyn, z_k, θ)
        Hfs = calc_Hfs_param(dyn, z_k, θ)
        Hf_θs = calc_Hf_θs(dyn, z_k, θ)
        mixed_hess = calc_mixed_hessians(dyn, z_k, θ)
        mixed_hess_derivs = calc_mixed_hessian_derivs(dyn, z_k, θ)

        # GGN term derivative
        dH_k_θ = Hfs[d]' * Q_inv * f_θ + Jf' * Q_inv * Hf_θs[d]

        # Observed correction derivative
        Q_inv_Jf_col = Q_inv * Jf[:, d]
        for i in 1:D
            dH_k_θ += Q_inv_Jf_col[i] * mixed_hess[i]
        end

        Q_inv_r = Q_inv * residual
        for i in 1:D
            dH_k_θ -= Q_inv_r[i] * mixed_hess_derivs[i][d]
        end

        for i in 1:D
            for p in 1:P
                if abs(dH_k_θ[i, p]) > 1e-15
                    push!(I_idx, idx_base + i)
                    push!(J_idx, K * D + p)
                    push!(V_val, dH_k_θ[i, p])
                    push!(I_idx, K * D + p)
                    push!(J_idx, idx_base + i)
                    push!(V_val, dH_k_θ[i, p])
                end
            end
        end
    end

    # ===== CORNER BLOCK (parameter-parameter) =====

    # ----- Effect on G_{θ,θ} from outgoing dynamics at k -----
    if k < K
        z_kp1 = @SVector [z[idx_base + D + 1], z[idx_base + D + 2]]
        μ = f_param(dyn, z_k, θ)
        residual = z_kp1 - μ
        Jf = calc_Jf_param(dyn, z_k, θ)
        f_θ = calc_f_θ(dyn, z_k, θ)
        Hf_θs = calc_Hf_θs(dyn, z_k, θ)

        # GGN term derivative
        dH_θθ = Hf_θs[d]' * Q_inv * f_θ + f_θ' * Q_inv * Hf_θs[d]

        # Observed correction derivative
        # ∂[(Q⁻¹r)_i · ∇²_θ f_i]/∂z_d = (Q⁻¹ Jf[:,d])_i · ∇²_θ f_i
        # For VDP: ∇²_θ f_i = f_θ[i, 1]
        Q_inv_Jf_col = Q_inv * Jf[:, d]
        for i in 1:D
            dH_θθ += Q_inv_Jf_col[i] * @SMatrix([f_θ[i, 1]])
        end

        for i in 1:P
            for j in 1:P
                if abs(dH_θθ[i, j]) > 1e-15
                    push!(I_idx, K * D + i)
                    push!(J_idx, K * D + j)
                    push!(V_val, dH_θθ[i, j])
                end
            end
        end
    end

    # ----- Effect on G_{θ,θ} from incoming dynamics at k -----
    if k > 1
        idx_base_km1 = (k - 2) * D
        z_km1 = @SVector [z[idx_base_km1 + 1], z[idx_base_km1 + 2]]
        f_θ_km1 = calc_f_θ(dyn, z_km1, θ)

        e_d = zeros(SVector{D,T})
        e_d = setindex(e_d, one(T), d)
        Q_inv_ed = Q_inv * e_d

        # For VDP: ∇²_θ f_i = f_θ[i, 1]
        dH_θθ = zeros(SMatrix{P,P,T})
        for i in 1:D
            dH_θθ -= Q_inv_ed[i] * @SMatrix([f_θ_km1[i, 1]])
        end

        for i in 1:P
            for j in 1:P
                if abs(dH_θθ[i, j]) > 1e-15
                    push!(I_idx, K * D + i)
                    push!(J_idx, K * D + j)
                    push!(V_val, dH_θθ[i, j])
                end
            end
        end
    end

    return sparse(I_idx, J_idx, V_val, n, n)
end

"""
Compute ∂G/∂θ^{(p)} for quadratic observation model.
"""
function calc_dG_dθ_p_quadratic(
    z::AbstractVector{T}, ys, dyn, K::Int, p::Int, θ, Q_inv, R_inv, obs_set
) where {T}
    D = 2
    P = 1
    n = K * D + P
    I_idx = Int[]
    J_idx = Int[]
    V_val = T[]

    @inbounds for k in 1:(K - 1)
        idx_base = (k - 1) * D
        z_k = @SVector [z[idx_base + 1], z[idx_base + 2]]
        z_kp1 = @SVector [z[idx_base + D + 1], z[idx_base + D + 2]]

        μ = f_param(dyn, z_k, θ)
        residual = z_kp1 - μ
        Jf = calc_Jf_param(dyn, z_k, θ)
        f_θ = calc_f_θ(dyn, z_k, θ)
        Jf_θ = calc_Jf_θ(dyn, z_k, θ)
        Hf_θs = calc_Hf_θs(dyn, z_k, θ)
        dyn_hess = calc_dyn_hessians(dyn, z_k, θ)

        # ----- Effect on G_{k,k} -----
        # GGN: ∂(Jf' Q⁻¹ Jf)/∂θ = Jf_θ[p]' Q⁻¹ Jf + Jf' Q⁻¹ Jf_θ[p]
        Mf = Q_inv * Jf
        dH_kk = Jf_θ[p]' * Mf + Mf' * Jf_θ[p]

        # Observed correction: derivative of -Σ_i (Q⁻¹r)_i ∇²f_i
        Q_inv_f_θ = Q_inv * f_θ[:, p]
        for i in 1:D
            dH_kk += Q_inv_f_θ[i] * dyn_hess[i]  # from ∂r/∂θ = -f_θ
        end

        # For VDP: ∂(∇²f_i)/∂θ = dyn_hess[i] (since exp(θ)·g differentiates to itself)
        Q_inv_r = Q_inv * residual
        for i in 1:D
            dH_kk -= Q_inv_r[i] * dyn_hess[i]
        end

        for i in 1:D
            for j in 1:D
                if abs(dH_kk[i, j]) > 1e-15
                    push!(I_idx, idx_base + i)
                    push!(J_idx, idx_base + j)
                    push!(V_val, dH_kk[i, j])
                end
            end
        end

        # ----- Effect on G_{k,k+1} -----
        dH_k_kp1 = -Jf_θ[p]' * Q_inv

        for i in 1:D
            for j in 1:D
                if abs(dH_k_kp1[i, j]) > 1e-15
                    push!(I_idx, idx_base + i)
                    push!(J_idx, idx_base + D + j)
                    push!(V_val, dH_k_kp1[i, j])
                    push!(I_idx, idx_base + D + j)
                    push!(J_idx, idx_base + i)
                    push!(V_val, dH_k_kp1[i, j])
                end
            end
        end

        # ----- Effect on G_{k+1,θ} -----
        # Incoming dynamics at k+1: -Q⁻¹ f_θ|_k
        # ∂(-Q⁻¹ f_θ)/∂θ = -Q⁻¹ Hf_θs[p]
        dH_kp1_θ = -Q_inv * Hf_θs[p]

        for i in 1:D
            for q in 1:P
                if abs(dH_kp1_θ[i, q]) > 1e-15
                    push!(I_idx, idx_base + D + i)
                    push!(J_idx, K * D + q)
                    push!(V_val, dH_kp1_θ[i, q])
                    push!(I_idx, K * D + q)
                    push!(J_idx, idx_base + D + i)
                    push!(V_val, dH_kp1_θ[i, q])
                end
            end
        end

        # ----- Effect on G_{k,θ} -----
        # GGN: ∂(Jf' Q⁻¹ f_θ)/∂θ
        dH_k_θ = Jf_θ[p]' * Q_inv * f_θ + Jf' * Q_inv * Hf_θs[p]

        # Observed correction derivative
        mixed_hess = calc_mixed_hessians(dyn, z_k, θ)
        for i in 1:D
            dH_k_θ += Q_inv_f_θ[i] * mixed_hess[i]
        end

        # For VDP: ∂(mixed_hess_i)/∂θ = mixed_hess[i] (since exp(θ)·g differentiates to itself)
        for i in 1:D
            dH_k_θ -= Q_inv_r[i] * mixed_hess[i]
        end

        for i in 1:D
            for q in 1:P
                if abs(dH_k_θ[i, q]) > 1e-15
                    push!(I_idx, idx_base + i)
                    push!(J_idx, K * D + q)
                    push!(V_val, dH_k_θ[i, q])
                    push!(I_idx, K * D + q)
                    push!(J_idx, idx_base + i)
                    push!(V_val, dH_k_θ[i, q])
                end
            end
        end

        # ----- Effect on G_{θ,θ} -----
        dH_θθ = Hf_θs[p]' * Q_inv * f_θ + f_θ' * Q_inv * Hf_θs[p]

        # For VDP: ∇²_θ f_i = f_θ[i, 1] (scalar as 1×1 matrix)
        for i in 1:D
            dH_θθ += Q_inv_f_θ[i] * @SMatrix([f_θ[i, 1]])
        end

        for i in 1:D
            dH_θθ -= Q_inv_r[i] * @SMatrix([f_θ[i, 1]])
        end

        for i in 1:P
            for j in 1:P
                if abs(dH_θθ[i, j]) > 1e-15
                    push!(I_idx, K * D + i)
                    push!(J_idx, K * D + j)
                    push!(V_val, dH_θθ[i, j])
                end
            end
        end
    end

    return sparse(I_idx, J_idx, V_val, n, n)
end

# ============================================================================
# Custom Metric Type
# ============================================================================

mutable struct QuadraticObservedHessianMetric{T} <: AdvancedHMC.AbstractRiemannianMetric
    prior::MvNormal
    dyn::VanDerPolDynamicsParam{T}
    ys::Vector{SVector{1,T}}
    obs_indices::Vector{Int}
    K::Int
    prior_mean::Vector{T}
    prior_prec::Vector{T}
    u::Vector{T}
    σ_obs::T
    symbolic::Any
    last_factorization::Union{Nothing,MCRHMC.SparseLDLFactor{T,Int}}
end

function QuadraticObservedHessianMetric(
    prior::MvNormal,
    dyn::VanDerPolDynamicsParam{T},
    ys,
    K::Int,
    prior_mean,
    prior_var,
    obs_indices::Vector{Int},
    u::Vector{T},
    σ_obs::T,
) where {T}
    D = 2
    P = 1
    prior_prec = 1 ./ prior_var
    n = K * D + P
    @assert length(u) == n "Regularization vector u must have length $n, got $(length(u))"
    return QuadraticObservedHessianMetric{T}(
        prior,
        dyn,
        ys,
        obs_indices,
        K,
        collect(prior_mean),
        collect(prior_prec),
        u,
        σ_obs,
        nothing,
        nothing,
    )
end

Base.size(m::QuadraticObservedHessianMetric) = (m.K * 2 + 1,)
Base.size(m::QuadraticObservedHessianMetric, ::Int) = m.K * 2 + 1
Base.eltype(::QuadraticObservedHessianMetric{T}) where {T} = T

function Base.show(io::IO, m::QuadraticObservedHessianMetric)
    return print(io, "QuadraticObservedHessianMetric(K=$(m.K))")
end

# ============================================================================
# AdvancedHMC Interface Implementation
# ============================================================================

function AdvancedHMC.rand_momentum(
    rng::Union{AbstractRNG,AbstractVector{<:AbstractRNG}},
    metric::QuadraticObservedHessianMetric{T},
    kinetic,
    z::AbstractVecOrMat,
) where {T}
    n = size(metric, 1)

    # Compute observed Hessian
    G = calc_observed_hessian_quadratic(
        z,
        metric.ys,
        metric.prior,
        metric.dyn,
        metric.K,
        metric.prior_prec,
        metric.obs_indices,
        metric.σ_obs,
    )

    # Modified Cholesky factorization
    F = MCRHMC.modified_cholesky(G, metric.u, 0)
    metric.last_factorization = F

    # Sample r ~ N(0, G) via r = L * sqrt(D) * Z where Z ~ N(0, I)
    Z = randn(rng, n)
    L = MCRHMC.get_L(F)
    D_diag = MCRHMC.get_D(F)

    r = L * (sqrt.(D_diag.diag) .* Z)

    return r
end

function AdvancedHMC.neg_energy(
    h::Hamiltonian{<:QuadraticObservedHessianMetric{T}}, r::V, z::V
) where {V<:AbstractVecOrMat,T}
    metric = h.metric
    n = size(metric, 1)

    # Compute observed Hessian
    G = calc_observed_hessian_quadratic(
        z,
        metric.ys,
        metric.prior,
        metric.dyn,
        metric.K,
        metric.prior_prec,
        metric.obs_indices,
        metric.σ_obs,
    )

    # Modified Cholesky factorization
    F = MCRHMC.modified_cholesky(G, metric.u, 0)
    metric.last_factorization = F

    # log|G| = sum(log(D_diag))
    logdetG = MCRHMC.logdet(F)
    logZ = 0.5 * (n * log(2π) + logdetG)

    # r' G^{-1} r via solve
    G_inv_r = F \ collect(r)
    rTG_inv_r = dot(r, G_inv_r)

    return -logZ - rTG_inv_r / 2
end

function AdvancedHMC.∂H∂θ(
    h::Hamiltonian{<:QuadraticObservedHessianMetric{T}},
    z::AbstractVecOrMat{T},
    r::AbstractVecOrMat{T},
) where {T}
    metric = h.metric
    n = size(metric, 1)

    # Compute log-likelihood and gradient
    ℓπ = calc_ll_quadratic(
        z,
        metric.ys,
        metric.prior,
        metric.dyn,
        metric.K,
        metric.prior_mean,
        metric.prior_prec,
        metric.obs_indices,
        metric.σ_obs,
    )
    ∂ℓπ∂z = calc_ll_grad_quadratic(
        z,
        metric.ys,
        metric.prior,
        metric.dyn,
        metric.K,
        metric.prior_mean,
        metric.prior_prec,
        metric.obs_indices,
        metric.σ_obs,
    )

    # Compute observed Hessian
    G = calc_observed_hessian_quadratic(
        z,
        metric.ys,
        metric.prior,
        metric.dyn,
        metric.K,
        metric.prior_prec,
        metric.obs_indices,
        metric.σ_obs,
    )

    # Modified Cholesky factorization
    F = MCRHMC.modified_cholesky(G, metric.u, 0)
    metric.last_factorization = F

    # Compute v = G^{-1} r (needed for quadform pullback)
    v = F \ collect(r)

    # Compute ∂H/∂G using MCRHMC's hamiltonian_grad
    H_grad = MCRHMC.hamiltonian_grad(F, G, v)

    # Compute derivatives of G w.r.t. each component
    dGs = calc_observed_hessian_derivs_quadratic(
        z,
        metric.ys,
        metric.prior,
        metric.dyn,
        metric.K,
        metric.prior_prec,
        metric.obs_indices,
        metric.σ_obs,
    )

    # Compute full gradient: ∂H/∂z_i = -∂ℓπ/∂z_i + trace(H_grad' × ∂G/∂z_i)
    grad = Vector{T}(undef, n)
    @inbounds for i in 1:n
        # Potential energy gradient
        grad[i] = -∂ℓπ∂z[i]

        # Metric tensor gradient via chain rule
        grad[i] += MCRHMC.sparse_trace_symmetric(H_grad, dGs[i])
    end

    return DualValue(ℓπ, grad)
end

function AdvancedHMC.∂H∂r(
    h::Hamiltonian{<:QuadraticObservedHessianMetric{T}},
    z::AbstractVecOrMat,
    r::AbstractVecOrMat,
) where {T}
    metric = h.metric

    # Compute observed Hessian
    G = calc_observed_hessian_quadratic(
        z,
        metric.ys,
        metric.prior,
        metric.dyn,
        metric.K,
        metric.prior_prec,
        metric.obs_indices,
        metric.σ_obs,
    )

    # Modified Cholesky factorization
    F = MCRHMC.modified_cholesky(G, metric.u, 0)
    metric.last_factorization = F

    # ∂H/∂r = G⁻¹ r
    return F \ collect(r)
end

# Get regularization sensitivities for adaptor
function AdvancedHMC.get_regularization_sensitivities(
    metric::QuadraticObservedHessianMetric{T}
) where {T}
    if metric.last_factorization === nothing
        return zeros(T, size(metric, 1))
    end
    return metric.last_factorization.sensitivities
end

# ============================================================================
# LogDensityProblems Model
# ============================================================================

struct LogTargetDensityQuadratic{T,PM,PV}
    K::Int  # Store K explicitly
    dim::Int
    prior::MvNormal
    dyn::VanDerPolDynamicsParam{T}
    ys::Vector{SVector{1,T}}
    prior_mean::PM
    prior_prec::PV
    obs_indices::Vector{Int}
    σ_obs::T
end

function LogTargetDensityQuadratic(
    K::Int, prior, dyn, ys, prior_mean, prior_var, obs_indices, σ_obs
)
    D = 2
    P = 1
    dim = K * D + P
    prior_prec = SVector{P,Float64}(1 ./ prior_var)
    return LogTargetDensityQuadratic{Float64,typeof(prior_mean),typeof(prior_prec)}(
        K, dim, prior, dyn, ys, prior_mean, prior_prec, obs_indices, σ_obs
    )
end

function LogDensityProblems.logdensity(p::LogTargetDensityQuadratic, θ)
    return calc_ll_quadratic(
        θ, p.ys, p.prior, p.dyn, p.K, p.prior_mean, p.prior_prec, p.obs_indices, p.σ_obs
    )
end

function LogDensityProblems.logdensity_and_gradient(p::LogTargetDensityQuadratic, θ)
    ll = calc_ll_quadratic(
        θ, p.ys, p.prior, p.dyn, p.K, p.prior_mean, p.prior_prec, p.obs_indices, p.σ_obs
    )
    grad = calc_ll_grad_quadratic(
        θ, p.ys, p.prior, p.dyn, p.K, p.prior_mean, p.prior_prec, p.obs_indices, p.σ_obs
    )
    return ll, grad
end

LogDensityProblems.dimension(p::LogTargetDensityQuadratic) = p.dim

function LogDensityProblems.capabilities(::Type{<:LogTargetDensityQuadratic})
    return LogDensityProblems.LogDensityOrder{1}()
end

# ============================================================================
# Run MCMC
# ============================================================================

println("Creating log density model with K=$K, D=$D, P=$P")
println("Expected dimension: $(K * D + P)")
ℓπ = LogTargetDensityQuadratic(K, prior, dyn, ys, prior_mean, prior_var, obs_indices, σ_obs)
println("LogTargetDensityQuadratic created: K=$(ℓπ.K), dim=$(ℓπ.dim)")
model = AdvancedHMC.LogDensityModel(ℓπ)

# Initial state: true states + true parameters
initial_θ = vcat(from_block_vector(zs_true_block), collect(θ_true))
println("initial_θ length: $(length(initial_θ))")

# ============================================================================
# Kleppe's Observed Hessian RHMC (Quadratic Observation)
# ============================================================================

println("\n" * "="^60)
println("KLEPPE'S OBSERVED HESSIAN RHMC (QUADRATIC OBSERVATION)")
println("="^60)

# Regularization parameter for modified Cholesky
n = K * D + P
u_init = fill(1e1, n)

println("\nSetting up Kleppe's Observed Hessian metric with quadratic observation...")
println("Creating metric with K=$K, u_init length=$(length(u_init))")
kleppe_metric = QuadraticObservedHessianMetric(
    prior, dyn, ys, K, prior_mean, prior_var, obs_indices, u_init, σ_obs
)
println("Metric created: K=$(kleppe_metric.K), size=$(size(kleppe_metric))")
println(kleppe_metric)

# Build Hamiltonian with the observed Hessian metric
kleppe_hamiltonian = Hamiltonian(kleppe_metric, ℓπ)

# Use the same integrator settings as the Fisher/GGN approach
initial_ϵ = 0.01
kleppe_integrator = AdaptiveGeneralizedLeapfrog(initial_ϵ; max_iters=7)
kleppe_kernel = HMCKernel(
    Trajectory{MultinomialTS}(kleppe_integrator, GeneralisedNoUTurn())
)

kleppe_adaptor = StepSizeAdaptor(0.8, kleppe_integrator)
println("Adaptor: ", kleppe_adaptor)
kleppe_sampler = HMCSampler(kleppe_kernel, kleppe_metric, kleppe_adaptor)

# For quick testing, use small sample sizes
N_samples = 2000
N_adapt = 1000

println("\nRunning Kleppe RHMC (N_samples=$N_samples, N_adapt=$N_adapt)...")
kleppe_chains = AbstractMCMC.sample(
    model,
    kleppe_sampler,
    N_samples;
    n_adapts=N_adapt,
    initial_params=initial_θ,
    verbose=false,
    progress=true,
)
kleppe_samples = [s.z.θ for s in kleppe_chains]

# Print total acceptance rate
accept = [s.stat.is_accept for s in kleppe_chains]
println("Total Acceptance Rate: ", mean(accept))

accept_prop = [s.stat.acceptance_rate for s in kleppe_chains]
println("Mean Acceptance Probability: ", mean(accept_prop))

# Extract parameter samples (last P elements)
kleppe_μ_samples = [exp(s[K * D + 1]) for s in kleppe_samples]

# Compute ESS for all dimensions
total_dim = D * K + P
kleppe_samples_arr = Array{Float64}(undef, N_samples, 1, total_dim)
for i in 1:N_samples
    for j in 1:total_dim
        kleppe_samples_arr[i, 1, j] = kleppe_samples[i][j]
    end
end
kleppe_ess = ess(kleppe_samples_arr) ./ N_samples

println("\n=== Kleppe RHMC State ESS Statistics ===")
kleppe_state_ess = kleppe_ess[1:(D * K)]
println("Minimum State ESS: ", minimum(kleppe_state_ess))
println("Median State ESS: ", median(kleppe_state_ess))
println("Mean State ESS: ", mean(kleppe_state_ess))

println("\n=== Kleppe RHMC Parameter ESS ===")
kleppe_param_ess = kleppe_ess[(D * K + 1):end]
println("log(μ) ESS: ", kleppe_param_ess[1])

println("\n=== Kleppe RHMC Parameter Posterior Summary ===")
burn_in = N_adapt
kleppe_μ_post = kleppe_μ_samples[(burn_in + 1):end]
println(
    "μ: true=$(μ_true), posterior mean=$(mean(kleppe_μ_post)), std=$(std(kleppe_μ_post))"
)

# ============================================================================
# Fisher/GGN RHMC (for comparison) - using linear observation model
# ============================================================================

println("\n" * "="^60)
println("FISHER/GGN RHMC (LINEAR OBSERVATION - COMPARISON)")
println("="^60)

println("\nNote: Fisher/GGN comparison uses linear observation model")
println("      since the existing implementation doesn't support quadratic observation.")

# Skip Fisher/GGN comparison since it requires different observation model
# (The existing BorderedBlockTridiagonalRiemannianMetric doesn't support quadratic observation)
println("Skipping Fisher/GGN comparison (requires different observation model)")

# ============================================================================
# Plots
# ============================================================================

# Trace plot for parameter
p_μ = plot(;
    title="μ trace (Kleppe with quadratic obs)",
    xlabel="Sample",
    ylabel="μ",
    legend=:topright,
    size=(800, 400),
)
plot!(p_μ, kleppe_μ_samples; label="Kleppe", lw=1, color=:blue, alpha=0.7)
hline!(p_μ, [μ_true]; label="True", lw=2, color=:red)

display(p_μ)

# Histogram of posterior parameter (post burn-in)
p_μ_hist = histogram(
    kleppe_μ_post;
    title="Kleppe (Quadratic Obs) Posterior",
    xlabel="μ",
    ylabel="Count",
    label="",
    bins=20,
    color=:blue,
    alpha=0.7,
)
vline!(p_μ_hist, [μ_true]; label="True", lw=2, color=:red)

display(p_μ_hist)

# Trajectory samples
n_plot_samples = min(50, length(kleppe_μ_post))
plot_idxs = round.(Int, LinRange(burn_in + 1, N_samples, n_plot_samples))

p_traj = plot(;
    title="Kleppe (Quadratic Obs) Trajectory Samples",
    xlabel="u",
    ylabel="v",
    size=(800, 600),
    aspect_ratio=1,
)

for i in plot_idxs
    s = kleppe_samples[i]
    plot!(
        p_traj,
        [s[1 + (k - 1) * D] for k in 1:K],
        [s[2 + (k - 1) * D] for k in 1:K];
        label="",
        lw=1,
        alpha=0.1,
        color=:blue,
    )
end
plot!(
    p_traj,
    [z[1] for z in zs_true],
    [z[2] for z in zs_true];
    label="Truth",
    lw=2,
    color=:black,
)

display(p_traj)

println("\nDone!")
