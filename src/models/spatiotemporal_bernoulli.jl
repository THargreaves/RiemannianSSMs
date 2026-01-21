"""
Spatiotemporal state space model with Bernoulli observations on irregular node locations.

State: x ∈ R^d where d is the number of spatial nodes
Parameters: θ = [log(κ), log(g), log(a), b] where
    - κ > 0: coupling/diffusion strength
    - g > 0: local reaction/nonlinearity strength
    - a > 0: measurement sensitivity
    - b ∈ R: baseline intercept

Dynamics (nonlinear diffusion + stabilizing reaction):
    x_{t+1} = x_t + Δt[-κ L_K x_t - g tanh(x_t)] + ε_t
    ε_t ~ N(0, σ² I_d)

The -g tanh(x) term provides stabilization toward zero (when x > 0, it pushes down; when x < 0, it pushes up).

where L_K = I - W is the kernel Laplacian with:
    - K_{ij} = exp(-r_{ij}²/(2ℓ²)) for i ≠ j, K_{ii} = 0
    - W = D^{-1/2} K D^{-1/2} (symmetric normalization)
    - D = diag(K·1) (degree matrix)

Observations (Bernoulli with logistic link):
    y_{t,i} ~ Bernoulli(p_{t,i})
    p_{t,i} = logistic(b + a·x_{t,i})
    η_{t,i} = b + a·x_{t,i} (natural parameter)
"""

export SpatiotemporalBernoulliModel, median_nearest_neighbor_distance

struct SpatiotemporalBernoulliModel{Dx,T,MT<:AbstractMatrix{T}} <: StateSpaceModel{Dx,Dx,4}
    Δt::T                           # Time step
    σ::T                            # Process noise std
    L_K::MT                         # Kernel Laplacian (d × d)
    μ0::SVector{Dx,T}               # Initial state mean
    Σ0::Diagonal{T,SVector{Dx,T}}   # Initial state covariance
end

"""
    SpatiotemporalBernoulliModel(positions, ℓ; Δt, σ, s0)

Construct a spatiotemporal Bernoulli model from node positions.

# Arguments
- `positions`: Vector of d node positions, each an SVector{p} for p-dimensional space
- `ℓ`: Kernel lengthscale

# Keyword Arguments
- `Δt`: Time step (default: 0.1)
- `σ`: Process noise std (default: 0.1)
- `s0`: Initial state prior std (default: 1.0), so P0 = s0² I
"""
function SpatiotemporalBernoulliModel(
    positions::Vector{SVector{P,T}}, ℓ::T; Δt::T=T(0.1), σ::T=T(0.1), s0::T=T(1.0)
) where {P,T}
    d = length(positions)

    # Compute kernel Laplacian
    L_K = compute_kernel_laplacian(positions, ℓ)

    # Initial state parameters
    μ0 = SVector{d,T}(zeros(T, d))
    Σ0_diag = SVector{d,T}(fill(s0^2, d))
    Σ0 = Diagonal(Σ0_diag)

    return SpatiotemporalBernoulliModel{d,T,typeof(L_K)}(Δt, σ, L_K, μ0, Σ0)
end

"""
    compute_kernel_laplacian(positions, ℓ)

Compute the symmetric normalized kernel Laplacian L_K = I - W.
"""
function compute_kernel_laplacian(positions::Vector{SVector{P,T}}, ℓ::T) where {P,T}
    d = length(positions)

    # Compute pairwise distances and kernel
    K = zeros(T, d, d)
    for i in 1:d
        for j in 1:d
            if i != j
                r_ij = norm(positions[i] - positions[j])
                K[i, j] = exp(-r_ij^2 / (2 * ℓ^2))
            end
        end
    end

    # Degree vector
    deg = vec(sum(K; dims=2))

    # Symmetric normalization: W = D^{-1/2} K D^{-1/2}
    D_inv_sqrt = Diagonal(1.0 ./ sqrt.(deg))
    W = D_inv_sqrt * K * D_inv_sqrt

    # Kernel Laplacian
    L_K = SMatrix{d,d,T}(I - W)

    return L_K
end

"""
    median_nearest_neighbor_distance(positions)

Compute the median nearest-neighbor distance among node positions.
Useful for setting the kernel lengthscale.
"""
function median_nearest_neighbor_distance(positions::Vector{SVector{P,T}}) where {P,T}
    d = length(positions)
    nn_dists = Vector{T}(undef, d)
    for i in 1:d
        min_dist = typemax(T)
        for j in 1:d
            if i != j
                dist = norm(positions[i] - positions[j])
                min_dist = min(min_dist, dist)
            end
        end
        nn_dists[i] = min_dist
    end
    return median(nn_dists)
end

# =============================================================================
# Configuration Methods
# =============================================================================

function dynamics_covariance(model::SpatiotemporalBernoulliModel{Dx,T}) where {Dx,T}
    return Diagonal(SVector{Dx,T}(ntuple(_ -> model.σ^2, Val(Dx))))
end

function dynamics_covariance_inv(model::SpatiotemporalBernoulliModel{Dx,T}) where {Dx,T}
    return Diagonal(SVector{Dx,T}(ntuple(_ -> 1 / model.σ^2, Val(Dx))))
end

function initial_prior(model::SpatiotemporalBernoulliModel)
    return MvNormal(model.μ0, model.Σ0)
end

function observation_family(::SpatiotemporalBernoulliModel{Dx}) where {Dx}
    return ProductExponentialFamily{Dx}(BernoulliNatural())
end

# =============================================================================
# Dynamics: f(x, θ) = x + Δt[-κ L_K x - g tanh(x)]
# θ = [log(κ), log(g), log(a), b]
# =============================================================================

function f(
    model::SpatiotemporalBernoulliModel{Dx,T}, x::SVector{Dx,T}, θ::SVector{4,T}
) where {Dx,T}
    κ = exp(θ[1])
    g = exp(θ[2])
    Δt = model.Δt
    L_K = model.L_K

    # Diffusion term: -κ L_K x
    diffusion = -κ * (L_K * x)

    # Reaction term: -g tanh(x) (stabilizing toward zero)
    reaction = -g * tanh.(x)

    return x + Δt * (diffusion + reaction)
end

"""
Jacobian ∂f/∂x = I + Δt[-κ L_K - g diag(sech²(x))]
"""
function Df_x(
    model::SpatiotemporalBernoulliModel{Dx,T}, x::SVector{Dx,T}, θ::SVector{4,T}
) where {Dx,T}
    κ = exp(θ[1])
    g = exp(θ[2])
    Δt = model.Δt
    L_K = model.L_K

    # sech²(x) = 1 - tanh²(x)
    sech2_x = SVector{Dx,T}(ntuple(i -> 1 - tanh(x[i])^2, Val(Dx)))

    # I + Δt[-κ L_K - g diag(sech²(x))]
    I_d = SMatrix{Dx,Dx,T}(I)
    return I_d + Δt * (-κ * L_K - g * Diagonal(sech2_x))
end

"""
Jacobian ∂f/∂θ (Dx × 4 matrix).

θ = [log(κ), log(g), log(a), b]

∂f/∂(log κ) = -Δt κ L_K x
∂f/∂(log g) = -Δt g tanh(x)
∂f/∂(log a) = 0 (a doesn't affect dynamics)
∂f/∂b = 0 (b doesn't affect dynamics)
"""
function Df_θ(
    model::SpatiotemporalBernoulliModel{Dx,T}, x::SVector{Dx,T}, θ::SVector{4,T}
) where {Dx,T}
    κ = exp(θ[1])
    g = exp(θ[2])
    Δt = model.Δt
    L_K = model.L_K

    df_dlogκ = -Δt * κ * (L_K * x)
    df_dlogg = -Δt * g * tanh.(x)
    df_dloga = @SVector zeros(T, Dx)
    df_db = @SVector zeros(T, Dx)

    return hcat(df_dlogκ, df_dlogg, df_dloga, df_db)
end

"""
∂(Df_x[:, d])/∂x for d = 1, ..., Dx.

Df_x = I + Δt[-κ L_K - g diag(sech²(x))]

The d-th column: Df_x[:, d] = e_d + Δt[-κ L_K[:, d] - g sech²(x_d) e_d]

∂(Df_x[:, d])/∂x_j:
- Only the diagonal reaction term depends on x
- ∂(-g sech²(x_d))/∂x_j = δ_{dj} g (2 tanh(x_d) sech²(x_d))

Returns Dx matrices, each Dx × Dx.
Entry [i, j] of matrix d is ∂²f_i/(∂x_d ∂x_j).
"""
function D2f_xx(
    model::SpatiotemporalBernoulliModel{Dx,T}, x::SVector{Dx,T}, θ::SVector{4,T}
) where {Dx,T}
    g = exp(θ[2])
    Δt = model.Δt

    # Precompute tanh and sech² for all components
    tanh_x = tanh.(x)
    sech2_x = SVector{Dx,T}(ntuple(i -> 1 - tanh_x[i]^2, Val(Dx)))

    # For each d, the only nonzero entry in ∂(Df_x[:, d])/∂x is at [d, d]
    # Value: Δt g (2 tanh(x_d) sech²(x_d)) -- note positive sign from -g * d/dx(-sech²)
    return ntuple(Val(Dx)) do d
        val = Δt * g * (2 * tanh_x[d] * sech2_x[d])
        # Use setindex workaround for SMatrix
        return SMatrix{Dx,Dx,T}(
            ntuple(Val(Dx * Dx)) do idx
                i = mod1(idx, Dx)
                j = div(idx - 1, Dx) + 1
                (i == d && j == d) ? val : zero(T)
            end,
        )
    end
end

"""
∂(Df_x[:, d])/∂θ for d = 1, ..., Dx.

Df_x[:, d] = e_d + Δt[-κ L_K[:, d] - g sech²(x_d) e_d]

∂(Df_x[:, d])/∂(log κ) = -Δt κ L_K[:, d]
∂(Df_x[:, d])/∂(log g) = -Δt g sech²(x_d) e_d
∂(Df_x[:, d])/∂(log a) = 0
∂(Df_x[:, d])/∂b = 0

Returns Dx matrices, each Dx × 4.
"""
function D2f_xθ(
    model::SpatiotemporalBernoulliModel{Dx,T}, x::SVector{Dx,T}, θ::SVector{4,T}
) where {Dx,T}
    κ = exp(θ[1])
    g = exp(θ[2])
    Δt = model.Δt
    L_K = model.L_K

    sech2_x = SVector{Dx,T}(ntuple(i -> 1 - tanh(x[i])^2, Val(Dx)))

    return ntuple(Val(Dx)) do d
        # ∂(Df_x[:, d])/∂(log κ) = -Δt κ L_K[:, d]
        col1 = -Δt * κ * SVector{Dx,T}(L_K[:, d])

        # ∂(Df_x[:, d])/∂(log g) = -Δt g sech²(x_d) e_d
        col2 = SVector{Dx,T}(ntuple(i -> i == d ? -Δt * g * sech2_x[d] : zero(T), Val(Dx)))

        # ∂(Df_x[:, d])/∂(log a) = 0, ∂(Df_x[:, d])/∂b = 0
        col3 = @SVector zeros(T, Dx)
        col4 = @SVector zeros(T, Dx)

        return hcat(col1, col2, col3, col4)
    end
end

"""
∂²f/∂θ_p∂θ for p = 1, 2, 3, 4.

For log-parameterization:
∂f/∂(log κ) = -Δt κ L_K x  →  ∂²f/∂(log κ)² = -Δt κ L_K x (same, since ∂κ/∂(log κ) = κ)
∂f/∂(log g) = -Δt g tanh(x)  →  ∂²f/∂(log g)² = -Δt g tanh(x) (same)

Cross terms are zero.

Returns 4 matrices, each Dx × 4.
"""
function D2f_θθ(
    model::SpatiotemporalBernoulliModel{Dx,T}, x::SVector{Dx,T}, θ::SVector{4,T}
) where {Dx,T}
    κ = exp(θ[1])
    g = exp(θ[2])
    Δt = model.Δt
    L_K = model.L_K

    df_dlogκ = -Δt * κ * (L_K * x)
    df_dlogg = -Δt * g * tanh.(x)
    Z = @SVector zeros(T, Dx)

    # ∂²f/∂(log κ)∂θ = [df_dlogκ, 0, 0, 0]
    H1 = hcat(df_dlogκ, Z, Z, Z)

    # ∂²f/∂(log g)∂θ = [0, df_dlogg, 0, 0]
    H2 = hcat(Z, df_dlogg, Z, Z)

    # ∂²f/∂(log a)∂θ = 0
    H3 = @SMatrix zeros(T, Dx, 4)

    # ∂²f/∂b∂θ = 0
    H4 = @SMatrix zeros(T, Dx, 4)

    return (H1, H2, H3, H4)
end

"""
∂(Df_x)/∂θ_p for p = 1, 2, 3, 4.

Df_x = I + Δt[-κ L_K - g diag(sech²(x))]

∂(Df_x)/∂(log κ) = -Δt κ L_K
∂(Df_x)/∂(log g) = -Δt g diag(sech²(x))
∂(Df_x)/∂(log a) = 0
∂(Df_x)/∂b = 0

Returns 4 matrices, each Dx × Dx.
"""
function DDf_x_θ(
    model::SpatiotemporalBernoulliModel{Dx,T}, x::SVector{Dx,T}, θ::SVector{4,T}
) where {Dx,T}
    κ = exp(θ[1])
    g = exp(θ[2])
    Δt = model.Δt
    L_K = model.L_K

    sech2_x = SVector{Dx,T}(ntuple(i -> 1 - tanh(x[i])^2, Val(Dx)))

    dDf_x_dlogκ = -Δt * κ * L_K
    dDf_x_dlogg = -Δt * g * SMatrix{Dx,Dx,T}(Diagonal(sech2_x))
    Z = @SMatrix zeros(T, Dx, Dx)

    return (dDf_x_dlogκ, dDf_x_dlogg, Z, Z)
end

# =============================================================================
# Observations: η(x, θ) = b + a·x (elementwise)
# θ = [log(κ), log(g), log(a), b]
# =============================================================================

"""
Natural parameter function. η_i = b + a·x_i where a = exp(θ[3]), b = θ[4].
"""
function η(
    model::SpatiotemporalBernoulliModel{Dx,T}, x::SVector{Dx,T}, θ::SVector{4,T}
) where {Dx,T}
    a = exp(θ[3])
    b = θ[4]
    return b .+ a .* x
end

"""
Jacobian ∂η/∂x (Dx × Dx diagonal matrix).
∂η_i/∂x_j = a δ_{ij}
"""
function Dη_x(
    model::SpatiotemporalBernoulliModel{Dx,T}, x::SVector{Dx,T}, θ::SVector{4,T}
) where {Dx,T}
    a = exp(θ[3])
    return a * SMatrix{Dx,Dx,T}(I)
end

"""
Jacobian ∂η/∂θ (Dx × 4 matrix).

∂η_i/∂(log κ) = 0
∂η_i/∂(log g) = 0
∂η_i/∂(log a) = a·x_i (chain rule)
∂η_i/∂b = 1
"""
function Dη_θ(
    model::SpatiotemporalBernoulliModel{Dx,T}, x::SVector{Dx,T}, θ::SVector{4,T}
) where {Dx,T}
    a = exp(θ[3])

    col1 = @SVector zeros(T, Dx)  # ∂η/∂(log κ)
    col2 = @SVector zeros(T, Dx)  # ∂η/∂(log g)
    col3 = a .* x                  # ∂η/∂(log a)
    col4 = @SVector ones(T, Dx)   # ∂η/∂b

    return hcat(col1, col2, col3, col4)
end

"""
∂(Dη_x[:, d])/∂x for d = 1, ..., Dx.

Dη_x = a·I, which doesn't depend on x.
All zeros.
"""
function D2η_xx(
    model::SpatiotemporalBernoulliModel{Dx,T}, x::SVector{Dx,T}, θ::SVector{4,T}
) where {Dx,T}
    Z = @SMatrix zeros(T, Dx, Dx)
    return ntuple(_ -> Z, Val(Dx))
end

"""
∂(Dη_x[:, d])/∂θ for d = 1, ..., Dx.

Dη_x[:, d] = a·e_d

∂(Dη_x[:, d])/∂(log κ) = 0
∂(Dη_x[:, d])/∂(log g) = 0
∂(Dη_x[:, d])/∂(log a) = a·e_d (chain rule)
∂(Dη_x[:, d])/∂b = 0

Returns Dx matrices, each Dx × 4.
"""
function D2η_xθ(
    model::SpatiotemporalBernoulliModel{Dx,T}, x::SVector{Dx,T}, θ::SVector{4,T}
) where {Dx,T}
    a = exp(θ[3])

    return ntuple(Val(Dx)) do d
        col1 = @SVector zeros(T, Dx)
        col2 = @SVector zeros(T, Dx)
        col3 = SVector{Dx,T}(ntuple(i -> i == d ? a : zero(T), Val(Dx)))
        col4 = @SVector zeros(T, Dx)
        return hcat(col1, col2, col3, col4)
    end
end

"""
∂²η/∂θ_p∂θ for p = 1, 2, 3, 4.

η_i = b + a·x_i = θ[4] + exp(θ[3])·x_i

∂η/∂(log κ) = 0  →  ∂²η/∂(log κ)∂θ = 0
∂η/∂(log g) = 0  →  ∂²η/∂(log g)∂θ = 0
∂η/∂(log a) = a·x  →  ∂²η/∂(log a)² = a·x (same)
∂η/∂b = 1  →  ∂²η/∂b∂θ = 0

Returns 4 matrices, each Dx × 4.
"""
function D2η_θθ(
    model::SpatiotemporalBernoulliModel{Dx,T}, x::SVector{Dx,T}, θ::SVector{4,T}
) where {Dx,T}
    a = exp(θ[3])
    Z = @SVector zeros(T, Dx)

    # ∂²η/∂(log κ)∂θ = 0
    H1 = @SMatrix zeros(T, Dx, 4)

    # ∂²η/∂(log g)∂θ = 0
    H2 = @SMatrix zeros(T, Dx, 4)

    # ∂²η/∂(log a)∂θ = [0, 0, a·x, 0]
    H3 = hcat(Z, Z, a .* x, Z)

    # ∂²η/∂b∂θ = 0
    H4 = @SMatrix zeros(T, Dx, 4)

    return (H1, H2, H3, H4)
end

"""
∂(Dη_x)/∂θ_p for p = 1, 2, 3, 4.

Dη_x = a·I

∂(Dη_x)/∂(log κ) = 0
∂(Dη_x)/∂(log g) = 0
∂(Dη_x)/∂(log a) = a·I
∂(Dη_x)/∂b = 0

Returns 4 matrices, each Dx × Dx.
"""
function DDη_x_θ(
    model::SpatiotemporalBernoulliModel{Dx,T}, x::SVector{Dx,T}, θ::SVector{4,T}
) where {Dx,T}
    a = exp(θ[3])
    Z = @SMatrix zeros(T, Dx, Dx)
    I_d = SMatrix{Dx,Dx,T}(I)

    return (Z, Z, a * I_d, Z)
end
