"""
Exponential family distributions for use in state space model observations.

Each exponential family is defined by its log-partition function A(η) and base measure h(y).
The key identities are:
- E[T(y)|η] = ∇A(η)
- Cov(T(y)|η) = ∇²A(η) = S (Fisher information)

For RHMC, we need A(η) and its first three derivatives.
"""

export ExponentialFamily, ProductExponentialFamily
export GaussianNatural, GaussianMean, PoissonNatural, BernoulliNatural
export log_partition, log_partition_d1, log_partition_d2, log_partition_d3
export log_base_measure, sufficient_stat, log_likelihood, rand_obs

# =============================================================================
# Abstract Type
# =============================================================================

"""
    ExponentialFamily

Abstract type for univariate exponential family distributions in natural parameter form.

Required methods:
- `log_partition(ef, η)` - A(η), the log-partition function
- `log_partition_d1(ef, η)` - A'(η) = E[T(y)|η]
- `log_partition_d2(ef, η)` - A''(η) = S (Fisher information, always ≥ 0)
- `log_partition_d3(ef, η)` - A'''(η) = ∂S/∂η
- `log_base_measure(ef, y)` - log h(y)
- `sufficient_stat(ef, y)` - T(y)
"""
abstract type ExponentialFamily end

"""
    log_likelihood(ef, η, y)

Compute log p(y|η) = η⋅T(y) - A(η) + log h(y) for a univariate exponential family.
"""
function log_likelihood(ef::ExponentialFamily, η::T, y) where {T}
    return η * sufficient_stat(ef, y) - log_partition(ef, η) + log_base_measure(ef, y)
end

"""
    rand_obs(rng, ef, η)

Sample an observation from the exponential family with natural parameter η.
"""
function rand_obs end

# =============================================================================
# Gaussian (known variance)
# =============================================================================

"""
    GaussianNatural{T}(σ²::T)

Gaussian distribution with known variance σ² in natural parameter form.

Natural parameter: η = μ/σ²
Sufficient statistic: T(y) = y/σ²
Log-partition: A(η) = σ²η²/2 = μ²/(2σ²)
Base measure: h(y) = exp(-y²/(2σ²)) / √(2πσ²)

Fisher information: S = A''(η) = σ² (constant, does not depend on η)
"""
struct GaussianNatural{T} <: ExponentialFamily
    σ²::T
end

function log_partition(ef::GaussianNatural, η)
    return ef.σ² * η^2 / 2
end

function log_partition_d1(ef::GaussianNatural, η)
    return ef.σ² * η  # = μ = E[y]
end

function log_partition_d2(ef::GaussianNatural{T}, η) where {T}
    return ef.σ²  # = Var(y) = S (constant)
end

function log_partition_d3(ef::GaussianNatural{T}, η) where {T}
    return zero(T)  # S does not depend on η
end

function log_base_measure(ef::GaussianNatural, y)
    return -y^2 / (2 * ef.σ²) - log(2π * ef.σ²) / 2
end

function sufficient_stat(ef::GaussianNatural, y)
    return y / ef.σ²
end

function rand_obs(rng::AbstractRNG, ef::GaussianNatural{T}, η) where {T}
    μ = ef.σ² * η  # η = μ/σ², so μ = η*σ²
    return μ + sqrt(ef.σ²) * randn(rng, T)
end

# For Gaussian, it's often more convenient to work with μ directly.
# The Fisher information in terms of the mean is 1/σ².
# When we use η = μ (treating μ as the natural parameter for metric purposes),
# we get S = 1/σ².
#
# To simplify, we provide an alternative interpretation where:
# - η represents the mean μ directly
# - S = 1/σ² (the precision)
# This is what the original code effectively does with R_inv.

"""
    GaussianMean{T}(σ²::T)

Gaussian distribution parameterized by mean μ with known variance σ².

This is a convenience wrapper where:
- η = μ (the mean, not the canonical natural parameter)
- S = 1/σ² (the precision, used as Fisher information for metric)

This matches the original implementation's use of R_inv.
"""
struct GaussianMean{T} <: ExponentialFamily
    σ²::T
end

function log_partition(ef::GaussianMean, η)
    # η = μ, so A(μ) = μ²/(2σ²)
    return η^2 / (2 * ef.σ²)
end

function log_partition_d1(ef::GaussianMean, η)
    return η / ef.σ²
end

function log_partition_d2(ef::GaussianMean, η)
    return 1 / ef.σ²  # Precision
end

function log_partition_d3(ef::GaussianMean{T}, η) where {T}
    return zero(T)
end

function log_base_measure(ef::GaussianMean, y)
    return -y^2 / (2 * ef.σ²) - log(2π * ef.σ²) / 2
end

function sufficient_stat(ef::GaussianMean, y)
    return y / ef.σ²
end

# Convenience: log-likelihood in mean parameterization
function log_likelihood(ef::GaussianMean, μ, y)
    return -(y - μ)^2 / (2 * ef.σ²) - log(2π * ef.σ²) / 2
end

function rand_obs(rng::AbstractRNG, ef::GaussianMean{T}, η) where {T}
    # η = μ (mean parameterization)
    return η + sqrt(ef.σ²) * randn(rng, T)
end

# =============================================================================
# Poisson
# =============================================================================

"""
    PoissonNatural

Poisson distribution in natural parameter form.

Natural parameter: η = log(λ) where λ is the rate
Sufficient statistic: T(y) = y
Log-partition: A(η) = exp(η) = λ
Base measure: h(y) = 1/y!

Fisher information: S = A''(η) = exp(η) = λ (depends on η)
"""
struct PoissonNatural <: ExponentialFamily end

function log_partition(::PoissonNatural, η)
    return exp(η)
end

function log_partition_d1(::PoissonNatural, η)
    return exp(η)  # = λ = E[y]
end

function log_partition_d2(::PoissonNatural, η)
    return exp(η)  # = λ = Var(y) = S
end

function log_partition_d3(::PoissonNatural, η)
    return exp(η)  # ∂S/∂η = λ
end

function log_base_measure(::PoissonNatural, y)
    return -logfactorial(y)
end

function sufficient_stat(::PoissonNatural, y)
    return y
end

function rand_obs(rng::AbstractRNG, ::PoissonNatural, η)
    λ = exp(η)
    return Float64(rand(rng, Poisson(λ)))
end

# =============================================================================
# Bernoulli
# =============================================================================

"""
    BernoulliNatural

Bernoulli distribution in natural parameter form (logistic).

Natural parameter: η = log(p/(1-p)) = logit(p)
Sufficient statistic: T(y) = y
Log-partition: A(η) = log(1 + exp(η))
Base measure: h(y) = 1

Fisher information: S = A''(η) = σ(η)(1-σ(η)) where σ(η) = 1/(1+exp(-η))
"""
struct BernoulliNatural <: ExponentialFamily end

function log_partition(::BernoulliNatural, η)
    # log(1 + exp(η)), numerically stable version
    return η > 0 ? η + log1p(exp(-η)) : log1p(exp(η))
end

function log_partition_d1(::BernoulliNatural, η)
    # σ(η) = 1/(1+exp(-η)) = exp(η)/(1+exp(η))
    return 1 / (1 + exp(-η))
end

function log_partition_d2(::BernoulliNatural, η)
    # σ(η)(1-σ(η))
    σ = 1 / (1 + exp(-η))
    return σ * (1 - σ)
end

function log_partition_d3(::BernoulliNatural, η)
    # ∂/∂η [σ(1-σ)] = σ(1-σ)(1-2σ)
    σ = 1 / (1 + exp(-η))
    return σ * (1 - σ) * (1 - 2σ)
end

function log_base_measure(::BernoulliNatural, y)
    return 0.0
end

function sufficient_stat(::BernoulliNatural, y)
    return y
end

function rand_obs(rng::AbstractRNG, ::BernoulliNatural, η)
    p = 1 / (1 + exp(-η))
    return Float64(rand(rng, Bernoulli(p)))
end

# =============================================================================
# Product Exponential Family
# =============================================================================

"""
    ProductExponentialFamily{Dy, EF<:ExponentialFamily}

Product of Dy independent copies of a univariate exponential family.

For observations y = (y_1, ..., y_Dy) with natural parameters η = (η_1, ..., η_Dy):
- log p(y|η) = Σ_m [η_m T(y_m) - A(η_m) + log h(y_m)]
- Fisher information S = diag(A''(η_1), ..., A''(η_Dy))

Methods take SVector{Dy} inputs for η and y.
"""
struct ProductExponentialFamily{Dy,EF<:ExponentialFamily}
    ef::EF
end

function ProductExponentialFamily{Dy}(ef::EF) where {Dy,EF<:ExponentialFamily}
    return ProductExponentialFamily{Dy,EF}(ef)
end

"""
    log_partition(pef, η::SVector{Dy})

Sum of log-partition values: Σ_m A(η_m)
"""
function log_partition(pef::ProductExponentialFamily{Dy}, η::SVector{Dy}) where {Dy}
    result = zero(eltype(η))
    @inbounds for m in 1:Dy
        result += log_partition(pef.ef, η[m])
    end
    return result
end

"""
    log_partition_d1(pef, η::SVector{Dy})

Vector of first derivatives: (A'(η_1), ..., A'(η_Dy))
"""
function log_partition_d1(pef::ProductExponentialFamily{Dy}, η::SVector{Dy,T}) where {Dy,T}
    return SVector{Dy,T}(ntuple(m -> log_partition_d1(pef.ef, η[m]), Val(Dy)))
end

"""
    log_partition_d2(pef, η::SVector{Dy})

Diagonal Fisher information matrix: diag(A''(η_1), ..., A''(η_Dy))
Returns an SVector{Dy} representing the diagonal.
"""
function log_partition_d2(pef::ProductExponentialFamily{Dy}, η::SVector{Dy,T}) where {Dy,T}
    return SVector{Dy,T}(ntuple(m -> log_partition_d2(pef.ef, η[m]), Val(Dy)))
end

"""
    log_partition_d3(pef, η::SVector{Dy})

Diagonal of ∂S/∂η: (A'''(η_1), ..., A'''(η_Dy))
Returns an SVector{Dy} representing the diagonal.
"""
function log_partition_d3(pef::ProductExponentialFamily{Dy}, η::SVector{Dy,T}) where {Dy,T}
    return SVector{Dy,T}(ntuple(m -> log_partition_d3(pef.ef, η[m]), Val(Dy)))
end

"""
    log_base_measure(pef, y::SVector{Dy})

Sum of log base measures: Σ_m log h(y_m)
"""
function log_base_measure(pef::ProductExponentialFamily{Dy}, y::SVector{Dy}) where {Dy}
    result = zero(eltype(y))
    @inbounds for m in 1:Dy
        result += log_base_measure(pef.ef, y[m])
    end
    return result
end

"""
    sufficient_stat(pef, y::SVector{Dy})

Vector of sufficient statistics: (T(y_1), ..., T(y_Dy))
"""
function sufficient_stat(pef::ProductExponentialFamily{Dy}, y::SVector{Dy,T}) where {Dy,T}
    return SVector{Dy,T}(ntuple(m -> sufficient_stat(pef.ef, y[m]), Val(Dy)))
end

"""
    log_likelihood(pef, η::SVector{Dy}, y::SVector{Dy})

Log-likelihood for product distribution: Σ_m log p(y_m|η_m)
"""
function log_likelihood(
    pef::ProductExponentialFamily{Dy}, η::SVector{Dy}, y::SVector{Dy}
) where {Dy}
    result = zero(eltype(η))
    @inbounds for m in 1:Dy
        result += log_likelihood(pef.ef, η[m], y[m])
    end
    return result
end

"""
    rand_obs(rng, pef, η::SVector{Dy})

Sample from product distribution: independent samples for each component.
"""
function rand_obs(
    rng::AbstractRNG, pef::ProductExponentialFamily{Dy}, η::SVector{Dy,T}
) where {Dy,T}
    return SVector{Dy,T}(ntuple(m -> rand_obs(rng, pef.ef, η[m]), Val(Dy)))
end

# =============================================================================
# Hessian Types for Clarity
# =============================================================================

export ComponentHessians, ComponentMixedHessians

"""
    ComponentHessians{N, D, T}

Hessians of each component of a vector-valued function f: R^D → R^N.
Entry `h[i]` is ∇²f_i, a D×D symmetric matrix.
"""
struct ComponentHessians{N,D,T,L}
    hessians::NTuple{N,SMatrix{D,D,T,L}}
end

function ComponentHessians(hessians::NTuple{N,SMatrix{D,D,T,L}}) where {N,D,T,L}
    return ComponentHessians{N,D,T,L}(hessians)
end

Base.getindex(ch::ComponentHessians, i::Int) = ch.hessians[i]
Base.length(::ComponentHessians{N}) where {N} = N

"""
    ComponentMixedHessians{N, D, P, T}

Mixed Hessians of each component of a vector-valued function f: R^D × R^P → R^N.
Entry `h[i]` is ∇²_{x,θ}f_i, a D×P matrix.
"""
struct ComponentMixedHessians{N,D,P,T,L}
    hessians::NTuple{N,SMatrix{D,P,T,L}}
end

function ComponentMixedHessians(hessians::NTuple{N,SMatrix{D,P,T,L}}) where {N,D,P,T,L}
    return ComponentMixedHessians{N,D,P,T,L}(hessians)
end

Base.getindex(cmh::ComponentMixedHessians, i::Int) = cmh.hessians[i]
Base.length(::ComponentMixedHessians{N}) where {N} = N
