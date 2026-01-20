"""
Smooth absolute value function for modified Cholesky decomposition.

Based on arXiv:1612.04093v2 (Kleppe, "Modified Cholesky RMHMC").

The function sabs(x; u) provides a smooth, differentiable approximation to max(|x|, u).
"""

const LOG2 = log(2.0)
const INV_LOG2 = 1.0 / LOG2

"""
    sabs(x, u)

Smooth absolute value function: sabs(x; u) = (u/log(2)) * log(exp(x*log(2)/u) + exp(-x*log(2)/u))

Properties:
- sabs(x; u) ≥ u for all x, with equality only at x = 0
- sabs(x; u) > |x| for all finite x
- Smooth and differentiable everywhere
- Approximates max(|x|, u) for |x| >> u

Uses numerically stable formulation to avoid overflow.
"""
@inline function sabs(x::T, u::T) where {T<:AbstractFloat}
    z = x * LOG2 / u

    if z >= zero(T)
        return u * INV_LOG2 * (z + log1p(exp(-2z)))
    else
        return u * INV_LOG2 * (-z + log1p(exp(2z)))
    end
end

@inline sabs(x::Real, u::Real) = sabs(promote(float(x), float(u))...)

"""
    sabs_deriv(x, u)

Derivative of sabs with respect to x: d/dx sabs(x; u) = tanh(x * log(2) / u)
"""
@inline function sabs_deriv(x::T, u::T) where {T<:AbstractFloat}
    z = x * LOG2 / u
    return tanh(z)
end

@inline sabs_deriv(x::Real, u::Real) = sabs_deriv(promote(float(x), float(u))...)

"""
    sabs_with_deriv(x, u)

Compute both sabs(x, u) and its derivative in a single pass.
Returns (value, derivative).
"""
@inline function sabs_with_deriv(x::T, u::T) where {T<:AbstractFloat}
    z = x * LOG2 / u

    if z >= zero(T)
        e2z = exp(-2z)
        val = u * INV_LOG2 * (z + log1p(e2z))
        deriv = (one(T) - e2z) / (one(T) + e2z)
    else
        e2z = exp(2z)
        val = u * INV_LOG2 * (-z + log1p(e2z))
        deriv = (e2z - one(T)) / (e2z + one(T))
    end

    return val, deriv
end

@inline sabs_with_deriv(x::Real, u::Real) = sabs_with_deriv(promote(float(x), float(u))...)
