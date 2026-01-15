export FitzHughNagumoParam
export f_param, calc_Jf_param, calc_Hfs_param, calc_Q_param, calc_Qinv_param
export calc_f_θ, calc_Hf_θs, calc_Jf_θ, calc_f_θθ

"""
FitzHugh-Nagumo dynamics for Poisson SSM benchmarking.

The state is `[u, v]` where u is the fast "voltage-like" variable and v is the
slow "recovery" variable. This excitable system produces occasional spikes
separated by calm periods, creating heterogeneous curvature ideal for testing RHMC.

Continuous-time dynamics:
    u̇ = u - (1/3)u³ - v + I
    v̇ = (1/τ)(u + a - b·v)

Euler discretization:
    u_{k+1} = u_k + δt·[u_k - (1/3)u_k³ - v_k + I] + ε_u
    v_{k+1} = v_k + δt·[(1/τ)(u_k + a - b·v_k)] + ε_v

where ε ~ N(0, Q) with Q = diag(σ_u², σ_v²).

The dynamics are parameter-free (θ only appears in observation model),
so all parameter derivatives are zero.
"""
struct FitzHughNagumoParam{T}
    σ_u::T    # Process noise std for u
    σ_v::T    # Process noise std for v
    δt::T     # Time step
    a::T      # FHN parameter (typically 0.7)
    b::T      # FHN parameter (typically 0.8)
    τ::T      # Time scale ratio (typically 12.0)
    I::T      # External input current (typically 0.5)
end

"""
Dynamics function f(z, θ) - the θ is unused but required for interface compatibility.
"""
function f_param(dyn::FitzHughNagumoParam{T}, z::SVector{2,T}, θ::SVector{P,T}) where {T,P}
    u, v = z
    δt = dyn.δt
    a, b, τ, I = dyn.a, dyn.b, dyn.τ, dyn.I

    new_u = u + δt * (u - (1 / 3) * u^3 - v + I)
    new_v = v + δt * (1 / τ) * (u + a - b * v)

    return @SVector [new_u, new_v]
end

"""
Jacobian ∂f/∂z (2×2 matrix).

∂f_u/∂u = 1 + δt·(1 - u²)
∂f_u/∂v = -δt
∂f_v/∂u = δt/τ
∂f_v/∂v = 1 - δt·b/τ
"""
function calc_Jf_param(
    dyn::FitzHughNagumoParam{T}, z::SVector{2,T}, θ::SVector{P,T}
) where {T,P}
    u, v = z
    δt = dyn.δt
    b, τ = dyn.b, dyn.τ

    df_u_du = 1 + δt * (1 - u^2)
    df_u_dv = -δt
    df_v_du = δt / τ
    df_v_dv = 1 - δt * b / τ

    return @SMatrix [
        df_u_du df_u_dv
        df_v_du df_v_dv
    ]
end

"""
Hessians ∂²f/∂z_d∂z (D matrices, each 2×2).

For f_u: ∂²f_u/∂u² = -2δt·u, all others zero
For f_v: all second derivatives zero (linear in state)
"""
function calc_Hfs_param(
    dyn::FitzHughNagumoParam{T}, z::SVector{2,T}, θ::SVector{P,T}
) where {T,P}
    u, v = z
    δt = dyn.δt

    # Hf_du (∂²f/∂u∂z) - derivatives of Jf[:, 1] w.r.t. z
    # Jf[:, 1] = [1 + δt·(1 - u²), δt/τ]
    # ∂/∂u: [-2δt·u, 0]
    # ∂/∂v: [0, 0]
    Hf1 = @SMatrix [
        -2*δt*u zero(T)
        zero(T) zero(T)
    ]

    # Hf_dv (∂²f/∂v∂z) - derivatives of Jf[:, 2] w.r.t. z
    # Jf[:, 2] = [-δt, 1 - δt·b/τ]
    # All constant w.r.t. z
    Hf2 = @SMatrix [
        zero(T) zero(T)
        zero(T) zero(T)
    ]

    return Hf1, Hf2
end

"""
Parameter Jacobian ∂f/∂θ (2×P matrix).
All zero since f doesn't depend on θ.
"""
function calc_f_θ(dyn::FitzHughNagumoParam{T}, z::SVector{2,T}, θ::SVector{P,T}) where {T,P}
    return @SMatrix zeros(T, 2, P)
end

"""
Mixed Hessians ∂²f/∂z_d∂θ (D matrices, each 2×P).
All zero since f doesn't depend on θ.
"""
function calc_Hf_θs(
    dyn::FitzHughNagumoParam{T}, z::SVector{2,T}, θ::SVector{P,T}
) where {T,P}
    Z = @SMatrix zeros(T, 2, P)
    return (Z, Z)
end

"""
Jacobian derivatives ∂Jf/∂θ^{(p)} (P matrices, each 2×2).
All zero since Jf doesn't depend on θ.
"""
function calc_Jf_θ(
    dyn::FitzHughNagumoParam{T}, z::SVector{2,T}, θ::SVector{P,T}
) where {T,P}
    Z = @SMatrix zeros(T, 2, 2)
    return ntuple(_ -> Z, P)
end

"""
Second parameter derivatives ∂²f/∂θ^{(p)}∂θ (P matrices, each 2×P).
All zero since f doesn't depend on θ.
"""
function calc_f_θθ(
    dyn::FitzHughNagumoParam{T}, z::SVector{2,T}, θ::SVector{P,T}
) where {T,P}
    Z = @SMatrix zeros(T, 2, P)
    return ntuple(_ -> Z, P)
end

function calc_Q_param(dyn::FitzHughNagumoParam)
    σ_u = dyn.σ_u
    σ_v = dyn.σ_v
    return Diagonal(@SVector([σ_u^2, σ_v^2]))
end

function calc_Qinv_param(dyn::FitzHughNagumoParam)
    σ_u = dyn.σ_u
    σ_v = dyn.σ_v
    return Diagonal(@SVector([1.0 / σ_u^2, 1.0 / σ_v^2]))
end
