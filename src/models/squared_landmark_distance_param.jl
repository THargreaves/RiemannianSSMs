export TwoLandmarkMeasurementModelParam
export h_param, calc_Jh_param, calc_Hhs_param, calc_R_param, calc_Rinv_param
export calc_h_θ, calc_Hh_θs, calc_Jh_θ, calc_h_θθ

"""
A 2D measurement model with squared distances to two fixed landmarks.
This version is compatible with parameter inference but h does not depend on θ.

The state is `[x, y, vx, vy]`, where `(x, y)` is position and `(vx, vy)` is velocity.
The measurement function returns squared distances to two landmarks:
y1 = (x - a1)^2 + (y - b1)^2 + v1
y2 = (x - a2)^2 + (y - b2)^2 + v2

where `(a1, b1)` and `(a2, b2)` are the landmark positions and `v1, v2` are measurement noise.

Since landmark positions are fixed constants, h does NOT depend on θ.
All θ-derivatives return zeros.
"""
struct TwoLandmarkMeasurementModelParam{T}
    a1::T   # landmark 1 x-coordinate
    b1::T   # landmark 1 y-coordinate
    a2::T   # landmark 2 x-coordinate
    b2::T   # landmark 2 y-coordinate
    σ1::T   # measurement noise std for landmark 1
    σ2::T   # measurement noise std for landmark 2
end

function h_param(
    model::TwoLandmarkMeasurementModelParam{T}, z::SVector{4,T}, θ::SVector{2,T}
) where {T}
    x, y, vx, vy = z
    a1, b1, a2, b2 = model.a1, model.b1, model.a2, model.b2

    h1 = (x - a1)^2 + (y - b1)^2
    h2 = (x - a2)^2 + (y - b2)^2

    return @SVector [h1, h2]
end

function calc_Jh_param(
    model::TwoLandmarkMeasurementModelParam{T}, z::SVector{4,T}, θ::SVector{2,T}
) where {T}
    x, y, vx, vy = z
    a1, b1, a2, b2 = model.a1, model.b1, model.a2, model.b2

    Jh = @SMatrix [
        2*(x - a1) 2*(y - b1) 0.0 0.0
        2*(x - a2) 2*(y - b2) 0.0 0.0
    ]
    return Jh
end

"""
Parameter Jacobian ∂h/∂θ (2×2 matrix).
Since h does not depend on θ, this returns zeros.
"""
function calc_h_θ(
    model::TwoLandmarkMeasurementModelParam{T}, z::SVector{4,T}, θ::SVector{2,T}
) where {T}
    return @SMatrix zeros(T, 2, 2)
end

function calc_Hhs_param(
    model::TwoLandmarkMeasurementModelParam{T}, z::SVector{4,T}, θ::SVector{2,T}
) where {T}
    # Same as original - these don't depend on θ
    Hh1 = @SMatrix [
        2.0 0.0 0.0 0.0
        2.0 0.0 0.0 0.0
    ]

    Hh2 = @SMatrix [
        0.0 2.0 0.0 0.0
        0.0 2.0 0.0 0.0
    ]

    Hh3 = @SMatrix [
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
    ]

    Hh4 = @SMatrix [
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
    ]

    return Hh1, Hh2, Hh3, Hh4
end

"""
Mixed Hessians ∂²h/∂z_d∂θ (D matrices, each 2×2).
Since h does not depend on θ, all return zeros.
"""
function calc_Hh_θs(
    model::TwoLandmarkMeasurementModelParam{T}, z::SVector{4,T}, θ::SVector{2,T}
) where {T}
    Z = @SMatrix zeros(T, 2, 2)
    return Z, Z, Z, Z
end

"""
Jacobian derivatives ∂Jh/∂θ^{(p)} (P matrices, each 2×4).
Since h does not depend on θ, all return zeros.
"""
function calc_Jh_θ(
    model::TwoLandmarkMeasurementModelParam{T}, z::SVector{4,T}, θ::SVector{2,T}
) where {T}
    Z = @SMatrix zeros(T, 2, 4)
    return Z, Z
end

"""
Second parameter derivatives ∂²h/∂θ^{(p)}∂θ (P matrices, each 2×2).
Since h does not depend on θ, all return zeros.
"""
function calc_h_θθ(
    model::TwoLandmarkMeasurementModelParam{T}, z::SVector{4,T}, θ::SVector{2,T}
) where {T}
    Z = @SMatrix zeros(T, 2, 2)
    return Z, Z
end

function calc_R_param(model::TwoLandmarkMeasurementModelParam)
    σ1 = model.σ1
    σ2 = model.σ2
    return Diagonal(@SVector([σ1^2, σ2^2]))
end

function calc_Rinv_param(model::TwoLandmarkMeasurementModelParam)
    σ1 = model.σ1
    σ2 = model.σ2
    return Diagonal(@SVector([1.0 / σ1^2, 1.0 / σ2^2]))
end
