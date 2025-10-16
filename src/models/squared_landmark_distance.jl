export TwoLandmarkMeasurementModel, h, calc_Jh, calc_Hhs, calc_R, calc_Rinv

"""
A 2D measurement model with squared distances to two fixed landmarks.

The state is `[x, y, vx, vy]`, where `(x, y)` is position and `(vx, vy)` is velocity. The
measurement function returns squared distances to two landmarks:
y1 = (x - a1)^2 + (y - b1)^2 + v1
y2 = (x - a2)^2 + (y - b2)^2 + v2

where `(a1, b1)` and `(a2, b2)` are the landmark positions and `v1, v2` are measurement noise.
"""
struct TwoLandmarkMeasurementModel{T}
    a1::T   # landmark 1 x-coordinate
    b1::T   # landmark 1 y-coordinate
    a2::T   # landmark 2 x-coordinate  
    b2::T   # landmark 2 y-coordinate
    σ1::T   # measurement noise std for landmark 1
    σ2::T   # measurement noise std for landmark 2
end

function h(model::TwoLandmarkMeasurementModel{T}, z::SVector{4,T}) where {T}
    x, y, vx, vy = z
    a1, b1, a2, b2 = model.a1, model.b1, model.a2, model.b2

    h1 = (x - a1)^2 + (y - b1)^2
    h2 = (x - a2)^2 + (y - b2)^2

    return @SVector [h1, h2]
end

# AbstractVector version
function h(model::TwoLandmarkMeasurementModel, z::AbstractVector)
    x, y, vx, vy = z
    a1, b1, a2, b2 = model.a1, model.b1, model.a2, model.b2
    h1 = (x - a1)^2 + (y - b1)^2
    h2 = (x - a2)^2 + (y - b2)^2
    return [h1, h2]
end

function calc_Jh(model::TwoLandmarkMeasurementModel{T}, z::SVector{4,T}) where {T}
    x, y, vx, vy = z
    a1, b1, a2, b2 = model.a1, model.b1, model.a2, model.b2

    # Jacobian matrix (2x4)
    Jh = @SMatrix [
        2*(x - a1) 2*(y - b1) 0.0 0.0
        2*(x - a2) 2*(y - b2) 0.0 0.0
    ]
    return Jh
end

function calc_Hhs(model::TwoLandmarkMeasurementModel{T}, z::SVector{4,T}) where {T}
    x, y, vx, vy = z

    # Hh1: ∂²h/∂x∂[x,y,vx,vy] (2x4 matrix)
    Hh1 = @SMatrix [
        2.0 0.0 0.0 0.0
        2.0 0.0 0.0 0.0
    ]

    # Hh2: ∂²h/∂y∂[x,y,vx,vy] (2x4 matrix)
    Hh2 = @SMatrix [
        0.0 2.0 0.0 0.0
        0.0 2.0 0.0 0.0
    ]

    # Hh3: ∂²h/∂vx∂[x,y,vx,vy] (2x4 matrix) - all zeros
    Hh3 = @SMatrix [
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
    ]

    # Hh4: ∂²h/∂vy∂[x,y,vx,vy] (2x4 matrix) - all zeros
    Hh4 = @SMatrix [
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
    ]

    return Hh1, Hh2, Hh3, Hh4
end

function calc_R(model::TwoLandmarkMeasurementModel)
    σ1 = model.σ1
    σ2 = model.σ2
    return Diagonal(@SVector([σ1^2, σ2^2]))
end

function calc_Rinv(model::TwoLandmarkMeasurementModel)
    σ1 = model.σ1
    σ2 = model.σ2
    return Diagonal(@SVector([1.0 / σ1^2, 1.0 / σ2^2]))
end
