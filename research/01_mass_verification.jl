using GaussianDistributions
using LinearAlgebra
using Random
using SparseArrays

Dx = 2
Dy = 2
T = 10

SEED = 1729
rng = MersenneTwister(SEED)

A = rand(rng, Dx, Dx)
b = rand(rng, Dx)
Q = rand(rng, Dx, Dx)
Q = Q * Q' + I

H = rand(rng, Dy, Dx)
c = rand(rng, Dy)
R = rand(rng, Dy, Dy)
R = R * R' + I

μ0 = rand(rng, Dx)
Σ0 = rand(rng, Dx, Dx)
Σ0 = Σ0 * Σ0' + I

ys = [rand(rng, Dy) for _ in 1:T]

# Let Z = [X0, X1, ..., XT, Y1, ..., YT] be the joint state vector
# Write Z = P.Z + ϵ, where ϵ ~ N(μ_ϵ, Σ_ϵ)
P = zeros(Dx + T * (Dx + Dy), Dx + T * (Dx + Dy))
for t in 1:T
    iA = t * Dx + 1
    jA = (t - 1) * Dx + 1
    P[iA:(iA + Dx - 1), jA:(jA + Dx - 1)] = A

    iH = Dx * (T + 1) + (t - 1) * Dy + 1
    jH = Dx * t + 1
    P[iH:(iH + Dy - 1), jH:(jH + Dx - 1)] = H
end

μ_ϵ = zeros(Dx + T * (Dx + Dy))
μ_ϵ[1:Dx] .= μ0
for t in 1:T
    ib = t * Dx + 1
    μ_ϵ[ib:(ib + Dx - 1)] = b

    ic = Dx * (T + 1) + (t - 1) * Dy + 1
    μ_ϵ[ic:(ic + Dy - 1)] = c
end

Σ_ϵ = zeros(Dx + T * (Dx + Dy), Dx + T * (Dx + Dy))
Σ_ϵ[1:Dx, 1:Dx] .= Σ0
for t in 1:T
    iQ = t * Dx + 1
    Σ_ϵ[iQ:(iQ + Dx - 1), iQ:(iQ + Dx - 1)] = Q

    iR = Dx * (T + 1) + (t - 1) * Dy + 1
    Σ_ϵ[iR:(iR + Dy - 1), iR:(iR + Dy - 1)] = R
end

# Note (I - P)Z = ϵ and solve for Z ~ N(μ_Z, Σ_Z)
μ_Z = (I - P) \ μ_ϵ
Σ_Z = ((I - P) \ Σ_ϵ) / (I - P)'

Y = vcat(ys...)
I_x = 1:((T + 1) * Dx)
I_y = ((T + 1) * Dx + 1):((T + 1) * Dx + T * Dy)
μ_X = μ_Z[I_x] + Σ_Z[I_x, I_y] * (Σ_Z[I_y, I_y] \ (Y - μ_Z[I_y]))
Σ_X = Σ_Z[I_x, I_x] - Σ_Z[I_x, I_y] * (Σ_Z[I_y, I_y] \ Σ_Z[I_y, I_x])

# Trim near-zero entries from Σ_X
Σ_X[abs.(Σ_X) .< 1e-10] .= 0.0
println("Σ_X:")
display(sparse(Σ_X))

Λ_X = inv(Σ_X)
# Trim near-zero entries from Λ_X
Λ_X[abs.(Λ_X) .< 1e-8] .= 0
println("Λ_X:")
display(sparse(Λ_X))

# Compute sparsity level by dimension
n_sparse(D, T) = D^2 * (3T + 1)
n_dense(D, T) = (D * (T + 1))^2

println("Sparse entries: $(n_sparse(Dx, T))")
println("Dense entries: $(n_dense(Dx, T))")

# Analytic construction
# TODO: does this generalise Cartan matrices?
Λ_X_anal = zeros((T + 1) * Dx, (T + 1) * Dx)
Σ0_inv = inv(Σ0)
Q_inv = inv(Q)
R_inv = inv(R)

# Fill diagonal terms
for t in 0:T
    block = zeros(Dx, Dx)

    # Base term
    if t == 0
        block += Σ0_inv
    else
        block += Q_inv
    end

    # Observation term
    if t > 0
        block += H' * R_inv * H
    end

    # Next term
    if t < T
        block += A' * Q_inv * A
    end

    i = t * Dx + 1
    j = (t + 1) * Dx
    Λ_X_anal[i:j, i:j] = block
end

# Add secondary terms
secondary_term = -A' * Q_inv
for t in 1:T
    i = (t - 1) * Dx + 1
    j = t * Dx + 1
    Λ_X_anal[i:(i + Dx - 1), j:(j + Dx - 1)] = secondary_term
    Λ_X_anal[j:(j + Dx - 1), i:(i + Dx - 1)] = secondary_term'
end

Λ_X_anal = sparse(Λ_X_anal)

println("Inverse Error: $(norm(Λ_X - Λ_X_anal))")

# Compute Cholesky decomposition in closed form
L = zeros((T + 1) * Dx, (T + 1) * Dx)
S = Array(Λ_X_anal[1:Dx, 1:Dx])
Lii = cholesky((S + S') / 2 ).L
B = Array(Λ_X_anal[Dx+1:2Dx, 1:Dx])
L[1:Dx, 1:Dx] = Lii
for t in 1:T
    L_secondary = B * inv(Lii')  # even the first one of these is wrong
    L[t * Dx + 1:(t + 1) * Dx, (t - 1) * Dx + 1:t * Dx] = L_secondary
    S = Q_inv + H' * R_inv * H - Q_inv * A * inv(S) * A' * Q_inv
    if t < T
        S += A' * Q_inv * A
    end
    Lii = cholesky((S + S') / 2).L
    L[t * Dx + 1:(t + 1) * Dx, t * Dx + 1:(t + 1) * Dx] = Lii
end

println("Cholesky Error: $(norm(L * L' - Λ_X))")

# Test pre-conditioning
G_X = Gaussian(μ_X, Σ_X)
G_X_cond = L' * G_X

Σ_X_cond = G_X_cond.Σ
Σ_X_cond[abs.(Σ_X_cond) .< 1e-10] .= 0.0
println("Σ_X_cond:")
display(sparse(Σ_X_cond))
