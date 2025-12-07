using Distributions
using LinearAlgebra
using Plots
using ProgressMeter
using Random
using StaticArrays

using RiemannianSSMs

SEED = 4
rng = MersenneTwister(SEED)

K = 100
δt = 1.0

μ0 = @SVector [0.0, 0.0, 0.0, 0.0]
Σ0 = Diagonal(@SVector([0.5, 0.5, 0.1, 0.1]))

prior = MvNormal(μ0, Σ0)

α = 0.01
β = 0.1
γ = 0.005
σ_p = 0.1
σ_v = 0.5

dyn = VariableRestoringForceDynamics(α, β, γ, σ_p, σ_v, δt)

# Try moving these away from trajectory to resolve numerical stability issues
a1, b1 = -1.0, -3.0
a2, b2 = 5.0, -1.5
σ1 = 0.5
σ2 = 0.5

obs = TwoLandmarkMeasurementModel(a1, b1, a2, b2, σ1, σ2)

struct SSM{PT,DY,OM}
    prior::PT
    dyn::DY
    sensor::OM
end
ssm = SSM(prior, dyn, obs);

function simulate(rng::AbstractRNG, ssm, K::Int)
    zs = Vector{SVector{4,Float64}}(undef, K)
    ys = Vector{SVector{2,Float64}}(undef, K)

    for k in 1:K
        if k == 1
            z = SVector{4,Float64}(rand(rng, ssm.prior))
        else
            z = f(ssm.dyn, zs[k - 1]) + rand(rng, MvNormal(zeros(4), calc_Q(ssm.dyn)))
        end
        zs[k] = z

        y = h(ssm.sensor, z) + rand(rng, MvNormal(zeros(2), calc_R(ssm.sensor)))
        ys[k] = y
    end

    return zs, ys
end
zs_true, ys = simulate(rng, ssm, K);

zs_true = BlockVector{Float64,4}(zs_true)

lf_params = LeapfrogParams{Float64}(0.2, 1e-6, 1000)

N_samples = 500
N_burnin = 0
n_steps = 50

function sample(
    rng::AbstractRNG,
    ssm::SSM,
    z_init::BlockVector{Float64,4},
    ys::Vector{SVector{2,Float64}},
    N_samples::Int,
    N_burnin::Int,
    n_steps::Int,
    lf_params::LeapfrogParams{Float64};
    progress=true,
    track_divergences=false,
)
    zs_samples = Vector{BlockVector{Float64,4}}(undef, N_samples)
    zs_curr = copy(z_init)
    n_accept = 0
    n_divergent = 0
    K = length(ys)

    # Track divergent steps and initial states
    initial_states = BlockVector{Float64,4}[]
    initial_momenta = BlockVector{Float64,4}[]
    divergent_states = BlockVector{Float64,4}[]
    divergent_momenta = BlockVector{Float64,4}[]

    prog = Progress(N_samples; enabled=progress)
    for i in 1:N_samples
        G = calc_G(zs_curr, ssm)
        G_chol = cholesky(G)
        U = BlockVector{Float64,4}([@SVector randn(rng, 4) for k in 1:K])
        ps_curr = G_chol.U.data' * U
        H_curr = calc_hamiltonian(zs_curr, ps_curr, ys, ssm)
        zs_new = copy(zs_curr)
        ps_new = copy(ps_curr)

        if track_divergences
            push!(initial_states, copy(zs_new))
            push!(initial_momenta, copy(ps_new))
        end

        early_reject = false
        for step in 1:n_steps
            zs_new, ps_new, diverged = glf_step(zs_new, ps_new, ys, ssm, lf_params)

            if diverged
                if track_divergences
                    push!(divergent_states, copy(zs_new))
                    push!(divergent_momenta, copy(ps_new))
                end
                early_reject = true
                break
            end
        end

        if early_reject
            zs_samples[i] = copy(zs_curr)
            n_divergent += 1
            next!(prog)
            continue
        end

        # Accept or reject
        H_new = calc_hamiltonian(zs_new, ps_new, ys, ssm)

        if log(rand(rng)) < H_curr - H_new
            zs_curr = zs_new
            n_accept += 1
        end

        zs_samples[i] = copy(zs_curr)
        next!(prog)
    end

    println("Finishing sampling with acceptance rate $(n_accept / N_samples)")
    println("Proportion of divergent transitions: $(n_divergent / N_samples)")

    return zs_samples, initial_states, initial_momenta, divergent_states, divergent_momenta
end

z_init = zs_true

zs_samples, initial_states, initial_momenta, divergent_states, divergent_momenta =
    sample(rng, ssm, z_init, ys, N_samples, N_burnin, n_steps, lf_params;
           track_divergences=true);

# Extract trajectories from states
function state_to_trajectory(zs)
    xs = [zs.blocks[k][1] for k in 1:length(zs.blocks)]
    ys = [zs.blocks[k][2] for k in 1:length(zs.blocks)]
    return xs, ys
end

initial_trajectories = [state_to_trajectory(zs) for zs in initial_states]
divergent_trajectories = [state_to_trajectory(zs) for zs in divergent_states]

# Calculate momentum norms for position and velocity parts
function calculate_momentum_norms(ps_vec)
    position_momentum_norms = Float64[]
    velocity_momentum_norms = Float64[]

    for ps in ps_vec
        for k in 1:length(ps.blocks)
            # Position momentum (first 2 components)
            p_pos = sqrt(ps.blocks[k][1]^2 + ps.blocks[k][2]^2)
            # Velocity momentum (last 2 components)
            p_vel = sqrt(ps.blocks[k][3]^2 + ps.blocks[k][4]^2)

            push!(position_momentum_norms, p_pos)
            push!(velocity_momentum_norms, p_vel)
        end
    end

    return position_momentum_norms, velocity_momentum_norms
end

initial_p_pos, initial_p_vel = calculate_momentum_norms(initial_momenta)
divergent_p_pos, divergent_p_vel = calculate_momentum_norms(divergent_momenta)

# Calculate state space norms
function calculate_state_norms(zs_vec)
    position_norms = Float64[]
    velocity_norms = Float64[]

    for zs in zs_vec
        for k in 1:length(zs.blocks)
            x, y, vx, vy = zs.blocks[k]
            push!(position_norms, sqrt(x^2 + y^2))
            push!(velocity_norms, sqrt(vx^2 + vy^2))
        end
    end

    return position_norms, velocity_norms
end

initial_pos_norms, initial_vel_norms = calculate_state_norms(initial_states)
divergent_pos_norms, divergent_vel_norms = calculate_state_norms(divergent_states)

# Statistics
println("\n=== Divergence Statistics ===")
println("Total MCMC samples: $N_samples")
println("Initial states tracked: $(length(initial_states))")
println("Divergent steps: $(length(divergent_states))")
println("Divergence rate: $(length(divergent_states) / N_samples * 100)%")

if !isempty(divergent_states)
    println("\nState space statistics:")
    println("Initial states:")
    println("  Position norm - mean: $(mean(initial_pos_norms)), std: $(std(initial_pos_norms))")
    println("  Velocity norm - mean: $(mean(initial_vel_norms)), std: $(std(initial_vel_norms))")
    println("Divergent states:")
    println("  Position norm - mean: $(mean(divergent_pos_norms)), std: $(std(divergent_pos_norms))")
    println("  Velocity norm - mean: $(mean(divergent_vel_norms)), std: $(std(divergent_vel_norms))")

    println("\nMomentum statistics:")
    println("Initial momenta:")
    println("  Position momentum - mean: $(mean(initial_p_pos)), std: $(std(initial_p_pos))")
    println("  Velocity momentum - mean: $(mean(initial_p_vel)), std: $(std(initial_p_vel))")
    println("Divergent momenta:")
    println("  Position momentum - mean: $(mean(divergent_p_pos)), std: $(std(divergent_p_pos))")
    println("  Velocity momentum - mean: $(mean(divergent_p_vel)), std: $(std(divergent_p_vel))")

    # Create 4-panel figure
    # Panel 1: Spatial trajectories
    p1 = plot(
        title="Leapfrog Integration Starting Points and Divergent Steps",
        xlabel="x position",
        ylabel="y position",
        legend=:topright
    )

    # Plot true trajectory
    true_xs = [z[1] for z in zs_true.blocks]
    true_ys = [z[2] for z in zs_true.blocks]
    plot!(p1, true_xs, true_ys,
          label="True trajectory",
          color=:black,
          linewidth=2,
          linestyle=:dash)

    # Plot landmarks
    scatter!(p1, [a1, a2], [b1, b2],
             label="Landmarks",
             color=:gold,
             markersize=10,
             markershape=:star5)

    # Plot initial trajectories
    for (i, (xs, ys)) in enumerate(initial_trajectories)
        plot!(p1, xs, ys,
              label=(i == 1 ? "Initial states" : ""),
              color=:blue,
              alpha=0.2,
              linewidth=0.5)
    end

    # Plot divergent trajectories
    for (i, (xs, ys)) in enumerate(divergent_trajectories)
        plot!(p1, xs, ys,
              label=(i == 1 ? "Divergent steps ($(length(divergent_states)))" : ""),
              color=:red,
              alpha=0.6,
              linewidth=1.5)
    end

    # Panel 2: State space (position vs velocity)
    p2 = scatter(
        initial_pos_norms,
        initial_vel_norms,
        xlabel="Position norm",
        ylabel="Velocity norm",
        title="State Space: Initial vs Divergent",
        label="Initial states",
        color=:blue,
        alpha=0.5,
        markersize=3
    )
    scatter!(p2, divergent_pos_norms, divergent_vel_norms,
             label="Divergent steps",
             color=:red,
             alpha=0.7,
             markersize=4)

    # Panel 3: Momentum space (position momentum vs velocity momentum)
    p3 = scatter(
        initial_p_pos,
        initial_p_vel,
        xlabel="Position momentum norm",
        ylabel="Velocity momentum norm",
        title="Momentum Space: Initial vs Divergent",
        label="Initial momenta",
        color=:blue,
        alpha=0.5,
        markersize=3
    )
    scatter!(p3, divergent_p_pos, divergent_p_vel,
             label="Divergent momenta",
             color=:red,
             alpha=0.7,
             markersize=4)

    # Panel 4: Combined phase space view
    p4 = scatter(
        initial_pos_norms,
        initial_p_pos,
        xlabel="Position norm",
        ylabel="Position momentum norm",
        title="Phase Space: Position & Position Momentum",
        label="Initial",
        color=:blue,
        alpha=0.5,
        markersize=3
    )
    scatter!(p4, divergent_pos_norms, divergent_p_pos,
             label="Divergent",
             color=:red,
             alpha=0.7,
             markersize=4)

    combined = plot(p1, p2, p3, p4, layout=(4, 1), size=(800, 1600))
    display(combined)
else
    println("\nNo divergences detected.")
end
