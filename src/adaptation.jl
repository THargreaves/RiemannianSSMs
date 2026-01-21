"""
Rate-based regularization adaptation for Riemannian HMC with modified Cholesky.

Implements the u-adaptation scheme from Kleppe (arXiv:1612.04093v2) that adjusts
regularization parameters based on fixed-point iteration divergence rates.
"""

using AdvancedHMC
using AdvancedHMC.Adaptation: AbstractAdaptor, StepSizeAdaptor, getϵ
using Setfield: @set
using Statistics: median

export RiemannianRegularizationAdaptor, RiemannianHMCAdaptor
export getu, update_regularization!

"""
    RiemannianRegularizationAdaptor{T}

Rate-based adaptor for regularization parameters in modified Cholesky factorization.

Tracks fixed-point iteration divergences over windows and adjusts u values
proportionally based on accumulated sensitivities.

# Fields
- `u`: Current regularization parameters (length n)
- `γ`: Adaptation scaling factor (default 0.05)
- `δ`: Target divergence rate (default 0.05)
- `window_size`: Number of samples per adaptation window
- `recent_divergences`: Count of divergent steps in current window
- `recent_steps`: Count of samples in current window
- `accumulated_sensitivities`: Sum of sensitivities during divergent steps
"""
mutable struct RiemannianRegularizationAdaptor{T<:AbstractFloat} <: AbstractAdaptor
    u::Vector{T}
    γ::T
    δ::T
    window_size::Int

    recent_divergences::Int
    recent_steps::Int
    total_leapfrog_steps::Int
    accumulated_sensitivities::Vector{T}
end

function RiemannianRegularizationAdaptor(
    u::Vector{T}; γ::T=T(0.05), δ::T=T(0.05), window_size::Int=50
) where {T<:AbstractFloat}
    return RiemannianRegularizationAdaptor{T}(
        copy(u), γ, δ, window_size, 0, 0, 0, zeros(T, length(u))
    )
end

function Base.show(io::IO, adaptor::RiemannianRegularizationAdaptor)
    return print(
        io,
        "RiemannianRegularizationAdaptor(γ=",
        adaptor.γ,
        ", δ=",
        adaptor.δ,
        ", window_size=",
        adaptor.window_size,
        ", u_range=",
        extrema(adaptor.u),
        ")",
    )
end

"""
    getu(adaptor::RiemannianRegularizationAdaptor)

Get the current regularization parameters.
"""
AdvancedHMC.Adaptation.getu(adaptor::RiemannianRegularizationAdaptor) = adaptor.u

"""
    adapt!(adaptor::RiemannianRegularizationAdaptor, θ, tstat, h)

Update regularization parameters based on divergence rate.

Accumulates divergence counts and sensitivities over the window, then at
window boundaries checks if the divergence rate exceeds the target δ.
If so, increases u values proportionally to their sensitivities.
"""
function AdvancedHMC.Adaptation.adapt!(
    adaptor::RiemannianRegularizationAdaptor{T}, θ, tstat, h
) where {T}
    adaptor.recent_steps += 1

    # Get leapfrog step count from this trajectory (for proper rate calculation)
    n_leapfrog_steps = get(tstat, :n_steps, 1)
    divergent_steps = get(tstat, :divergent_steps, 0)
    sens = get(tstat, :accumulated_sensitivities, nothing)

    # Track total leapfrog steps for proper rate calculation
    adaptor.total_leapfrog_steps += n_leapfrog_steps

    if divergent_steps > 0
        adaptor.recent_divergences += divergent_steps
        if sens !== nothing && !isempty(sens)
            if length(adaptor.accumulated_sensitivities) != length(sens)
                adaptor.accumulated_sensitivities = zeros(T, length(sens))
            end
            # Only accumulate finite sensitivities
            @inbounds for i in eachindex(sens)
                if isfinite(sens[i])
                    adaptor.accumulated_sensitivities[i] += sens[i]
                end
            end
        end
    end

    if adaptor.recent_steps >= adaptor.window_size
        # Rate is divergent steps / total leapfrog steps (not samples!)
        rate = if adaptor.total_leapfrog_steps > 0
            adaptor.recent_divergences / adaptor.total_leapfrog_steps
        else
            zero(T)
        end
        sens_sum = sum(adaptor.accumulated_sensitivities)

        # Log window summary with stats
        sens = adaptor.accumulated_sensitivities
        u = adaptor.u
        @info "RRA window" rate target=adaptor.δ divergences=adaptor.recent_divergences leapfrog_steps=adaptor.total_leapfrog_steps sens_stats=(
            if isempty(sens)
                "empty"
            else
                (min=minimum(sens), med=median(sens), max=maximum(sens))
            end
        ) u_stats=(min=minimum(u), med=median(u), max=maximum(u))

        if rate > adaptor.δ && sens_sum > 0
            η = adaptor.γ * (rate - adaptor.δ) / adaptor.δ
            s_normalized = adaptor.accumulated_sensitivities ./ sens_sum

            # Find indices with largest sensitivities and largest u values
            top_sens_idx = partialsortperm(
                s_normalized, 1:min(3, length(s_normalized)); rev=true
            )
            top_u_idx = partialsortperm(adaptor.u, 1:min(3, length(adaptor.u)); rev=true)

            adaptor.u .*= exp.(η .* s_normalized)
            @info "Regularization updated" η top_sens_indices=top_sens_idx top_sens_values=s_normalized[top_sens_idx] top_u_indices=top_u_idx u_at_top_u=adaptor.u[top_u_idx] u_new_stats=(
                min=minimum(adaptor.u), med=median(adaptor.u), max=maximum(adaptor.u)
            )
        end

        adaptor.recent_divergences = 0
        adaptor.recent_steps = 0
        adaptor.total_leapfrog_steps = 0
        fill!(adaptor.accumulated_sensitivities, zero(T))
    end

    return nothing
end

AdvancedHMC.Adaptation.reset!(adaptor::RiemannianRegularizationAdaptor) = nothing
function AdvancedHMC.Adaptation.initialize!(
    adaptor::RiemannianRegularizationAdaptor, n_adapts::Int
)
    nothing
end
AdvancedHMC.Adaptation.finalize!(adaptor::RiemannianRegularizationAdaptor) = nothing

"""
    RiemannianHMCAdaptor{SSA,RRA}

Composite adaptor combining step size adaptation and regularization adaptation.

# Fields
- `ssa`: Step size adaptor (e.g., NesterovDualAveraging)
- `rra`: Regularization adaptor (RiemannianRegularizationAdaptor)
"""
struct RiemannianHMCAdaptor{SSA<:StepSizeAdaptor,RRA<:RiemannianRegularizationAdaptor} <:
       AbstractAdaptor
    ssa::SSA
    rra::RRA
end

function Base.show(io::IO, adaptor::RiemannianHMCAdaptor)
    return print(io, "RiemannianHMCAdaptor(ssa=", adaptor.ssa, ", rra=", adaptor.rra, ")")
end

AdvancedHMC.Adaptation.getϵ(adaptor::RiemannianHMCAdaptor) = getϵ(adaptor.ssa)
AdvancedHMC.Adaptation.getu(adaptor::RiemannianHMCAdaptor) = getu(adaptor.rra)

function AdvancedHMC.Adaptation.adapt!(adaptor::RiemannianHMCAdaptor, θ, tstat, h)
    AdvancedHMC.Adaptation.adapt!(adaptor.ssa, θ, tstat, h)
    AdvancedHMC.Adaptation.adapt!(adaptor.rra, θ, tstat, h)
    return nothing
end

function AdvancedHMC.Adaptation.reset!(adaptor::RiemannianHMCAdaptor)
    AdvancedHMC.Adaptation.reset!(adaptor.ssa)
    AdvancedHMC.Adaptation.reset!(adaptor.rra)
    return nothing
end

function AdvancedHMC.Adaptation.initialize!(adaptor::RiemannianHMCAdaptor, n_adapts::Int)
    AdvancedHMC.Adaptation.initialize!(adaptor.ssa, n_adapts)
    AdvancedHMC.Adaptation.initialize!(adaptor.rra, n_adapts)
    return nothing
end

function AdvancedHMC.Adaptation.finalize!(adaptor::RiemannianHMCAdaptor)
    AdvancedHMC.Adaptation.finalize!(adaptor.ssa)
    AdvancedHMC.Adaptation.finalize!(adaptor.rra)
    return nothing
end

# Extend AdvancedHMC's update to handle RiemannianHMCAdaptor
function AdvancedHMC.update(τ::AdvancedHMC.Trajectory, adaptor::RiemannianHMCAdaptor)
    integrator = AdvancedHMC.update_nom_step_size(τ.integrator, getϵ(adaptor))
    return @set τ.integrator = integrator
end
