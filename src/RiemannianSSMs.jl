module RiemannianSSMs

using LinearAlgebra
using Random
using StaticArrays
using PDMats

include("block_types.jl")
include("linalg.jl")
include("rhmc.jl")
include("rhmc_param.jl")
include("rhmc_poisson_param.jl")

include("models/variable_restoring_force.jl")
include("models/squared_landmark_distance.jl")
include("models/variable_restoring_force_param.jl")
include("models/squared_landmark_distance_param.jl")
include("models/van_der_pol_param.jl")
include("models/partial_linear_observation_param.jl")
include("models/random_walk_dynamics_param.jl")
include("models/poisson_observation_param.jl")
include("models/fitzhugh_nagumo_param.jl")
include("models/poisson_multi_channel_param.jl")

end
