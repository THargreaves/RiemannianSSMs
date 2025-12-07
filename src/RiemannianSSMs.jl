module RiemannianSSMs

using LinearAlgebra
using Random
using StaticArrays
using PDMats

include("block_types.jl")
include("linalg.jl")
include("rhmc.jl")

include("models/variable_restoring_force.jl")
include("models/squared_landmark_distance.jl")

end
