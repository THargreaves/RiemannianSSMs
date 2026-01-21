module RiemannianSSMs

using LinearAlgebra
using Random
using StaticArrays
using PDMats
using SparseArrays
using Distributions
using SpecialFunctions: logfactorial
using Statistics: median

# Include MCRHMC submodule first
include("MCRHMC/MCRHMC.jl")

include("block_types.jl")
include("linalg.jl")

# New unified interface
include("exponential_families.jl")
include("model_interface.jl")
include("rhmc_unified.jl")

# Model implementations
include("models/van_der_pol.jl")
include("models/random_walk_poisson.jl")
include("models/spatiotemporal_bernoulli.jl")

# Kleppe's observed Hessian metric (uses MCRHMC submodule)
include("kleppe_vdp.jl")

# Adaptation for Riemannian HMC
include("adaptation.jl")

end
