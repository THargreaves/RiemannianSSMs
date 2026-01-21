"""
    MCRHMC

Modified Cholesky Riemann Manifold Hamiltonian Monte Carlo.

Implementation of the modified Cholesky decomposition from arXiv:1612.04093v2
(Kleppe, "Modified Cholesky RMHMC: Exploiting Sparsity for Fast Sampling").

This submodule provides:
- Sparse modified LDL^T factorization with smooth regularization
- Efficient adjoint computation for automatic differentiation
- Exploitation of sparsity patterns for high-dimensional problems
"""
module MCRHMC

using SparseArrays
using LinearAlgebra

include("sabs.jl")
include("sparse_ldl_types.jl")
include("sparse_ldl_symbolic.jl")
include("sparse_ldl_numeric.jl")
include("sparse_ldl_adjoint.jl")

export sabs, sabs_deriv, sabs_with_deriv

export SparseLDLSymbolic, SparseLDLFactor, SparseLDLAdjoint

export analyze_symbolic, nnz_L

export modified_cholesky, modified_cholesky!

export get_L,
    get_D, reconstruct, compute_regularization_sensitivities, diagnose_regularization

export modified_cholesky_adjoint!, modified_cholesky_pullback, logdet_pullback

export quadform, quadform_pullback, hamiltonian_grad, sparse_trace_symmetric

end # module
