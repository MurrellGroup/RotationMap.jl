module RotationMap

using ChainRulesCore: ChainRulesCore, unthunk
using LinearAlgebra: svd, det
using NNlib: batched_mul, batched_transpose
using CUDA: CUDA, CUSOLVER

checkdimension(A::AbstractArray) =
    size(A, 1) == 9 || throw(DimensionMismatch("The input array should have 9 rows"))

"""
    rotation(A::AbstractArray)

Map 9-dimensional vectors to proper 3×3 rotation matrices.

# Example

```jldoctest
julia> using Random

julia> A = randn(Xoshiro(0), 9, 2)
9×2 Matrix{Float64}:
 -0.231909   -0.86473
  0.94039    -0.269885
  0.596762    0.578004
  1.99782     1.16034
 -0.0515618   0.287888
  1.17224    -0.441193
 -1.69681     0.134576
 -2.11615     0.237598
  0.558366    0.100588

julia> using RotationMap: rotation

julia> R = rotation(A)
3×3×2 Array{Float64, 3}:
[:, :, 1] =
  0.42337    0.838204  -0.343761
 -0.209441  -0.27861   -0.937289
 -0.881415   0.468817   0.0575993

[:, :, 2] =
 -0.187231   0.981467  0.0408275
 -0.411524  -0.116109  0.903973
  0.89196    0.15245   0.425637

julia> R[:,:,1]'R[:,:,1]
3×3 Matrix{Float64}:
  1.0          -1.11022e-16  -3.88578e-16
 -1.11022e-16   1.0           9.71445e-17
 -3.88578e-16   9.71445e-17   1.0
```

Levinson, Jake, et al. "An analysis of svd for deep rotation estimation."
Advances in Neural Information Processing Systems 33 (2020): 22554-22565.
"""
function rotation(A::AbstractArray)
    checkdimension(A)
    reshape(rotation(reshape(A, 9, :)), 3, 3, Base.tail(size(A))...)
end

function rotation(A::AbstractMatrix)
    checkdimension(A)
    A = reshape(A, 3, 3, :)
    U, _, V = batchedsvd(A)
    ν = sign.(batcheddet(A))
    # U * diagm([1, 1, ν]) * V'
    U[:,3,:] .*= reshape(ν, 1, :)
    batched_mul(U, batched_transpose(V))
end

function ChainRulesCore.rrule(::typeof(rotation), A::AbstractMatrix)
    checkdimension(A)
    A = reshape(A, 3, 3, :)
    U, s, V = batchedsvd(A)
    ν = sign.(batcheddet(A))
    function rotation_pullback(R̄)
        R̄ = unthunk(R̄)
        # ∂A = U * (G .* (K - K')) * diagm([1, 1, ν]) * V'
        # where K = U'R̄*V and
        #       G[i,j] = i == j           ? 0 :
        #                i <= 2 && j <= 2 ? inv(s[i] +   s[j]) :
        #                                   inv(s[i] + ν*s[j])
        G = zero(A)
        G[1,2,:] .= inv.(s[2,:] .+ s[1,:])
        G[1,3,:] .= inv.(s[3,:] .+ ν .* s[1,:])
        G[2,1,:] .= inv.(s[1,:] .+ s[2,:])
        G[2,3,:] .= inv.(s[3,:] .+ ν .* s[2,:])
        G[3,1,:] .= inv.(s[1,:] .+ ν .* s[3,:])
        G[3,2,:] .= inv.(s[2,:] .+ ν .* s[3,:])
        K = batched_mul(batched_mul(batched_transpose(U), R̄), V)
        H = G .* (K .- batched_transpose(K))
        H[:,3,:] .*= reshape(ν, 1, :)
        ∂A = batched_mul(batched_mul(U, H), batched_transpose(V))
        ChainRulesCore.NoTangent(), reshape(∂A, 9, :)
    end
    # U * diagm([1, 1, ν]) * V'
    U[:,3,:] .*= reshape(ν, 1, :)
    batched_mul(U, batched_transpose(V)), rotation_pullback
end

function ChainRulesCore.rrule(::typeof(rotation), A::AbstractArray)
    checkdimension(A)
    R, pb = ChainRulesCore.rrule(rotation, reshape(A, 9, :))
    function rotation_pullback(R̄)
        t, ∂A = pb(reshape(R̄, 3, 3, :))
        t, reshape(∂A, size(A))
    end
    reshape(R, 3, 3, Base.tail(size(A))...), rotation_pullback
end

# fallback for development
function batchedsvd(A::AbstractArray{<: Union{Float32, Float64}, 3})
    @assert size(A, 1) == 3 && size(A, 2) == 3
    U = similar(A)
    s = similar(A, 3, size(A, 3))
    V = similar(A)
    for k in axes(A, 3)
        U[:,:,k], s[:,k], V[:,:,k] = svd(A[:,:,k])
    end
    U, s, V
end

function batcheddet(A::AbstractArray{<: Union{Float32, Float64}, 3})
    @assert size(A, 1) == 3 && size(A, 2) == 3
    out = similar(A, size(A, 3))
    for k in axes(A, 3)
        out[k] = det(A[:,:,k])
    end
    out
end

function batchedsvd(A::CUDA.CuArray{<: Union{Float32, Float64}, 3})
    @assert size(A, 1) == 3 && size(A, 2) == 3
    CUSOLVER.gesvdj!('V', copy(A))
end

function batcheddet(A::CUDA.CuArray{<: Union{Float32, Float64}, 3})
    @assert size(A, 1) == 3 && size(A, 2) == 3
    function kernel!(out, A)
        k = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
        if k ≤ size(A, 3)
            a11 = A[1,1,k]; a12 = A[1,2,k]; a13 = A[1,3,k]
            a21 = A[2,1,k]; a22 = A[2,2,k]; a23 = A[2,3,k]
            a31 = A[3,1,k]; a32 = A[3,2,k]; a33 = A[3,3,k]
            out[k] = a11 * (a22 * a33 - a23 * a32) - a12 * (a21 * a33 - a23 * a31) + a13 * (a21 * a32 - a22 * a31)
        end
        return
    end
    n = size(A, 3)
    out = similar(A, n)
    kernel! = CUDA.@cuda launch = false kernel!(out, A)
    config = CUDA.launch_configuration(kernel!.fun)
    threads = min(n, config.threads)
    kernel!(out, A; threads, blocks = cld(n, threads))
    out
end

end # module
