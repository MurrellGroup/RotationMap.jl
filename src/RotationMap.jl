module RotationMap

using ChainRulesCore: ChainRulesCore, unthunk
using LinearAlgebra: svd, det
using NNlib: batched_mul, batched_transpose
using CUDA: CUDA

checkdimension(A::AbstractArray) =
    size(A, 1) == 9 || throw(DimensionMismatch("The input array should have 9 rows"))

function rotation(A::AbstractMatrix)
    checkdimension(A)
    F = eltype(A)
    A = reshape(A, 3, 3, :)
    U, _, V = batchedsvd(A)
    ν = sign.(batcheddet(A))
    # U * diagm([1, 1, ν]) * V'
    U[:,3,:] .*= reshape(ν, 1, :)
    batched_mul(U, batched_transpose(V))
end

function rotation(A::AbstractArray)
    checkdimension(A)
    rotation(reshape(A, 9, :))
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

ChainRulesCore.rrule(::typeof(rotation), A::AbstractArray) = ChainRulesCore.rrule(rotation, A)

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

function bachedsvd(A::CUDA.CuArray{<: Union{Float32, Float64}, 3})
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
