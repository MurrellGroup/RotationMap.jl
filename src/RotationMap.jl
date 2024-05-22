module RotationMap

using ChainRulesCore: ChainRulesCore, unthunk
using LinearAlgebra: svd, det
using NNlib: batched_mul, batched_transpose

checkdimension(A::AbstractArray) =
    size(A, 1) == 9 || throw(DimensionMismatch("The input array should have 9 rows"))

function rotation(A::AbstractMatrix)
    checkdimension(A)
    A = reshape(A, 3, 3, :)
    U, _, V = batchedsvd(A)
    ν = batchedsigndet(A)
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
    ν = batchedsigndet(A)
    function rotation_pullback(R̄)
        R̄ = unthunk(R̄)
        # ∂A = U * (G .* (K - K')) * diagm([1, 1, ν]) * V'
        # where K = U'R̄*V and
        #       G[i,j] = i == j ? 0 : inv(s[i] + ν*s[j])
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
    U = similar(A)
    s = similar(A, 3, size(A, 3))
    V = similar(A)
    for k in axes(A, 3)
        U[:,:,k], s[:,k], V[:,:,k] = svd(A[:,:,k])
    end
    U, s, V
end

function batchedsigndet(A::AbstractArray{<: Union{Float32, Float64}, 3})
    d = similar(A, size(A, 3))
    for k in axes(A, 3)
        d[k] = ifelse(det(A[:,:,k]) > 0, 1, -1)
    end
    d
end

end # module
