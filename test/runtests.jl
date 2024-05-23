using RotationMap: rotation
using LinearAlgebra: det, I
using ChainRulesTestUtils: test_rrule
using CUDA
using Test

isrotation(R) = det(R) > 0 && R'R ≈ I

@testset "RotationMap.jl" begin
    for T in [Float32, Float64]
        # relax the relative tolerance for Float32
        rtol = ∛eps(T)

        A = randn(T, 9, 4)
        R = rotation(A)
        @test size(R) == (3, 3, 4)
        @test eltype(R) == eltype(A)
        @test all(isrotation(R[:,:,k]) for k in axes(R, 3))
        test_rrule(rotation, A; rtol)

        A = randn(T, 9, 4, 5)
        R = rotation(A)
        @test size(R) == (3, 3, 4, 5)
        @test eltype(R) == eltype(A)
        @test all(isrotation(R[:,:,k,l]) for k in axes(R, 3) for l in axes(R, 4))
        test_rrule(rotation, A; rtol)
    end
end
