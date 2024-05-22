using RotationMap: rotation
using LinearAlgebra: det, I
using ChainRulesTestUtils: test_rrule
using Test

isrotation(R) = det(R) ≈ 1 && R'R ≈ I

@testset "RotationMap.jl" begin
    for T in [Float32, Float64]
        A = randn(T, 9, 10)
        R = rotation(A)
        @test size(R) == (3, 3, 10)
        @test eltype(R) == eltype(A)
        @test all(isrotation(R[:,:,k]) for k in axes(R, 3))
        # slightly relax the relative tolerance for Float32
        test_rrule(rotation, A, rtol = √eps(T))
    end
end
