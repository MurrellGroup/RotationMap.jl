# RotationMap

[![Build Status](https://github.com/MurrellGroup/RotationMap.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/RotationMap.jl/actions/workflows/CI.yml?query=branch%3Amain)


This package implements the SVDO+ function, as described by Levinson, Jake, et
al. in "An analysis of svd for deep rotation estimation." Advances in Neural
Information Processing Systems 33 (2020): 22554-22565.

The `rotation` function implements the SVDO+ function. It takes 9-dimensional
representation vectors and maps each one to a 3×3 rotation matrix. This package
also supports batching, CUDA-acceleration and automatic differentiation,
facilitated by ChainRulesCore.jl.

```julia
using RotationMap: rotation

# Initialize a batch of 9-dimensional representation vectors.
A = randn(Float32, 9, 12)

# Use the `rotation` function to map each vector to a 3×3 rotation matrix.
R = rotation(A)
```
