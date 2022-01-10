#=
test_julia:
- Julia version: 1.6.5
- Author: tomyaacov
- Date: 2021-12-27
=#

# installing the module
#using Pkg
#Pkg.add(PackageSpec(url="https://github.com/johncwok/SpectralEnvelope.jl.git", rev="master"))
using SpectralEnvelope

f, se, eigvecs = spectral_envelope([2,3,4,3,2,3,4]; m = 3)
print(f)