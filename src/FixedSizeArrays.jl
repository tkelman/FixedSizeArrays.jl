__precompile__(true)
module FixedSizeArrays

using Compat

importall Base
import Base.LinAlg.chol!
import Base.LinAlg._chol!


# for 0.5 and 0.4 compat, use our own functor type
abstract Functor{N}

if VERSION < v"0.5.0-dev+1949"
    supertype(x) = super(x)
end

if VERSION < v"0.5.0-dev+698"
    macro pure(ex)
        esc(ex)
    end
else
    import Base: @pure
end

include("core.jl")
include("concrete_types.jl")

include("functors.jl")

#=
include("constructors.jl")

include("mapreduce.jl")
include("destructure.jl")
include("indexing.jl")
include("ops.jl")
include("expm.jl")
include("array_of_fixedsize.jl")
include("conversion.jl")
=#


export FixedArray
export FixedVector
export FixedMatrix
export Mat, Vec, Point

#=
export @fsa
export similar_type
export construct_similar

export unit
export normalize
export row
export column
export MatMulFunctor
export setindex
export @fslice
export destructure
=#

end
