module F2

if VERSION < v"0.5.0-dev+1949"
    supertype(x) = super(x)
end

#-------------------------------------------------------------------------------
# Abstract type hierarchy

"""
    FixedArray{T,D}

Super type for all fixed size arrays of eltype `T` and dimension `D`.

The extents of the fixed dimensions aren't specified here due to technical
limitations.  (It'd have to be an `NTuple{Int,D}`, but `TypeVar`s representing
the extent of the dimensions in `FixedArray` subtypes can't be placed into this
tuple when subtyping.)
"""
abstract FixedArray{T,D} <: AbstractArray{T,D}

# Define a subtype of FixedArray for each dimension.  Ideally we'd have the
# fixed size type parameters in FixedArray itself, but it's not clear how we
# can actually do this in a good way since type constructors are fairly
# restricted.
abstract FixedArray1{N,T}       <: FixedArray{T,1}
abstract FixedArray2{N,M,T}     <: FixedArray{T,2}
abstract FixedArray3{N,M,P,T}   <: FixedArray{T,3}
abstract FixedArray4{N,M,P,Q,T} <: FixedArray{T,4}
# Is there any point going more dimensions than MAX_TUPLE_DEPTH?
abstract FixedArray5{N,M,P,Q,R,T} <: FixedArray{T,5}

typealias FixedVector FixedArray1
typealias FixedMatrix FixedArray2

abstract FixedVectorNoTuple{N,T} <: FixedVector{N,T}


abstract FixedArrayN{SZ,T,D} <: FixedArray{T,D}

# TEST FIXME
immutable Vecc{SZ,T,Storage} <: FixedArrayN{SZ,T,1}
    values::Storage
end

function Vecc(args...)
    vals = promote(args...)
    Vecc{(length(vals),),eltype(vals),typeof(vals)}(vals)
end

@inline Base.size{A<:FixedArrayN}(::Type{A}) = size(supertype(A))
@inline Base.size{SZ,T,D}(::Type{FixedArrayN{SZ,T,D}}) = SZ


#-------------------------------------------------------------------------------
# Fixed size linear algebra types
"Canonical length-`N` vector with element type `T`"
immutable Vec{N,T} <: FixedVector{N,T}
    values::NTuple{N,T}
end

"Canonical N×M matrix with element type `T`"
immutable Mat{N,M,T} <: FixedMatrix{N,M,T}
    values::NTuple{M,NTuple{N,T}}
end

"Canonical rank three tensor with element type `T`"
immutable Ar3{N,M,P,T} <: FixedArray3{N,M,P,T}
    values::NTuple{P,NTuple{M,NTuple{N,T}}}
end

"Canonical rank four tensor with element type `T`"
immutable Ar4{N,M,P,Q,T} <: FixedArray4{N,M,P,Q,T}
    values::NTuple{Q,NTuple{P,NTuple{M,NTuple{N,T}}}}
end

"Canonical rank five tensor with element type `T`"
immutable Ar5{N,M,P,Q,R,T} <: FixedArray5{N,M,P,Q,R,T}
    values::NTuple{R,NTuple{Q,NTuple{P,NTuple{M,NTuple{N,T}}}}}
end


"""
An element of an `N`-dimensional Cartesian space with element type `T`
"""
immutable Point{N,T} <: FixedVector{N,T}
    values::NTuple{N,T}
end

#-------------------------------------------------------------------------------
# Constructors
Vec() = Vec{0,Bool}(())
Vec(args...) = Vec(promote(args...))

macro Mat(expr)
    if expr.head == :vcat
        rows = expr.args
        N = length(rows)
        rowlens = map(r->length(r.args), rows)
        M = rowlens[1]
        if any(M .!= rowlens)
            throw(DimensionMismatch("Row lengths should match"))
        end
        cols = [Expr(:tuple) for _=1:M]
        for row in rows
            @assert row.head == :row
            for (i,a) in enumerate(row.args)
                push!(cols[i].args, a)
            end
        end
        cols = Expr(:tuple, cols...)
        quote
            Mat($cols)
        end
    else
        error("Argh")
    end
end

function Ar5{T}(A::AbstractArray{T,5})
    reinterpret(Ar5{size(A)...,T}, vec(A))[1]
end

function Ar4{T}(A::AbstractArray{T,4})
    reinterpret(Ar4{size(A)...,T}, vec(A))[1]
end

#-------------------------------------------------------------------------------
# Primary interface
@inline Base.Tuple(A::FixedArray) = A.values


#-------------------------------------------------------------------------------
# Implement parts of the AbstractArray interface

# Get the fixed-dimension subtype of a FixedArray
@generated function fsa_abstract1{A<:FixedArray}(::Type{A})
    T = A
    while supertype(T).name.name != :FixedArray
       T = supertype(T)
    end
    :($T)
end

@inline Base.size{A<:FixedArray}(::Type{A}) = size(supertype(A))
@inline Base.size{A<:FixedArray}(a::A) = size(A)

@inline Base.size{N,T}(::Type{FixedArray1{N,T}}) = (N,)
@inline Base.size{N,M,T}(::Type{FixedArray2{N,M,T}}) = (N,M)
@inline Base.size{N,M,P,Q,T}(::Type{FixedArray4{N,M,P,Q,T}}) = (N,M,P,Q)
@inline Base.size{N,M,P,Q,R,T}(::Type{FixedArray5{N,M,P,Q,R,T}}) = (N,M,P,Q,R)

Base.linearindexing{A<:FixedArray}(::Type{A}) = Base.LinearFast()

# Multidimensional indexing
@inline Base.getindex(A::FixedArray1, i::Int) = Tuple(A)[i]
@inline Base.getindex(A::FixedArray2, i::Int, j::Int) = Tuple(A)[j][i]
@inline Base.getindex(A::FixedArray4, i::Int, j::Int, p::Int, q::Int) = Tuple(A)[q][p][j][i]
@inline Base.getindex(A::FixedArray5, i::Int, j::Int, p::Int, q::Int, r::Int) = Tuple(A)[r][q][p][j][i]

# Linear indexing
@inline Base.getindex{N,M}(A::FixedArray2{N,M}, i::Int) = A[ind2sub((N,M), i)...]
@inline Base.getindex{N,M,P}(A::FixedArray3{N,M,P}, i::Int) = A[ind2sub((N,M,P), i)...]
@inline Base.getindex{N,M,P,Q}(A::FixedArray4{N,M,P,Q}, i::Int) = A[ind2sub((N,M,P,Q), i)...]
@inline Base.getindex{N,M,P,Q,R}(A::FixedArray5{N,M,P,Q,R}, i::Int) = A[ind2sub((N,M,P,Q,R), i)...]


end

V = F2.Vec(1,2)
M = F2.@Mat [1 2; 3 4]
A4 = F2.Ar4(rand(2,2,2,2))
A5 = F2.Ar5(rand(2,2,2,2,2))

foo4(A) = A[1,1,1,1]
foo5(A) = A[1,1,1,1,1]
