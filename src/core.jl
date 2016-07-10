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
# Is there any point going to more dimensions than MAX_TUPLE_DEPTH?
abstract FixedArray5{N,M,P,Q,R,T} <: FixedArray{T,5}

typealias FixedVector FixedArray1
typealias FixedMatrix FixedArray2

abstract FixedVectorNoTuple{N,T} <: FixedVector{N,T}

#-------------------------------------------------------------------------------
# Primary interface to unwrap elements: Tuple(FSA)
@inline Base.Tuple(A::FixedArray) = A.values

#= FIXME: Do we need the functions below?
@compat @generated function (::Type{T}){T<:Tuple, N, T1}(f::FixedVectorNoTuple{N, T1})
    return Expr(:tuple, ntuple(i->:(f[$i]), N)...)
end
@compat function (::Type{T}){T<:Tuple}(f::FixedArray)
    getfield(f, 1)
end
=#


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

# TODO: LinearFast or LinearSlow?  Probably LinearFast because the divisors in
# ind2sub are known at compile time.
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


#-------------------------------------------------------------------------------
# similar_type implementation
"""
    similar_type(::Type{FSA}, [::Type{T}=eltype(FSA)], [sz=size(FSA)])

Given an array type `FSA`, element type `T` and size `sz`, return a `FixedArray`
subtype which is as similar as possible.  `similar_type` is used in the same
spirit as `Base.similar` to store the results of `map()` operations, etc.
(`similar` cannot work here, because the types are generally immutable.)

By default, `similar_type` introspects `FSA` to determine whether `T` and `sz`
can be used; if not a canonical FixedArray container is returned instead.
"""
@pure function similar_type{FSA <: FixedArray, T}(::Type{FSA}, ::Type{T}, sz::Tuple)
    fsa_size = (fsa_abstract1(FSA).parameters[1:end-1]...)
    if eltype(FSA) == T && fsa_size == sz
        return FSA # Common case optimization
    end

    # The default implementation for similar_type is follows.  It involves a
    # fair bit of crazy type introspection: We check whether the type `FSA` has
    # the necessary type parameters to replace with `T` and `sz`, and if so
    # figure out how to do the replacement.  It's complicated because users may
    # arbitrarily rearrange type parameters in their subtypes, and possibly
    # even add new type parameters which aren't related to the abstract
    # FixedArray but should be preserved.

    # Propagate the available type parameters of FSA down to the abstract base
    # FixedArray as `TypeVar`s.
    pritype = FSA.name.primary
    abstract_params = fsa_abstract1(pritype).parameters
    sz_parameters  = abstract_params[1:end-1]
    T_parameter    = abstract_params[end]

    # Figure out whether FSA can accommodate the new eltype `T` and size `sz`.
    # If not, delegate to the fallback by default.
    if !((eltype(FSA) == T          || isa(T_parameter, TypeVar)) &&
         (fsa_size    == sz         || (length(fsa_size) == length(sz) && all(i -> (sz[i] == fsa_size[i] || isa(sz_parameters[i],TypeVar)), 1:length(sz)))))
        return similar_type(FixedArray, T, sz)
    end

    # Iterate type parameters, replacing as necessary with T and sz
    newparams = collect(FSA.parameters)
    priparams = pritype.parameters
    for i=1:length(newparams)
        if priparams[i] === T_parameter
            newparams[i] = T
        else
            for j = 1:length(sz_parameters)
                if priparams[i] === sz_parameters[j]
                    newparams[i] = sz[j]
                end
            end
        end
    end
    pritype{newparams...}
end

# similar_type versions with defaulted eltype and size
@pure similar_type{FSA <: FixedArray, T}(::Type{FSA}, ::Type{T}) = similar_type(FSA, T, size(FSA))
@pure similar_type{FSA <: FixedArray}(::Type{FSA}, sz::Tuple) = similar_type(FSA, eltype(FSA), sz)


#-------------------------------------------------------------------------------
# construct_similar implementation

"Compute promoted element type of the potentially nested Tuple type `ty`"
function promote_type_nested(ty)
    if ty.name.primary === Tuple
        promote_type([promote_type_nested(t) for t in ty.parameters]...)
    else
        ty
    end
end

"""
Construct tuple expression converting inner elements of the nested Tuple type
`ty` with name `varexpr` to the scalar `outtype`
"""
function convert_nested_tuple_expr(outtype, varexpr, ty)
    if ty.name.primary === Tuple
        Expr(:tuple, [convert_nested_tuple_expr(outtype, :($varexpr[$i]), t)
                      for (i,t) in enumerate(ty.parameters)]...)
    else
        :(convert($outtype, $varexpr))
    end
end

"""
Compute the N-dimensional array shape of a nested Tuple if it were used as
column-major storage for a FixedArray.
"""
function nested_Tuple_shape(ty)
    if ty.name.primary !== Tuple
        return 0
    end
    subshapes = [nested_Tuple_shape(t) for t in ty.parameters]
    if isempty(subshapes)
        return ()
    end
    if any(subshapes .!= subshapes[1])
        throw(DimensionMismatch("Nested tuples must have equal length to form a FixedSizeArray"))
    end
    if subshapes[1] == 0
        return (length(subshapes),) # Scalar elements
    end
    return (subshapes[1]..., length(subshapes))
end

"""
    construct_similar(::Type{FSA}, elements::Tuple)

Construct FixedArray as similar as possible to `FSA`, but with shape given by
the shape of the nested tuple `elements` and with `eltype` equal to the
promoted element type of the nested tuple `elements`.
"""
@generated function construct_similar{FSA <: FixedArray}(::Type{FSA}, elements::Tuple)
    etype = promote_type_nested(elements)
    shape = nested_Tuple_shape(elements)
    outtype = similar_type(FSA, etype, shape)
    converted_elements = convert_nested_tuple_expr(etype, :elements, elements)
    constructor_expr(outtype, converted_elements)
end
