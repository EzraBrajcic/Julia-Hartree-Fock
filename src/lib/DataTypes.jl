struct STO
    n::Int
    l::Int
    m::Int
    ζ::Float64
    normconst::Float64
    coeff::Float64
end

struct GTO
    n::Int
    l::Int
    m::Int
    ζ::Float64
    normconst::Float64
    coeff::Float64
end

struct BasisFunction
    STOs::Vector{STO}
end

struct BasisSet
    BasisFunctions::Vector{BasisFunction}
    BasisSize::Int
    z::Int
end

struct Atom
    Basis::BasisSet
    pos::Vector{Float64}
end

struct MolecularBasisSet
    Atoms::Vector{Atom}
    z::Int
end

struct ElectronConfig
    occα::Vector{Float64}
    occβ::Vector{Float64}
    Nα::Int
    Nβ::Int
    Multiplicity::Int
end