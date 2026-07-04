function GetProjectRoot()
    # This function attempts to determine the root directory of the project. It first checks for an environment variable "HARTREE_FOCK_PATH". If that is not set, it looks for a "Project.toml" file in the current working directory and its parent directories. If neither method succeeds, it defaults to returning the current working directory.

    # Try environment variable first
    if haskey(ENV, "HARTREE_FOCK_PATH")
        return ENV["HARTREE_FOCK_PATH"]
    end
    
    # Try to find by looking for Project.toml
    Current = pwd()
    while Current != dirname(Current)  # Stop at root
        if isfile(joinpath(Current, "Project.toml"))
            return Current
        end
        Current = dirname(Current)
    end
    
    # Fallback: assume pwd() is project root
    return pwd()
end

function GenerateSTO(n::Int, l::Int, m::Int, ζ::Float64, coeff::Float64)

    # This function generates a Slater-type orbital (STO) with the specified quantum numbers n, l, m, exponent ζ, and coefficient coeff. It also computes the normalization constant for the STO based on its quantum numbers and exponent. The normalization constant ensures that the integral of the square of the STO over all space equals 1.

    if l < 0 || l >= n
        throw(ArgumentError("Invalid quantum numbers: l must satisfy 0 ≤ l ≤ n - 1. Got n = $n, l = $l."))
    end
    normconst = (2 * ζ)^n * sqrt((2 * ζ) / (factorial(2 * n)))
    return STO(n, l, m, ζ, normconst, coeff)
end

function GenerateEC(occα::Vector{Int64}, occβ::Vector{Int64})
    # This function generates an ElectronConfig object based on the provided occupation vectors for alpha (occα) and beta (occβ) spin orbitals. Formally, α spins are defined as +1/2 and β spins are defined as -1/2, so the total spin S is given by S = (Nα - Nβ) / 2, where Nα is the total number of α electrons and Nβ is the total number of β electrons. The multiplicity is then given by 2S + 1.
    Nα = sum(occα)
    Nβ = sum(occβ)
    S = sum(occα - occβ) / 2
    return ElectronConfig(occα, occβ, Nα, Nβ, 2*S + 1)
end

function GenerateEC(occα::Vector{Float64}, occβ::Vector{Float64})
    # This function generates an ElectronConfig object based on the provided occupation vectors for alpha (occα) and beta (occβ) spin orbitals. Formally, α spins are defined as +1/2 and β spins are defined as -1/2, so the total spin S is given by S = (Nα - Nβ) / 2, where Nα is the total number of α electrons and Nβ is the total number of β electrons. The multiplicity is then given by 2S + 1.
    Nα = round(Int,sum(occα))
    Nβ = round(Int,sum(occβ))
    S = sum(occα - occβ) / 2
    return ElectronConfig(occα, occβ, Nα, Nβ, 2*S + 1)
end

function GenerateBasisSet(BasisFunctions::Vector{BasisFunction}, z::Int)
    # This function generates a BasisSet object based on the provided vector of BasisFunction objects and the atomic number z. The BasisSet object contains the list of basis functions, the total number of basis functions (BasisSize), and the atomic number z. The BasisSet is used in quantum chemistry calculations to represent the electronic structure of atoms and molecules.
    return BasisSet(BasisFunctions, length(BasisFunctions), z)
end

function ReadAtomBFs(Element::String)

    # This function reads the basis functions for a given element from a data file. The data file is expected to be located in the "Data/Input/Basis Functions/STOs" directory relative to the project root, and its name should match the provided element and it's associated term symbol (e.g., "H_1S.dat" for hydrogen). The function parses the data file to extract the quantum numbers, exponents, and coefficients for each Slater-type orbital (STO) associated with the element. It returns a vector of BasisFunction objects, each containing a list of STOs corresponding to a specific orbital to later be constructed into a basis set.

    ProjectRoot = GetProjectRoot()
    BasisPath = joinpath(ProjectRoot, "Data", "Input", "Basis Functions", "STOs")
    ElementFile = joinpath(BasisPath, Element)

    if !isfile(ElementFile)
        throw(ArgumentError("Element file not found: $ElementFile"))
    end

    Shells = Dict("S" => 0, "P" => 1, "D" => 2, "F" => 3)
    BasisFuncs = Vector{Vector{STO}}()

    CurrentShell = nothing
    ShellRows = Vector{String}()

    function EmitShell!(Shell::AbstractString, Rows::Vector{String})
        l = Shells[Shell]

        for m in -l:l
            CurrentOrbital = Vector{Vector{STO}}()
            NewOrbital = true

            for Row in Rows
                Data = split(strip(row), r"(\t+\s+|\t+|\s+)")
                len = length(Data)

                if len < 4
                    throw(ArgumentError("Data Line has fewer than 4 fields: $Row"))
                end

                n = tryparse(Int, Data[1])
                lRow = tryparse(Int, Data[2])
                ζ = tryparse(Float64, Data[3])

                if any(x -> x === nothing, (n, lRow, ζ)) 
                    throw(ArgumentError("Failed to parse integers for n, l, or ζ: $Row")) 
                end

                if lRow != l
                    throw(ArgumentError("Shell label $Shell disagrees with row l = $lRow"))
                end

                for i in 4:len
                    coeff = tryparse(Float64, Data[i])
                    if isnothing(coeff) == true
                        throw(ArgumentError("Failed to parse float for coefficient: $Row"))
                    end

                    if NewOrbital == true
                        push!(CurrentOrbital, [GenerateSTO(n, lRow, m, ζ, coeff)])
                    else
                        if i - 3 > length(CurrentOrbital)
                            throw(ArgumentError("Inconsistent number of coefficient columns in block: $Row"))
                        end  
                        push!(CurrentOrbital[i - 3], GenerateSTO(n, lRow, m, ζ, coeff))
                    end
                end
                NewOrbital = false
            end

            append!(BasisFuncs, CurrentOrbital)
        end
    end

    for RawLine in eachline(ElementFile)
        Line = strip(RawLine)

        if isempty(Line)
            continue
        elseif haskey(Shells, Line) && length(Line) == 1
            if CurrentShell !== nothing
                EmitShell!(CurrentShell, ShellRows)
                empty!(ShellRows)
            end
            CurrentShell = Line
        else
            push!(ShellRows, Line)
        end
    end

    if CurrentShell !== nothing
        EmitShell!(CurrentShell, ShellRows)
    end

    BFs = Vector{BasisFunction}()
    for i in 1:lastindex(BasisFuncs)
        push!(BFs, BasisFunction(BasisFuncs[i]))
    end
    return BFs
end
