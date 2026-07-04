using TimerOutputs
using LinearAlgebra
using ProgressMeter

include("STOFuncs.jl")

function OverlapMatrix(Basis::BasisSet, to::TimerOutput)
    
    # ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    # Compute raw overlap matrix, view OverlapInt() in STOFuncs.jl for more information. The overlap matrix essentially shows how much each basis function interacts with each other basis function. Acts a measure of the linear dependence of the basis set.
    # ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

    N = Basis.BasisSize
    S = zeros(Float64, N, N)

    for μ in 1:N
        for ν in μ:N
            sumμν = 0.0
            for STOμ in Basis.BasisFunctions[μ].STOs
                for STOν in Basis.BasisFunctions[ν].STOs
                    @timeit to "Overlap Integral" begin
                        # overlapint already includes normalization constants
                        sumμν += OverlapInt(STOμ, STOν)
                    end
                end
            end
            S[μ,ν] = sumμν
            S[ν,μ] = sumμν
        end
    end  
    return S
end

function OverlapMatrix(Basis::BasisSet)

    # ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    # Compute raw overlap matrix, view OverlapInt() in STOFuncs.jl for more information. The overlap matrix essentially shows how much each basis function interacts with each other basis function. Acts a measure of the linear dependence of the basis set.
    # No timing functionality included in this version to improve performance
    # ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

    N = Basis.BasisSize
    S = zeros(Float64, N, N)

    for μ in 1:N
        for ν in μ:N
            sumμν = 0.0
            for STOμ in Basis.BasisFunctions[μ].STOs
                for STOν in Basis.BasisFunctions[ν].STOs
                    # overlapint already includes normalization constants
                    sumμν += OverlapInt(STOμ, STOν)
                end
            end
            S[μ,ν] = sumμν
            S[ν,μ] = sumμν
        end
    end  
    return S
end

function CoreHamiltonian(Basis::BasisSet, to::TimerOutput)

    # ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    # Compute kinetic + nuclear potential matrix elements, view KineticInt() and NuclearPotentialInt() in STOFuncs.jl for more information. The core Hamiltonian matrix is the sum of the kinetic energy and nuclear potential energy matrices, which are both one-electron operators. The core Hamiltonian matrix is used to compute the Fock matrix.
    # Iterating only from μ = 1 to N and ν = μ to N to exploit
    # symmetry conditions since Hamiltonian matrix is symmetric about the diagonal.
    # Off-diagonal values can then just be copied from the first half of the matrix
    # about the diagonal reducing the number of computations by half
    # ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

    N = Basis.BasisSize
    H = zeros(Float64, N, N)

    pH = Progress(N; desc="Computing core Hamiltonian integrals", dt=0.2)
    for μ in 1:N
        for ν in μ:N
            sumμν = 0.0
            for STOμ in Basis.BasisFunctions[μ].STOs
                for STOν in Basis.BasisFunctions[ν].STOs
                    @timeit to "Kinetic Integral" begin
                        sumμν += KineticInt(STOμ, STOν)
                    end
                    @timeit to "Nuclear Potential Integral" begin
                        sumμν += NuclearPotentialInt(STOμ, STOν, Basis.z)
                    end
                end
            end
            H[μ,ν] = sumμν
            H[ν,μ] = sumμν
        end
        next!(pH)
    end

    finish!(pH)
    return H
end

function CoreHamiltonian(Basis::BasisSet)

    # ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    # Compute kinetic + nuclear potential matrix elements, view KineticInt() and NuclearPotentialInt() in STOFuncs.jl for more information. The core Hamiltonian matrix is the sum of the kinetic energy and nuclear potential energy matrices, which are both one-electron operators. The core Hamiltonian matrix is used to compute the Fock matrix.
    # No timing functionality included in this version to improve performance
    # Iterating only from μ = 1 to N and ν = μ to N to exploit
    # symmetry conditions since Hamiltonian matrix is symmetric about the diagonal.
    # Off-diagonal values can then just be copied from the first half of the matrix
    # about the diagonal reducing the number of computations by half
    # ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

    N = Basis.BasisSize
    H = zeros(Float64, N, N)

    for μ in 1:N
        for ν in μ:N
            sumμν = 0.0
            for STOμ in Basis.BasisFunctions[μ].STOs
                for STOν in Basis.BasisFunctions[ν].STOs
                    sumμν += KineticInt(STOμ, STOν)
                    sumμν += NuclearPotentialInt(STOμ, STOν, Basis.z)
                end
            end
            H[μ,ν] = sumμν
            H[ν,μ] = sumμν
        end
    end
    return H
end

function FockMatrixInts(Basis::BasisSet, to::TimerOutput)

    # ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    # Compute Coulomb repulsion and electron exchange matrix elements of the complete Fock matrix, view CoulombInt() and ExchangeInt() in STOFuncs.jl for more information. The Fock matrix is the sum of the core Hamiltonian and the electron-electron repulsion matrices, which are both one-electron operators.
    # Iterating only from μ = 1 to N and ν = μ to N to exploit
    # symmetry conditions since Fock matrix is symmetric about the diagonal.
    # Off-diagonal values can then just be copied from the first half of the matrix
    # about the diagonal reducing the number of computations by half
    # ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

    N = Basis.BasisSize
    J = zeros(Float64, N, N, N, N)
    K = zeros(Float64, N, N, N, N)

    pμ = Progress(N; desc="Computing Coulomb and exchange integrals", dt=0.2)

    for μ in 1:N
        for ν in μ:N
            for λ in 1:N
                for σ in 1:N
                    j = 0.0
                    k = 0.0
                    for STOμ in Basis.BasisFunctions[μ].STOs
                        for STOν in Basis.BasisFunctions[ν].STOs
                            for STOλ in Basis.BasisFunctions[λ].STOs
                                for STOσ in Basis.BasisFunctions[σ].STOs
                                    @timeit to "Coulomb Integral" begin
                                        j += CoulombInt(STOμ, STOλ, STOν, STOσ)
                                    end
                                    @timeit to "Exchange Integral" begin
                                        k += ExchangeInt(STOμ, STOν, STOλ, STOσ)
                                    end
                                end
                            end
                        end
                    end

                    J[μ,ν,λ,σ] = j
                    K[μ,ν,λ,σ] = k
                end
            end
            if μ != ν
                J[ν,μ, :, :] .= J[μ,ν, :, :]
                K[ν,μ, :, :] .= K[μ,ν, :, :]
            end
        end
        next!(pμ)
    end

    finish!(pμ)
    return J, K
end

function FockMatrixInts(Basis::BasisSet)

    # ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    # Compute Coulomb repulsion and electron exchange matrix elements of the complete Fock matrix, view CoulombInt() and ExchangeInt() in STOFuncs.jl for more information. The Fock matrix is the sum of the core Hamiltonian and the electron-electron repulsion matrices, which are both one-electron operators.
    # No timing functionality included in this version to improve performance
    # Iterating only from μ = 1 to N and ν = μ to N to exploit
    # symmetry conditions since Fock matrix is symmetric about the diagonal.
    # Off-diagonal values can then just be copied from the first half of the matrix
    # about the diagonal reducing the number of computations by half
    # ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

    N = Basis.BasisSize
    J = zeros(Float64, N, N, N, N)
    K = zeros(Float64, N, N, N, N)

    for μ in 1:N
        for ν in μ:N
            for λ in 1:N
                for σ in 1:N
                    j = 0.0
                    k = 0.0
                    for STOμ in Basis.BasisFunctions[μ].STOs
                        for STOν in Basis.BasisFunctions[ν].STOs
                            for STOλ in Basis.BasisFunctions[λ].STOs
                                for STOσ in Basis.BasisFunctions[σ].STOs
                                    j += CoulombInt(STOμ, STOλ, STOν, STOσ)
                                    k += ExchangeInt(STOμ, STOλ, STOν, STOσ)
                                end
                            end
                        end
                    end
                    J[μ,ν,λ,σ] = j
                    K[μ,ν,λ,σ] = k
                end
            end
            if μ != ν
                J[ν,μ, :, :] .= J[μ,ν, :, :]
                K[ν,μ, :, :] .= K[μ,ν, :, :]
            end
        end
    end

    return J, K
end

function FockMatrix(H::Matrix{Float64}, J::Array{Float64,4}, K::Array{Float64,4}, P::Matrix{Float64}, M::Matrix{Float64}, to::TimerOutput)

    # ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    # Compute full Fock matrix elements
    # Iterating only from μ = 1 to N and ν = μ to N to exploit
    # symmetry conditions since Fock and density matrix are symmetric about the diagonal.
    # Off-diagonal values can then just be copied from the first half of the matrix
    # about the diagonal reducing the number of computations by half
    # ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

    N = size(H, 1)
    Fcs = copy(H)
    KM  = zeros(Float64, N, N)

    @timeit to "P⋅(J - 1/2*K) and KM" begin

        for μ in 1:N
            for ν in μ:N
                ClosedShell = 0.0
                ExchangeCorrection = 0.0
                for λ in 1:N
                    for σ in 1:N
                        ClosedShell += P[λ,σ] * (J[μ,ν,λ,σ] - 0.5 * K[μ,ν,λ,σ])
                        ExchangeCorrection += M[λ,σ] * K[μ,ν,λ,σ]
                    end
                end
                Fcs[μ,ν] += ClosedShell
                KM[μ,ν] += ExchangeCorrection
                if μ != ν
                    Fcs[ν,μ] = Fcs[μ,ν]
                    KM[ν,μ] = KM[μ,ν]
                end
            end
        end
    end

    return Fcs, KM
end

function FockMatrix(H::Matrix{Float64}, J::Array{Float64,4}, K::Array{Float64,4}, P::Matrix{Float64}, M::Matrix{Float64})

    # ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    # Compute full Fock matrix elements
    # No timing functionality included in this version to improve performance
    # Iterating only from μ = 1 to N and ν = μ to N to exploit
    # symmetry conditions since Fock and density matrix are symmetric about the diagonal.
    # Off-diagonal values can then just be copied from the first half of the matrix
    # about the diagonal reducing the number of computations by half
    # ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

    N = size(H, 1)
    Fcs = copy(H)
    KM  = zeros(Float64, N, N)

    for μ in 1:N
        for ν in μ:N
            ClosedShell = 0.0
            ExchangeCorrection = 0.0
            for λ in 1:N
                for σ in 1:N
                    ClosedShell += P[λ,σ] * (J[μ,ν,λ,σ] - 0.5 * K[μ,ν,λ,σ])
                    ExchangeCorrection += M[λ,σ] * K[μ,ν,λ,σ]
                end
            end
            Fcs[μ,ν] += ClosedShell
            KM[μ,ν] += ExchangeCorrection
            if μ != ν
                Fcs[ν,μ] = Fcs[μ,ν]
                KM[ν,μ] = KM[μ,ν]
            end
        end
    end

    return Fcs, KM
end

function Orthogonalize(S::Matrix{Float64}, to::TimerOutput)

    # ───────────────────────────────────────────────────────────────────────
    # Eigen-decomposition of S
    # ───────────────────────────────────────────────────────────────────────

    @timeit to "Eigen-decomposition" begin
        evals, evecs = eigen(Symmetric(S))
    end
    eps = maximum(evals) * 1e-20
    evals_clamped = map(v -> max(v, eps), evals)

    # Construct 1/√S
    @timeit to "Construct S^(-1/2)" begin
        S_InverseSqrt = evecs * Diagonal(1 ./ sqrt.(evals_clamped)) * evecs'
    end

    return S_InverseSqrt
end

function Orthogonalize(S::Matrix{Float64})
    # ───────────────────────────────────────────────────────────────────────
    # Eigen-decomposition of S
    # No timing functionality included in this version to improve performance
    # ───────────────────────────────────────────────────────────────────────
    evals, evecs = eigen(Symmetric(S))
    eps = maximum(evals) * 1e-20
    evals_clamped = map(v -> max(v, eps), evals)

    # Construct 1/√S
    S_InverseSqrt = evecs * Diagonal(1 ./ sqrt.(evals_clamped)) * evecs'

    return S_InverseSqrt
end

function DensityMatrices(Cα::Matrix{Float64}, Cβ::Matrix{Float64}, EC::ElectronConfig)
    # ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    # Compute density matrices for alpha and beta spin orbitals, as well as total charge and spin density matrices. The density matrix is a key quantity in quantum chemistry that describes the distribution of electrons in a system. The trace of the total density matrix gives the total number of electrons in the system, while the trace of the spin density matrix gives the total spin of the system. The density matrices are used to compute the Fock matrix and other quantities in the SCF procedure. The α and β density matrices are given by Σ_i^BasisSizeΣ_j^BasisSize C_μi * C_νi * occ_i, where C is the molecular orbital coefficient matrix and occ is the occupation number of the i-th molecular orbital. The total charge density matrix is then given by P = Pα + Pβ, while the spin density matrix is given by M = (Pα - Pβ) / 2.
    # ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

    Sα = size(Cα, 1)
    Sβ = size(Cβ, 1)
    Pα = zeros(Float64, Sα, Sα)
    Pβ = zeros(Float64, Sβ, Sβ)

    for i in 1:Sα
        for j in 1:Sα
            for k in 1:length(EC.occα)
                Pα[i,j] += Cα[i,k] * Cα[j,k] * EC.occα[k]
            end
        end
    end

    for i in 1:Sβ
        for j in 1:Sβ
            for k in 1:length(EC.occβ)
                Pβ[i,j] += Cβ[i,k] * Cβ[j,k] * EC.occβ[k]
            end
        end
    end

    # One may ask why one would chose 2 different density matrix conventions, this is due to a severe lack of foresight when developing this program. The first convention is the one used in the original code for an RHF scheme, where the total density matrix is defined as P = Pα + Pβ. The second convention is the one used in the DIIS procedure, where the total density matrix is defined as P = (Pα + Pβ) / 2. The second convention is more commonly used in the literature, but the first convention was used in the original code for historical reasons and I did not want to have to refactor the entire SCF algorithm around the more commonly used convention. The choice of convention does not affect the results of the calculations as long as each convention is used properly, but it does affect the interpretation of the results.
    P  = (Pα + Pβ)         # total charge density
    M  = (Pα - Pβ) / 2     # spin density
    return Pα, Pβ, P, M
end

function DeltaP(P::Matrix{Float64}, P_Old::Matrix{Float64})
    # Compute change in density matrix using root mean square deviation
    N = size(P, 1)
    Delta = 0
    for i in 1:N
        for j in 1:N
            Delta += (P[i,j] - P_Old[i,j])^2
        end
    end
    return sqrt(Delta)
end

function DIIS(TransformationMatrix::Matrix{Float64}, S::Matrix{Float64}, F::Matrix{Float64}, P::Matrix{Float64}, FockList::Vector{Matrix{Float64}}, ErrorList::Vector{Matrix{Float64}})

    # ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    # Direct inversion of the iterative subspace (DIIS) extrapolates solution for a set of linear equations by minimizing an error residual of a linear combination of vectors generated by each iteration of the SCF procedure, increasing the rate of convergence. If the residual product matrix B is singular, the pseudoinverse of B is used instead. If the pseudoinverse fails, the current Fock matrix is used instead.
    # ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

    # If the length of the error list exceeds N, the oldest error and Fock matrices are removed to maintain stability of the DIIS procedure. This is because the error residuals from earlier iterations may not be as relevant to the current iteration, and including too many error residuals can lead to numerical instability in the DIIS extrapolation.
    if length(ErrorList) == 7
        popfirst!(ErrorList)
        popfirst!(FockList)
    end

    # ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    # Compute error residual matrix via the comutator idenities [Fα,Pα] - [Fβ,Pβ] = 0 where Fα and Fβ are the Fock matrices for the α and β spin orbitals, respectively, and Pα and Pβ are the density matrices for the α and β spin orbitals, respectively. The commutator is defined as [A,B] = AB - BA. The error residual matrix is then transformed into the orthogonal basis along with the Fock matrix using the orthogonal transformation matrix 1/√S.
    # ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    ResidualMatrix = TransformationMatrix' * (Fcs * M * S - S * M * Fcs - KM * (P/2) * S + S * (P/2) * KM) * TransformationMatrix

    F_Ortho_Current = TransformationMatrix' * F * TransformationMatrix

    push!(FockList, F_Ortho_Current)
    push!(ErrorList, ResidualMatrix)

    len = length(ErrorList)

    if len > 2
        B = zeros(Float64, len + 1, len + 1)
        BasisSize = size(F, 1)
        F_Ortho = zeros(Float64, BasisSize, BasisSize)

        for i in 1:len
            for j in 1:len
                B[i,j] = tr(ErrorList[i]' * ErrorList[j])
            end
        end
        B[1:len, len + 1] .= -1.0
        B[len + 1, 1:len] .= -1.0
        B[len + 1, len + 1] = 0.0
        vec = zeros(Float64, len + 1)
        vec[end] = -1.0

        try
            # Attempt to solve the linear system B * coeffs = vec using the backslash operator. If B is singular, this will throw an error, which is caught in the catch block where the pseudoinverse of B will be used instead.
            coeffs = Symmetric(B) \ vec
            for i in 1:len
                F_Ortho .+= coeffs[i] * FockList[i]
            end
        catch e

            # DIIS failed, try pseudoinverse
            try
                coeffs = pinv(Symmetric(B)) * vec
                for i in 1:len
                    F_Ortho .+= coeffs[i] * FockList[i]
                end
                println("DIIS inv failed, applying pseudoinverse: ", e)
            catch e
                # DIIS pseudoinv failed, use current Fock matrices instead
                F_Ortho = F_Ortho_Current
                println("DIIS pseudoinv failed, using current Fock matrices instead: ", e)
            end
        end
    else
        F_Ortho = F_Ortho_Current
    end

    return F_Ortho, FockList, ErrorList
end

function DIIS(TransformationMatrix::Matrix{Float64}, S::Matrix{Float64}, Fcs::Matrix{Float64}, KM::Matrix{Float64}, Fα::Matrix{Float64}, Fβ::Matrix{Float64}, P::Matrix{Float64}, M::Matrix{Float64}, FockListα::Vector{Matrix{Float64}}, FockListβ::Vector{Matrix{Float64}}, ErrorList::Vector{Matrix{Float64}})

    # ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    # Direct inversion of the iterative subspace (DIIS) extrapolates solution for a set of linear equations by minimizing an error residual of a linear combination of vectors generated by each iteration of the SCF procedure, increasing the rate of convergence. If the residual product matrix B is singular, the pseudoinverse of B is used instead. If the pseudoinverse fails, the current Fock matrix is used instead.
    # ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    # If the length of the error list exceeds N, the oldest error and Fock matrices are removed to maintain stability of the DIIS procedure. This is because the error residuals from earlier iterations may not be as relevant to the current     
    # ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    if length(ErrorList) == 5
        popfirst!(ErrorList)
        popfirst!(FockList)
    end

    # ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    # Compute error residual matrix via the comutator idenities [Fα,Pα] - [Fβ,Pβ] = 0 where Fα and Fβ are the Fock matrices for the α and β spin orbitals, respectively, and Pα and Pβ are the density matrices for the α and β spin orbitals, respectively. The commutator is defined as [A,B] = AB - BA. The error residual matrix is then transformed into the orthogonal basis along with the Fock matrices using the orthogonal transformation matrix 1/√S.
    # ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

    ResidualMatrix = TransformationMatrix' * (Fcs * M * S - S * M * Fcs - KM * (P/2) * S + S * (P/2) * KM) * TransformationMatrix
    Fα_Ortho_Current = TransformationMatrix' * Fα * TransformationMatrix
    Fβ_Ortho_Current = TransformationMatrix' * Fβ * TransformationMatrix

    push!(FockListα, Fα_Ortho_Current)
    push!(FockListβ, Fβ_Ortho_Current)
    push!(ErrorList, ResidualMatrix)

    len = length(ErrorList)

    if len > 2
        B = zeros(Float64, len + 1, len + 1)
        BasisSize = size(Fcs, 1)
        Fα_Ortho = zeros(Float64, BasisSize, BasisSize)
        Fβ_Ortho = zeros(Float64, BasisSize, BasisSize)

        for i in 1:len
            for j in 1:len
                B[i,j] = tr(ErrorList[i]' * ErrorList[j])
            end
        end
        B[1:len, len + 1] .= -1.0
        B[len + 1, 1:len] .= -1.0
        B[len + 1, len + 1] = 0.0
        vec = zeros(Float64, len + 1)
        vec[end] = -1.0

        try
            # Attempt to solve the linear system B * coeffs = vec using the backslash operator. If B is singular, this will throw an error, which is caught in the catch block where the pseudoinverse of B will be used instead.
            coeffs = Symmetric(B) \ vec
            for i in 1:len
                Fα_Ortho .+= coeffs[i] * FockListα[i]
                Fβ_Ortho .+= coeffs[i] * FockListβ[i]
            end
        catch e

            # DIIS failed, try pseudoinverse
            try
                coeffs = pinv(Symmetric(B)) * vec
                for i in 1:len
                    Fα_Ortho .+= coeffs[i] * FockListα[i]
                    Fβ_Ortho .+= coeffs[i] * FockListβ[i]
                end
            catch e
                # DIIS pseudoinv failed, use current Fock matrices instead
                Fα_Ortho = Fα_Ortho_Current
                Fβ_Ortho = Fβ_Ortho_Current
            end
        end
    else
        Fα_Ortho = Fα_Ortho_Current
        Fβ_Ortho = Fβ_Ortho_Current
    end

    return Fα_Ortho, Fβ_Ortho, FockListα, FockListβ, ErrorList, ResidualMatrix
end

function SCF(Basis::BasisSet, EC::ElectronConfig, PGuess::Matrix{Float64} = zeros(Float64, Basis.BasisSize, Basis.BasisSize), ConvC::Float64 = 1e-10, MaxIter::Int = 1000, to::TimerOutput = nothing)

    # ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    # Performs the constrained unrestricted Hartree-Fock (CUHF) self-consistent field (SCF) procedure for a given basis set and electron configuration. Specifically, it iteratively solves the Pople-Nesbet equations for the α and β spin orbitals, while enforcing constraints on the spin density matrix to ensure that the total spin of the system is conserved, effectively solving Roothaan's equations with an effective Fock operator within the UHF framework. The procedure continues until convergence is achieved based on the specified convergence criteria or until the maximum number of iterations is reached.
    # ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

    # Initialize variables and matrices for the SCF procedure
    BasisSize = Basis.BasisSize

    E = Float64[]

    Fα = zeros(Float64, BasisSize, BasisSize)
    Fβ = zeros(Float64, BasisSize, BasisSize)
    Fcs = zeros(Float64, BasisSize, BasisSize)
    KM = zeros(Float64, BasisSize, BasisSize)

    Fα_Ortho = zeros(Float64, BasisSize, BasisSize)
    Fβ_Ortho = zeros(Float64, BasisSize, BasisSize)

    Cα = zeros(Float64, BasisSize, BasisSize)
    Cβ = zeros(Float64, BasisSize, BasisSize)

    J = zeros(Float64, BasisSize, BasisSize, BasisSize, BasisSize)
    K = zeros(Float64, BasisSize, BasisSize, BasisSize, BasisSize)

    # lists of previous Fock matrices and error matrices for DIIS
    FockListα = Vector{Matrix{Float64}}()
    FockListβ = Vector{Matrix{Float64}}()
    ErrorList = Vector{Matrix{Float64}}()

    evecsα = zeros(Float64, BasisSize, 1)
    evecsβ = zeros(Float64, BasisSize, 1)

    evalsα = zeros(Float64, BasisSize, 1)
    evalsβ = zeros(Float64, BasisSize, 1)

    Count = 0
    Nc = EC.Nβ
    ΔPα = 1.0
    ΔPβ = 1.0
    FNR = 1.0
    max_R = 1.0

    # Initialize density matrices from guess parameters. 
    Pα = PGuess / 2
    Pβ = PGuess / 2
    P = (Pα + Pβ)
    M = (Pα - Pβ) / 2

    # Compute overlap matrix and orthogonalize it to obtain the transformation matrix for the Fock matrices
    S = OverlapMatrix(Basis, to)
    S_Ortho = Orthogonalize(S, to)

    # Compute core Hamiltonian matrix elements
    H = CoreHamiltonian(Basis, to)

    # Compute Coulomb repulsion and electron exchange matrix elements of the complete Fock matrix
    J, K = FockMatrixInts(Basis, to)
    
    # Perform the SCF procedure iteratively until convergence is achieved or the maximum number of iterations is reached
    @timeit to "SCF Procedure" begin
        @showprogress 1 "SCF Converging..." for i in 1:MaxIter
            Count += 1

            # Build new Fock matrix
            Fcs, KM = FockMatrix(H, J, K, P, M, to)

            @timeit to "Apply Lagrange Multiplier Constraints" begin
                # ──────────────────────────────────────────────────────────────────────────────────────────────────────────────
                # Apply Lagrange multiplier constraints to the Fock matrices to enforce the spin density matrix constraints. This is done by transforming the exchange correction matrix KM into the molecular orbital basis, zeroing out the off-diagonal elements corresponding to the occupied-virtual orbitals (cv and vc), and then transforming it back into the original basis. The resulting constrained Fock matrices for the α and β spin orbitals are then computed by adding and subtracting the constrained exchange correction matrix from the closed-shell Fock matrix. This ensures that the total spin of the system is conserved during the SCF procedure.
                # ──────────────────────────────────────────────────────────────────────────────────────────────────────────────

                KM_MO = Cα' * KM * Cα
                KM_MO[1:Nc,             EC.Nα+1:BasisSize] .= 0.0   # cv
                KM_MO[EC.Nα+1:BasisSize, 1:Nc            ] .= 0.0   # vc
                KM = S * Cα * KM_MO * Cα' * S

                Fα = Fcs - KM
                Fβ = Fcs + KM
            end

            # DIIS on the constrained matrices
            @timeit to "Orthogonalize Fock Matrices" begin
                Fα_Ortho, Fβ_Ortho, FockListα, FockListβ, ErrorList, ResidualMatrix = DIIS(S_Ortho, S, Fcs, KM, Fα, Fβ, P, M, FockListα, FockListβ, ErrorList)
            end
            
            # Solve eigenvalue problem for orthogonalized Fock matrix
            @timeit to "Eigenvalues and Eigenvectors of Fα' and Fβ'" begin
                evalsα, evecsα = eigen(Symmetric(Fα_Ortho))
                evalsβ, evecsβ = eigen(Symmetric(Fβ_Ortho))
            end

            # Back-transform eigenvectors to original basis
            @timeit to "Transform Cα and Cβ to original basis" begin
                Cα = S_Ortho * evecsα
                Cβ = S_Ortho * evecsβ
            end

            # Update density matrices
            Pα_Old = copy(Pα)
            Pβ_Old = copy(Pβ)
            @timeit to "Construct Density Matrices" begin
                Pα, Pβ, P, M = DensityMatrices(Cα, Cβ, EC)
            end

            # Compute change in density matrices and other convergence criterion
            @timeit to "ΔPα, ΔPβ, √(ΣR^2)/N, and max(|R|)" begin

                # Compute change in density matrices using root mean square deviation
                ΔPα = DeltaP(Pα, Pα_Old)
                ΔPβ = DeltaP(Pβ, Pβ_Old)

                # Compute the Frobenius norm of the error residual matrix and the maximum absolute value of the error residual matrix. The Frobenius norm is a measure of the overall magnitude of the error residuals, while the maximum absolute value indicates the largest individual error residual.
                FNR = sqrt(tr(ResidualMatrix' * ResidualMatrix)) / (BasisSize * (BasisSize - 1))
                max_R = maximum(abs.(ResidualMatrix))
            end

            # Compute total system energy and store it in the energy history array
            @timeit to "Total System Energy" begin
                Energy = 0.5 * (tr(Pα*(H + Fα)) + tr(Pβ*(H + Fβ))) 
                push!(E, Energy)
            end
            
            # Check for convergence based on the specified convergence criteria. If the change in density matrices (ΔPα and ΔPβ) and the error residuals (FNR and max_R) are all below the convergence threshold (ConvC), and at least one iteration has been completed, the SCF procedure is considered converged, and the loop is exited.
            if ((ΔPα < ConvC) && (ΔPβ < ConvC)) && ((FNR < ConvC) && (max_R < 10*ConvC)) && Count > 1
                println("\nSCF Converged in $Count iterations.")
                break
            end
        end
    end

    # Return the final results of the SCF procedure, including the total energy, energy history, number of iterations, changes in density matrices, error residuals, eigenvalues and eigenvectors of the Fock matrices, Coulomb and exchange integrals, density matrices, overlap matrix and its orthogonalization, Fock matrices, and molecular orbital coefficients for both α and β spin orbitals. Additionally, the TimerOutput object is returned to provide timing information for the various computational steps in the SCF procedure.
    return E[Count], E, Count, ΔPα, ΔPβ, FNR, max_R,
    evecsα, evecsβ, evalsα, evalsβ,
    J, K, Pα, Pβ, P, M, S, S_Ortho, 
    Fcs, Fα, Fβ, Fα_Ortho, Fβ_Ortho, 
    Cα, Cβ, 
    to
end

function SCF(Basis::BasisSet, EC::ElectronConfig, PGuess::Matrix{Float64} = zeros(Float64, Basis.BasisSize, Basis.BasisSize), ConvC::Float64 = 1e-10, MaxIter::Int = 1000)

    # Initialize variables and matrices for the SCF procedure
    BasisSize = Basis.BasisSize

    E = Float64[]

    Fα = zeros(Float64, BasisSize, BasisSize)
    Fβ = zeros(Float64, BasisSize, BasisSize)
    Fcs = zeros(Float64, BasisSize, BasisSize)
    KM = zeros(Float64, BasisSize, BasisSize)

    Fα_Ortho = zeros(Float64, BasisSize, BasisSize)
    Fβ_Ortho = zeros(Float64, BasisSize, BasisSize)

    Cα = zeros(Float64, BasisSize, BasisSize)
    Cβ = zeros(Float64, BasisSize, BasisSize)

    J = zeros(Float64, BasisSize, BasisSize, BasisSize, BasisSize)
    K = zeros(Float64, BasisSize, BasisSize, BasisSize, BasisSize)

    # lists of previous Fock matrices and error matrices for DIIS
    FockListα = Vector{Matrix{Float64}}()
    FockListβ = Vector{Matrix{Float64}}()
    ErrorList = Vector{Matrix{Float64}}()

    evecsα = zeros(Float64, BasisSize, 1)
    evecsβ = zeros(Float64, BasisSize, 1)

    evalsα = zeros(Float64, BasisSize, 1)
    evalsβ = zeros(Float64, BasisSize, 1)

    Count = 0
    Nc = EC.Nβ
    ΔPα = 1.0
    ΔPβ = 1.0
    FNR = 1.0
    max_R = 1.0

    # Initialize density matrices from guess parameters. 
    Pα = PGuess / 2
    Pβ = PGuess / 2
    P = (Pα + Pβ)
    M = (Pα - Pβ) / 2

    # Compute overlap matrix and orthogonalize it to obtain the transformation matrix for the Fock matrices
    S = OverlapMatrix(Basis, to)
    S_Ortho = Orthogonalize(S, to)

    # Compute core Hamiltonian matrix elements
    H = CoreHamiltonian(Basis, to)

    # Compute Coulomb repulsion and electron exchange matrix elements of the complete Fock matrix
    J, K = FockMatrixInts(Basis, to)
    
    # Perform the SCF procedure iteratively until convergence is achieved or the maximum number of iterations is reached
    for i in 1:MaxIter
        Count += 1

        # Build new Fock matrix
        Fcs, KM = FockMatrix(H, J, K, P, M)

        # Apply Lagrange multiplier constraints to the Fock matrices to enforce the spin density matrix constraints. This is done by transforming the exchange correction matrix KM into the molecular orbital basis, zeroing out the off-diagonal elements corresponding to the occupied-virtual orbitals (cv and vc), and then transforming it back into the original basis. The resulting constrained Fock matrices for the α and β spin orbitals are then computed by adding and subtracting the constrained exchange correction matrix from the closed-shell Fock matrix. This ensures that the total spin of the system is conserved during the SCF procedure.
        KM_MO = Cα' * KM * Cα
        KM_MO[1:Nc,             EC.Nα+1:BasisSize] .= 0.0   # cv
        KM_MO[EC.Nα+1:BasisSize, 1:Nc            ] .= 0.0   # vc
        KM = S * Cα * KM_MO * Cα' * S

        Fα = Fcs - KM
        Fβ = Fcs + KM

        # DIIS on the constrained matrices
        Fα_Ortho, Fβ_Ortho, FockListα, FockListβ, ErrorList, ResidualMatrix = DIIS(S_Ortho, S, Fcs, KM, Fα, Fβ, P, M, FockListα, FockListβ, ErrorList)
        
        # Solve eigenvalue problem for orthogonalized Fock matrix
        evalsα, evecsα = eigen(Symmetric(Fα_Ortho))
        evalsβ, evecsβ = eigen(Symmetric(Fβ_Ortho))

        # Back-transform eigenvectors to original basis
        Cα = S_Ortho * evecsα
        Cβ = S_Ortho * evecsβ

        # Update density matrices
        Pα_Old = copy(Pα)
        Pβ_Old = copy(Pβ)
        Pα, Pβ, P, M = DensityMatrices(Cα, Cβ, EC)

       # Compute change in density matrices using root mean square deviation
        ΔPα = DeltaP(Pα, Pα_Old)
        ΔPβ = DeltaP(Pβ, Pβ_Old)

        # Compute the Frobenius norm of the error residual matrix and the maximum absolute value of the error residual matrix. The Frobenius norm is a measure of the overall magnitude of the error residuals, while the maximum absolute value indicates the largest individual error residual.        
        FNR = sqrt(tr(ResidualMatrix.^2)) / (BasisSize * (BasisSize - 1))
        max_R = maximum(abs.(ResidualMatrix))

        # Compute total system energy and store it in the energy history array
        Energy = 0.0
        Energy = 0.5 * (tr(Pα*(H + Fα)) + tr(Pβ*(H + Fβ))) 
        push!(E, Energy)
        
        # Check for convergence based on the specified convergence criteria. If the change in density matrices (ΔPα and ΔPβ) and the error residuals (FNR and max_R) are all below the convergence threshold (ConvC), and at least one iteration has been completed, the SCF procedure is considered converged, and the loop is exited.
        if ((ΔPα < ConvC) && (ΔPβ < ConvC)) || (FNR < ConvC && max_R < 10*ConvC && Count > 1)
            println("\nSCF Converged in $Count iterations.")
            break
        end
    end

    # Return the final results of the SCF procedure, including the total energy, energy history, number of iterations, changes in density matrices, error residuals, eigenvalues and eigenvectors of the Fock matrices, Coulomb and exchange integrals, density matrices, overlap matrix and its orthogonalization, Fock matrices, and molecular orbital coefficients for both α and β spin orbitals.
    return E[Count], E, Count, ΔPα, ΔPβ, FNR, max_R,
    evecsα, evecsβ, evalsα, evalsβ,
    J, K, Pα, Pβ, P, M, S, S_Ortho, 
    Fcs, Fα, Fβ, Fα_Ortho, Fβ_Ortho, 
    Cα, Cβ
end