using TimerOutputs
using LinearAlgebra
using ProgressMeter

function OverlapMatrix(BasisSet::BasisSet, to::TimerOutput)
    N = BasisSet.BasisSize
    S = zeros(Float64, N, N)
    
    # Compute raw overlap matrix with contraction coefficients
    for μ in 1:N
        for ν in μ:N
            sumμν = 0.0
            for STOμ in BasisSet.BasisFunctions[μ].STOs
                for STOν in BasisSet.BasisFunctions[ν].STOs
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

function CoreHamiltonian(BasisSet::BasisSet, to::TimerOutput)
    N = BasisSet.BasisSize
    H = zeros(Float64, N, N)

    # Compute kinetic + nuclear potential matrix elements

    # Iterating only from μ = 1 to N and ν = μ to N to exploit
    # symmetry conditions since Hamiltonian matrix is symmetric about the diagonal.
    # Off-diagonal values can then just be copied from the first half of the matrix
    # about the diagonal reducing the number of computations by half
    for μ in 1:N
        for ν in μ:N
            sumμν = 0.0
            for STOμ in BasisSet.BasisFunctions[μ].STOs
                for STOν in BasisSet.BasisFunctions[ν].STOs
                    @timeit to "Kinetic Integral" begin
                        sumμν += KineticInt(STOμ, STOν)
                    end
                    @timeit to "Nuclear Potential Integral" begin
                        sumμν += NuclearPotentialInt(STOμ, STOν, BasisSet.z)
                    end
                end
            end
            H[μ,ν] = sumμν
            H[ν,μ] = sumμν  
        end
    end
    return H
end

function FockMatrixInts(BasisSet::BasisSet, to::TimerOutput)
    N = BasisSet.BasisSize
    JmK = zeros(Float64, N, N, N, N)

    # Compute Coulomb repulsion and electron exchange matrix elements of the complete Fock matrix

    # Iterating only from μ = 1 to N and ν = μ to N to exploit
    # symmetry conditions since Fock matrix is symmetric about the diagonal.
    # Off-diagonal values can then just be copied from the first half of the matrix
    # about the diagonal reducing the number of computations by half
    for μ in 1:N
        for ν in μ:N
            for λ in 1:N
                for σ in 1:N
                    J = 0.0
                    K = 0.0
                    for STOμ in BasisSet.BasisFunctions[μ].STOs
                        for STOν in BasisSet.BasisFunctions[ν].STOs
                            for STOλ in BasisSet.BasisFunctions[λ].STOs
                                for STOσ in BasisSet.BasisFunctions[σ].STOs
                                    @timeit to "Coulomb Integral" begin
                                        J += CoulombInt(STOμ, STOλ, STOν, STOσ)
                                    end
                                    @timeit to "Exchange Integral" begin
                                        K += ExchangeInt(STOμ, STOλ, STOν, STOσ)
                                    end
                                end
                            end
                        end
                    end
                    JmK[μ,ν,λ,σ] = (J - 0.5*K)
                end
            end
            if μ != ν
                copyto!(view(JmK, ν, μ, :, :), view(JmK, μ, ν, :, :))
            end
        end
    end
    return JmK
end

function FockMatrix(BasisSet::BasisSet, P::Matrix{Float64}, HCore::Matrix{Float64}, JmK::Array{Float64, 4}, to::TimerOutput)
    N = BasisSet.BasisSize
    F = copy(HCore)

    # Compute full Fock matrix elements and iterate SCF procedure

    @timeit to "P⋅(J - 1/2*K)" begin
        # Iterating only from μ = 1 to N and ν = μ to N to exploit
        # symmetry conditions since Fock and density matrix are symmetric about the diagonal.
        # Off-diagonal values can then just be copied from the first half of the matrix
        # about the diagonal reducing the number of computations by half
        for μ in 1:N
            for ν in μ:N
                for λ in 1:N
                    for σ in 1:N
                        F[μ,ν] += P[λ,σ] * JmK[μ,ν,λ,σ]
                    end
                end
                if μ != ν
                    F[ν,μ] = F[μ,ν]
                end
            end
        end
    end
    return F
end

function Orthogonalize(S::Matrix{Float64}, to::TimerOutput)

    # Eigen-decomposition of S
    @timeit to "Eigen-decomposition" begin
        evals, evecs = eigen(Symmetric(S))
    end
    eps = maximum(evals) * 1e-10
    evals_clamped = map(v -> max(v, eps), evals)

    # Construct S^(-1/2)
    @timeit to "Construct S^(-1/2)" begin
        S_InverseSqrt = evecs * Diagonal(1 ./ sqrt.(evals_clamped)) * evecs'
    end

    return S_InverseSqrt
end

function DensityMatrix(C::Matrix{Float64}, BasisSet::BasisSet, Nelec::Int)
    N = size(C, 1)
    P_New = zeros(Float64, N, N)
   
    for i in 1:N
        for j in 1:N
            for k in 1:(BasisSet.BasisSize)
                P_New[i,j] += C[i,k] * C[j,k] * BasisSet.BasisFunctions[k].Occupations
            end
        end
    end
    return P_New
end

function DeltaP(P::Matrix{Float64}, P_Old::Matrix{Float64}, BasisSet::BasisSet)
    # Compute change in density matrix using root mean square deviation
    Delta = 0
    for i in 1:BasisSet.BasisSize
        for j in 1:BasisSet.BasisSize
            Delta += (P[i,j] - P_Old[i,j])^2
        end
    end
    return sqrt(Delta)
end

function DIIS(TransformationMatrix::Matrix{Float64}, S::Matrix{Float64}, F::Matrix{Float64}, P::Matrix{Float64}, FockList::Vector{Matrix{Float64}}, ErrorList::Vector{Matrix{Float64}})
    
    # Direct inversion of the iterative subspace (DIIS) extrapolates solution for a set of linear equations
    # by minimizing an error residual of a linear combination of vectors generated by each iteration of the
    # SCF procedure, increasing the rate of convergence. Will fail if the matrix B is singular.

    if length(ErrorList) == 7
        popfirst!(ErrorList)
        popfirst!(FockList)
    end

    ErrorVector = TransformationMatrix' * (F * P * S - S * P * F) * TransformationMatrix

    push!(FockList, F)
    push!(ErrorList, ErrorVector)

    len = length(ErrorList)

    if len > 20
        B = zeros(Float64, len + 1, len + 1)
        BasisSize = size(F, 1)
        F_Ortho = zeros(Float64, BasisSize, BasisSize)

        for i in 1:len
            for j in 1:len
                B[i,j] = tr(ErrorList[i]' .* ErrorList[j])
            end
        end
        B[1:len, len + 1] .= -1.0
        B[len + 1, 1:len] .= -1.0
        B[len + 1, len + 1] = 0.0

        vec = zeros(Float64, len + 1)
        vec[end] = -1.0

        coeffs = Symmetric(B) \ vec

        for i in 1:len
            F_Ortho .+= coeffs[i] * FockList[i]
        end
    else
        F_Ortho = TransformationMatrix' * F * TransformationMatrix
    end

    return F_Ortho, FockList, ErrorList
end

function SCF(Basis::BasisSet, Nelec::Int, PGuess::Matrix{Float64} = zeros(Float64, Basis.BasisSize, Basis.BasisSize), ConvC::Float64 = 1e-10, MaxIter::Int = 1000, to::TimerOutput = nothing)

    BasisSize = Basis.BasisSize

    E = Float64[]

    F = zeros(Float64, BasisSize, BasisSize)
    F_Ortho = zeros(Float64, BasisSize, BasisSize)
    C = zeros(Float64, BasisSize, BasisSize)
    JmK = zeros(Float64, BasisSize, BasisSize, BasisSize, BasisSize)

    # lists of previous Fock matrices and error matrices for DIIS
    FockList = Vector{Matrix{Float64}}()
    ErrorList = Vector{Matrix{Float64}}()

    evecs = zeros(Float64, BasisSize, 1)
    evals = zeros(Float64, BasisSize, 1)

    Count = 0
    Delta = 1.0

    P = PGuess


    S = OverlapMatrix(Basis, to)
    S_Ortho = Orthogonalize(S, to)

    H = CoreHamiltonian(Basis, to)
    
    JmK = FockMatrixInts(Basis, to)
    
    @timeit to "SCF Procedure" begin
        @showprogress 2 "SCF Converging..." for i in 1:MaxIter
            Count += 1

            # Build new Fock matrix
            F = FockMatrix(Basis, P, H, JmK, to)

            # Orthogonalize new Fock matrix
            @timeit to "Orthogonalize Fock Matrix" begin
                F_Ortho, FockList, ErrorList = DIIS(S_Ortho, S, F, P, FockList, ErrorList)
            end
            
            # Solve eigenvalue problem for orthogonalized Fock matrix
            @timeit to "Eigenvalues and Eigenvectors of F'" begin
                evals, evecs = eigen(Symmetric(F_Ortho))
            end

            # Back-transform eigenvectors to original basis
            @timeit to "Transform C to original basis" begin
                C = S_Ortho * evecs
            end

            # Update density matrix
            P_Old = copy(P)
            @timeit to "Construct Density Matrix" begin
                P = DensityMatrix(C, Basis, Nelec)
            end

            # Compute change in density matrix
            @timeit to "ΔP" begin
                Delta = DeltaP(P, P_Old, Basis)
            end

            @timeit to "Total System Energy" begin
                Energy = 0.0
                Energy = tr(P * (H + F) / 2)
                push!(E, Energy)
            end
            
            if Delta < ConvC
                println("\nSCF Converged in $Count iterations.")
                break
            end
        end
    end

    return E[Count], E, Count, Delta, evecs, evals, JmK, P, S, S_Ortho, F, F_Ortho, C, to
end