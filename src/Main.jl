using TimerOutputs
using PrettyTables
using LinearAlgebra

include("lib/CUHFSCF.jl")
include("lib/Reader.jl")
include("PlotSCF.jl")

function main()
    to = TimerOutput()

    # ── Basis set construction ────────────────────────────────────────────────
    Basis = GenerateBasisSet(ReadAtomBFs("Ti_3F.dat"), 22)

    # ── Electronic structure parameters ───────────────────────────────────────
    occα = [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0]
    occβ = [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0]
    EC   = GenerateEC(occα, occβ)

    # ── Working arrays ───────────────────────────────────────────────────────
    BasisSize = Basis.BasisSize
    J         = zeros(Float64, BasisSize, BasisSize)
    K         = zeros(Float64, BasisSize, BasisSize)
    KM        = zeros(Float64, BasisSize, BasisSize)
    Fα        = zeros(Float64, BasisSize, BasisSize)
    Fβ        = zeros(Float64, BasisSize, BasisSize)
    Fα_Ortho  = zeros(Float64, BasisSize, BasisSize)
    Fβ_Ortho  = zeros(Float64, BasisSize, BasisSize)
    Cα        = zeros(Float64, BasisSize, BasisSize)
    Cβ        = zeros(Float64, BasisSize, BasisSize)
    Pα        = zeros(Float64, BasisSize, BasisSize)
    Pβ        = zeros(Float64, BasisSize, BasisSize)
    P_Old     = zeros(Float64, BasisSize, BasisSize)
    evecsα    = zeros(Float64, BasisSize, BasisSize)
    evecsβ    = zeros(Float64, BasisSize, BasisSize)
    evalsα    = zeros(Float64, BasisSize)
    evalsβ    = zeros(Float64, BasisSize)

    # ── Integral evaluation ───────────────────────────────────────────────────
    S        = OverlapMatrix(Basis, to)
    S_Ortho  = Orthogonalize(S, to)
    H        = CoreHamiltonian(Basis, to)
    H_ortho  = S_Ortho' * H * S_Ortho
    H_evals, H_evecs = eigen(Symmetric(H_ortho))
    C_Init = S_Ortho * H_evecs

    Cα_InitR = C_Init
    Cβ_InitR = C_Init
    Cβ_InitR[14:15, 14:15] *= [cos(π/4) -sin(π/4); sin(π/4) cos(π/4)]
    Pα, Pβ, P_Init, M = DensityMatrices(Cα_InitR, Cβ_InitR, EC)

    # ── SCF procedure ─────────────────────────────────────────────────────────
    max_iter = 50000
    ConvC    = 5.0e-14
    P_Guess  = P_Init

    FinalEnergy, Ei, Count, ΔPα, ΔPβ, FNR, max_R,
    evecsα, evecsβ, evalsα, evalsβ,
    J, K, Pα, Pβ, P, M, S, S_Ortho,
    Fcs, Fα, Fβ, Fα_Ortho, Fβ_Ortho,
    Cα, Cβ,
    to = SCF(Basis, EC, P_Guess, ConvC, max_iter, to)

    print_timer(to)

    println("\n── SCF Results ──────────────────────────────────────────────────")
    println("Final Energy  (Eₕ) :      ", FinalEnergy)
    println("Iterations         :      ", Count < max_iter ? string(Count) : "DID NOT CONVERGE")
    println("Final ΔPα          :      ", ΔPα)
    println("Final ΔPβ          :      ", ΔPβ)
    println("Final RMS of Residual:    ", FNR)
    println("Final absmax of Residual: ", max_R)

    # ── Dashboard ─────────────────────────────────────────────────────────────
    PlotResultsSCF(
        Basis, EC,                       # basis + electronic config
        H_evecs, Cβ_InitR, P_Guess,          # initial guess
        ConvC, Count, max_iter,          # convergence metadata
        ΔPα, ΔPβ, FNR, max_R,            # final convergence parameters
        FinalEnergy, Ei,                 # final energy & per-iteration energies
        evalsα, evalsβ,                  # final eigenvalues
        evecsα, evecsβ,                  # final eigenvectors
        Cα, Cβ,                          # final coefficient matrices
        Fcs, Fα, Fβ, Fα_Ortho, Fβ_Ortho, # Fock matrices
        Pα, Pβ, P, M;  
        output_file = "Data/Output/SCF Results Ti 3F.png"
    )

end

if !isdefined(Base, :test)
    main()
end